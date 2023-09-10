import os
import typing
import datasets
import constants
import accelerate
import transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration, HfArgumentParser
import torch
import tokenizer as tokenizer_lib
import metrics as metrics_lib
from constants import DATASET_VALS
import preprocess
import utils
import flags


if __name__ == "__main__":

    # Parse flags
    parser = HfArgumentParser((flags.ModelArguments, flags.DataTrainingArguments, flags.TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Set the seed for reproducibility
    utils.set_seed(training_args.seed)

    # Initialize accelerator
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=training_args.training_accumulation_steps,
        log_with='wandb',
    )

    # Load the T5 model and tokenizer
    tokenizer = T5Tokenizer.from_pretrained(model_args.tokenizer_name)
    model = T5ForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name_or_path,
    )

    # If the model is a deepspeed checkpoint, load the model state dict from the checkpoint.
    if training_args.deepspeed and os.path.exists(model_args.model_name_or_path):
        accelerator.print(f"Loading model state dict from {model_args.model_name_or_path}")
        model.load_state_dict(
            torch.load(
                os.path.join(model_args.model_name_or_path, 'pytorch_model.bin')
            )
        )

    # The experiment's name is based on the learning rate and the scheduler type. This is to make it easier to
    # compare experiments.
    experiment_name = (f"{model_args.model_name_or_path.split('/')[-1].replace('.', '_')}"
                       f"dataset_{data_args.dataset_name}_"
                       f"lr_{training_args.learning_rate}_"
                       f"scheduler_{training_args.lr_scheduler_type}"
                       )
    output_dir = os.path.join(training_args.output_dir, data_args.benchmark, data_args.dataset_name, experiment_name)
    if accelerator.is_local_main_process and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize W&B tracking
    accelerator.init_trackers(
        project_name='T5 Evaluation',
        config={
            'model_name': model_args.model_name_or_path,
            'output_dir': output_dir,
            'logging_dir': training_args.logging_dir,
            'benchmark': data_args.benchmark,
            'dataset_name': data_args.dataset_name,
            'checkpoint_origin': training_args.checkpoint_origin,
            'training_batch_size': training_args.per_device_train_batch_size,
            'eval_batch_size': training_args.per_device_eval_batch_size,
            'learning_rate': training_args.learning_rate,
            'num_train_epochs': training_args.num_train_epochs,
            'seed': training_args.seed,
            'optimizer': training_args.optimizer,
            'warmup_ratio': training_args.warmup_ratio,
            'weight_decay': training_args.weight_decay,
            'lr_scheduler_type': training_args.lr_scheduler_type,
            'training_accumulation_steps': training_args.training_accumulation_steps,
            'eval_accumulation_steps': training_args.eval_accumulation_steps,
            'input_seq_length': data_args.input_seq_length,
            'target_seq_length': data_args.target_seq_length,
            'num_beams': training_args.beam_search_num_beams,
            'length_penalty': training_args.beam_search_length_penalty,
            'eval_with_beam_search': training_args.eval_with_beam_search,
            'early_stopping_patience': training_args.early_stopping_patience,
            'early_stopping_threshold': training_args.early_stopping_threshold,
            'eval_with_teacher_forcing': training_args.eval_with_teacher_forcing,
        },
        init_kwargs={
            "wandb": {
                "name": experiment_name,
            }
        }
    )

    # Load and pre-process the appropriate dataset/s
    if data_args.dataset_name == 'all':

        # Load all datasets
        accelerator.print("Loading all datasets...")
        training_sets = []
        validation_sets = {}  # Create a validation set for each dataset in the benchmark.
        test_sets = {}  # Create a test set for each dataset in the benchmark.

        for dataset_name in datasets.get_dataset_config_names(data_args.benchmark):

            if dataset_name in data_args.excluded_datasets.split(','):
                continue

            accelerator.print(f"\tLoading {dataset_name}...")
            dataset = datasets.load_dataset(data_args.benchmark, dataset_name)

            # Create a function to pre-process the dataset
            preprocessing_function = preprocess.create_preprocess_function(
                dataset_info=DATASET_VALS[data_args.benchmark][dataset_name],
                dataset_name=dataset_name
            )

            accelerator.print(f"\tPreprocessing {dataset_name}...")
            dataset = dataset.map(
                preprocessing_function,
                batched=True,
                desc=f"Preprocessing {dataset_name}",
                remove_columns=['label'],
            )

            if constants.TRAIN in dataset.keys():
                training_sets.append(dataset[constants.TRAIN])
            if constants.VALIDATION in dataset.keys():
                validation_sets[dataset_name] = dataset[constants.VALIDATION]
            if constants.TEST in dataset.keys():
                test_sets[dataset_name] = dataset[constants.TEST]

        # Combine the datasets
        # Shuffle only the training set.
        accelerator.print("Combining datasets...")
        training_set = datasets.concatenate_datasets(training_sets).shuffle()
        dataset = datasets.DatasetDict({
            constants.TRAIN: training_set,
        })
        for dataset_name, validation_set in validation_sets.items():
            dataset[f'{constants.VALIDATION}_{dataset_name}'] = validation_set
        for dataset_name, test_set in test_sets.items():
            dataset[f'{constants.TEST}_{dataset_name}'] = test_set

    else:
        # Load the specified dataset
        # TODO: MNLI is a special case since it has two validation sets. This needs to be handled separately.
        accelerator.print(f"Loading {data_args.dataset_name}...")
        dataset = datasets.load_dataset(data_args.benchmark, data_args.dataset_name)
        preprocessing_function = preprocess.create_preprocess_function(
            dataset_info=DATASET_VALS[data_args.benchmark][data_args.dataset_name],
            dataset_name=data_args.dataset_name,
        )
        accelerator.print(f"Preprocessing {data_args.dataset_name}...")
        dataset = dataset.map(
            preprocessing_function,
            batched=True,
            desc=f"Preprocessing {data_args.dataset_name}",
        )

    # Tokenize the dataset
    def wrapped_tokenizer_function(examples):
        return tokenizer_lib.tokenize_function(
            examples=examples,
            tokenizer=tokenizer,
            input_column_name='processed_inputs',
            target_column_name='processed_outputs',
            input_max_length=data_args.input_seq_length,
            target_max_length=data_args.target_seq_length,
        )

    encoded_dataset = dataset.map(
        wrapped_tokenizer_function,
        batched=True,
        desc=f"Tokenizing {data_args.dataset_name}",
    )

    accelerator.print(f'column names are: {encoded_dataset.column_names}')

    train_dataset = encoded_dataset[constants.TRAIN]  # There is only one training dataset.
    train_dataset.set_format(type='torch', columns=constants.GLUE_TOKENIZED_COLUMN_NAMES)

    # Create a dictionary mapping dataset names to their respective validation sets.
    if data_args.dataset_name == constants.ALL:
        eval_dataset = {}
        for split in encoded_dataset.keys():
            if split.startswith(f'{constants.VALIDATION}_'):
                dataset_name = split[len(f'{constants.VALIDATION}_'):]
                eval_dataset[dataset_name] = encoded_dataset[split]
                eval_dataset[dataset_name].set_format(type='torch', columns=constants.GLUE_TOKENIZED_COLUMN_NAMES)

        # Create a dictionary mapping dataset names to their respective test sets.
        test_datasets = {}
        for split in encoded_dataset.keys():
            if split.startswith(f'{constants.TEST}_'):
                dataset_name = split[len(f'{constants.TEST}_'):]
                test_datasets[dataset_name] = encoded_dataset[split]
                test_datasets[dataset_name].set_format(type='torch', columns=constants.GLUE_TOKENIZED_COLUMN_NAMES)
                test_datasets[dataset_name].remove_columns([constants.LABELS])  # Remove the labels from the test dataset
    else:
        eval_dataset = encoded_dataset[constants.VALIDATION]
        eval_dataset.set_format(type='torch', columns=constants.GLUE_TOKENIZED_COLUMN_NAMES)
        test_dataset = encoded_dataset[constants.TEST]
        test_dataset.set_format(type='torch', columns=constants.GLUE_TOKENIZED_COLUMN_NAMES)

    # Define the training parameters
    trainer_arguments = transformers.Seq2SeqTrainingArguments(
        # Set up directories
        output_dir=output_dir,
        logging_dir=training_args.logging_dir,

        # Optimization parameters
        deepspeed=training_args.deepspeed_config if training_args.deepspeed else None,
        optim=training_args.optimizer,
        learning_rate=training_args.learning_rate,
        lr_scheduler_type=training_args.lr_scheduler_type,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        per_device_eval_batch_size=training_args.per_device_eval_batch_size,
        warmup_ratio=training_args.warmup_ratio,
        weight_decay=training_args.weight_decay,
        eval_accumulation_steps=training_args.eval_accumulation_steps,

        # Training strategy to adopt
        logging_strategy=constants.STEPS,
        evaluation_strategy=constants.STEPS,
        save_strategy=constants.STEPS,
        num_train_epochs=training_args.num_train_epochs,

        # Early stopping (only used if the dataset is not 'all')
        load_best_model_at_end=False if data_args.dataset_name == constants.ALL else True,
        metric_for_best_model=(
            None if data_args.dataset_name == constants.ALL else
            DATASET_VALS[data_args.benchmark][data_args.dataset_name][constants.METRIC_TO_OPTIMIZE]
        ),
        greater_is_better=(
            None if data_args.dataset_name == constants.ALL else
            DATASET_VALS[data_args.benchmark][data_args.dataset_name][constants.GREATER_IS_BETTER]
        ),

        # Frequency of training callbacks (logging, evaluation, checkpointing, etc.)
        logging_steps=training_args.logging_steps,
        eval_steps=training_args.eval_steps,
        save_steps=training_args.save_steps,
        save_total_limit=training_args.save_total_limit,  # Maintain a finite number of checkpoints

        # Miscellaneous parameters
        ddp_find_unused_parameters=False,

        # Generation parameters (used during evaluation and testing)
        predict_with_generate=not training_args.eval_with_teacher_forcing,
        generation_max_length=training_args.beam_search_max_length,
        generation_num_beams=training_args.beam_search_num_beams,
        generation_config=transformers.GenerationConfig(
            do_sample=training_args.eval_with_beam_search,              # If False, use greedy decoding
            max_length=training_args.beam_search_max_length,            # Maximum length of the generated sequence
            pad_token_id=tokenizer.pad_token_id,                        # Pad token ID
            eos_token_id=tokenizer.eos_token_id,                        # End of sentence token ID
            decoder_start_token_id=tokenizer.pad_token_id,              # Decoder start token ID
            length_penalty=training_args.beam_search_length_penalty if training_args.eval_with_beam_search else 1.0,
        ),
    )

    def compute_metrics(eval_pred: transformers.EvalPrediction) -> typing.Dict[str, float]:
        return metrics_lib.compute_metrics(
            eval_pred=eval_pred,
            metric_names=DATASET_VALS[data_args.benchmark][data_args.dataset_name][constants.METRIC_NAMES],
            padding_token=tokenizer.pad_token_id,
            eos_token=tokenizer.eos_token_id,
        )

    # Train the model
    trainer = transformers.Seq2SeqTrainer(
        model=model,
        args=trainer_arguments,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            label_pad_token_id=training_args.training_padding_token_id,
        ),
        preprocess_logits_for_metrics=(
            metrics_lib.preprocess_logits_for_metrics if training_args.eval_with_teacher_forcing else None
        ),
        callbacks=[
            transformers.trainer_callback.EarlyStoppingCallback(
                early_stopping_patience=training_args.early_stopping_patience,
                early_stopping_threshold=training_args.early_stopping_threshold,
            ),
        ] if data_args.dataset_name != 'all' else None,  # Only use early stopping if the dataset is not 'all'
        # TODO: Create a unified metric (e.g., loss) that can be used for early stopping on all datasets.
    )
    trainer.add_callback(utils.TrainingMetricsCallback(trainer))

    if model_args.model_name_or_path is not None:
        if os.path.isdir(model_args.model_name_or_path) and os.listdir(model_args.model_name_or_path) != []:
            resume_from_checkpoint = model_args.model_name_or_path
        else:
            resume_from_checkpoint = None
    else:
        resume_from_checkpoint = None

    trainer.train(
        # resume_from_checkpoint=resume_from_checkpoint,
    )
    accelerator.print(f"Training completed. Saving model to {output_dir}")

    # Evaluate the model
    if data_args.dataset_name == constants.ALL:
        # Evaluate the model on each dataset in the benchmark
        for dataset_name, eval_dataset in eval_dataset.items():
            accelerator.print(f"Evaluating on validation set: {dataset_name}")
            test_metrics = trainer.evaluate(
                eval_dataset=eval_dataset,
                metric_key_prefix=constants.VALIDATION,
                num_beams=training_args.beam_search_num_beams,
            )

        for dataset_name, test_dataset in test_datasets.items():
            accelerator.print(f"Predicting on test set: {dataset_name}")
            test_predictions = trainer.predict(
                test_dataset=test_dataset,
                metric_key_prefix=constants.TEST,
                num_beams=training_args.beam_search_num_beams,
            )
            if accelerator.is_local_main_process:
                with open(os.path.join(output_dir, f'test_predictions_{dataset_name}.txt'), 'w') as f:
                    for prediction in test_predictions.predictions:
                        f.write(prediction + '\n')
    else:
        # Evaluate the model on the specified dataset
        accelerator.print(f"Evaluating on validation set: {data_args.dataset_name}")
        test_metrics = trainer.evaluate(
            eval_dataset=eval_dataset[data_args.dataset_name],
            metric_key_prefix=constants.VALIDATION,
            num_beams=training_args.beam_search_num_beams,
        )

        accelerator.print(f"Predicting on test set: {data_args.dataset_name}")
        test_predictions = trainer.predict(
            test_dataset=test_datasets[data_args.dataset_name],
            metric_key_prefix=constants.TEST,
            num_beams=training_args.beam_search_num_beams,
        )
        if accelerator.is_local_main_process:
            with open(os.path.join(output_dir, 'test_predictions.txt'), 'w') as f:
                for prediction in test_predictions.predictions:
                    f.write(prediction + '\n')

    # TODO: Enable saving the test predictions to a file alongside the input and target sequences.
    # TODO: Enable writing a GLUE submission file. This might require iteration + model.generate().