import os
import random

import typing

import constants
from constants import DATASET_VALS
import accelerate
import transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration, HfArgumentParser
from datasets import load_dataset
import torch
import flags
import numpy as np
import tokenizer as tokenizer_lib
import metrics as metrics_lib


def set_seed(seed: int):
    """Set the seed for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    transformers.set_seed(seed)


if __name__ == "__main__":

    # Parse flags
    parser = HfArgumentParser((flags.ModelArguments, flags.DataTrainingArguments, flags.TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Set the seed for reproducibility
    set_seed(training_args.seed)

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
    experiment_name = (f"t5_{training_args.checkpoint_origin}_"
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
            'early_stopping_patience': training_args.patience,
        },
        init_kwargs={
            "wandb": {
                "name": experiment_name,
                "project": "T5 Evaluation",
                "group": f"{data_args.benchmark}/{data_args.dataset_name}",
            }
        }
    )

    # Load the appropriate dataset
    dataset = load_dataset(data_args.benchmark, data_args.dataset_name)

    if 'prefix' in DATASET_VALS[data_args.benchmark][data_args.dataset_name].keys():
        def wrapped_tokenizer_function(examples):
            return tokenizer_lib.tokenizer_function_one_input(
                examples=examples,
                tokenizer=tokenizer,
                prefix=DATASET_VALS[data_args.benchmark][data_args.dataset_name]['prefix'],
                text_column_name=DATASET_VALS[data_args.benchmark][data_args.dataset_name]['text_column_name'],
                label_column_name=DATASET_VALS[data_args.benchmark][data_args.dataset_name]['label_column_name'],
                label_names=DATASET_VALS[data_args.benchmark][data_args.dataset_name]['labels'],
                input_max_length=data_args.input_seq_length,
                target_max_length=data_args.target_seq_length,
            )

    elif (
            'prefix_1' in DATASET_VALS[data_args.benchmark][data_args.dataset_name].keys() and
            'prefix_2' in DATASET_VALS[data_args.benchmark][data_args.dataset_name].keys()
    ):
        is_regression = data_args.dataset_name == 'stsb'

        def wrapped_tokenizer_function(examples):
            return tokenizer_lib.tokenizer_function_two_input(
                examples=examples,
                tokenizer=tokenizer,
                prefix_1=DATASET_VALS[data_args.benchmark][data_args.dataset_name]['prefix_1'],
                prefix_2=DATASET_VALS[data_args.benchmark][data_args.dataset_name]['prefix_2'],
                text_column_name_1=DATASET_VALS[data_args.benchmark][data_args.dataset_name]['text_column_name_1'],
                text_column_name_2=DATASET_VALS[data_args.benchmark][data_args.dataset_name]['text_column_name_2'],
                label_column_name=DATASET_VALS[data_args.benchmark][data_args.dataset_name]['label_column_name'],
                label_names=DATASET_VALS[data_args.benchmark][data_args.dataset_name]['labels'],
                is_regression=is_regression,
                input_max_length=data_args.input_seq_length,
                target_max_length=data_args.target_seq_length,
            )

    else:
        raise RuntimeError(
            "Unsupported prefix structure. Must contain either `prefix` for single input tasks or `prefix_1` and "
            "`prefix_2` for two input tasks"
        )

    # Preprocess the datasets
    encoded_dataset = dataset.map(
        wrapped_tokenizer_function,  # Tokenizes the dataset
        batch_size=training_args.per_device_train_batch_size,
        batched=True,
        num_proc=torch.cuda.device_count() if torch.cuda.is_available() else 1,
    )

    # Define the training parameters
    trainer_arguments = transformers.Seq2SeqTrainingArguments(
        # Set up directories
        output_dir=output_dir,
        logging_dir=training_args.logging_dir,

        # Optimization parameters
        optim=training_args.optimizer,
        learning_rate=training_args.learning_rate,
        lr_scheduler_type=training_args.lr_scheduler_type,
        # auto_find_batch_size=True,  # Automatically find the batch size that fits on the GPU
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        per_device_eval_batch_size=training_args.per_device_eval_batch_size,
        warmup_ratio=training_args.warmup_ratio,
        weight_decay=training_args.weight_decay,
        gradient_accumulation_steps=training_args.training_accumulation_steps,
        eval_accumulation_steps=training_args.eval_accumulation_steps,

        # Training strategy to adopt
        logging_strategy="steps",
        evaluation_strategy="steps",
        save_strategy="steps",
        num_train_epochs=training_args.num_train_epochs,
        load_best_model_at_end=True,
        metric_for_best_model=DATASET_VALS[data_args.benchmark][data_args.dataset_name]['metric_to_optimize'],
        greater_is_better=DATASET_VALS[data_args.benchmark][data_args.dataset_name]['greater_is_better'],

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
            metric_names=DATASET_VALS[data_args.benchmark][data_args.dataset_name]['metric_names'],
            padding_token=tokenizer.pad_token_id,
            eos_token=tokenizer.eos_token_id,
        )

    # Set up the datasets for training and evaluation
    train_dataset = encoded_dataset['train']
    eval_dataset = encoded_dataset['validation']
    test_dataset = encoded_dataset['test']
    train_dataset.set_format(type='torch', columns=constants.COLUMN_NAMES)
    eval_dataset.set_format(type='torch', columns=constants.COLUMN_NAMES)
    test_dataset.set_format(type='torch', columns=constants.COLUMN_NAMES)

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
            transformers.trainer_callback.EarlyStoppingCallback(early_stopping_patience=training_args.patience),
        ],
    )

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
    validation_metrics = trainer.evaluate(
        eval_dataset=eval_dataset,
        metric_key_prefix='validation',
        num_beams=training_args.beam_search_num_beams,
    )
    accelerator.print(f"Validation metrics: {validation_metrics}")

    # Predict on the test dataset
    # TODO: Enable saving the test predictions to a file alongside the input and target sequences.
    # TODO: Enable writing a GLUE submission file.
    test_dataset.remove_columns(['labels'])  # Remove the labels from the test dataset
    test_predictions = trainer.predict(
        test_dataset=test_dataset,
        metric_key_prefix='test',
        num_beams=training_args.beam_search_num_beams,
    )

    # Decode the predictions
    test_predictions = tokenizer.batch_decode(
        test_predictions.predictions,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    # Save the predictions
    accelerator.print(f"Saving predictions to {output_dir}")
    if accelerator.is_local_main_process:
        with open(os.path.join(output_dir, 'test_predictions.txt'), 'w') as f:
            for prediction in test_predictions:
                f.write(prediction + '\n')