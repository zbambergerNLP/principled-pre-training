import os

import datasets as datasets_lib
import torch.cuda
import transformers
import accelerate

import flags
import metrics as metrics_lib
import data_collator_t5
import tokenizer as tokenizer_lib
import wandb


def main():

    # Parse flags
    parser = transformers.HfArgumentParser(
        (flags.ModelArguments, flags.DataTrainingArguments, flags.TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=training_args.training_accumulation_steps,
        log_with='wandb',
    )

    # Initialize accelerator
    accelerator.print('training_args:', training_args)
    accelerator.print('model_args:', model_args)
    accelerator.print('data_args:', data_args)

    accelerator.print(f'Started pre-training a {model_args.model_name_or_path} model')

    # Load the T5 model and tokenizer
    tokenizer = transformers.T5Tokenizer.from_pretrained(model_args.tokenizer_name)
    accelerator.print(f'Loaded tokenizer {model_args.tokenizer_name}')
    model = transformers.T5ForConditionalGeneration.from_pretrained(model_args.model_name_or_path)
    accelerator.print(f'Loaded model {model_args.model_name_or_path}')

    # TODO: Define wandb run initialization as a function, and call it below (to avoid code duplication).
    # Initialize W&B project
    if accelerator.is_local_main_process:
        wandb.init(
            project='T5 Evaluation',
            config={
                'model_name': model_args.model_name_or_path,
                'output_dir': training_args.output_dir,
                'logging_dir': training_args.logging_dir,
                'dataset_name': data_args.dataset_name,
                'batch_size': training_args.per_device_train_batch_size,
                'learning_rate': training_args.learning_rate,
                'num_train_epochs': training_args.num_train_epochs,
                'seed': training_args.seed,
                'optimizer': training_args.optimizer,
                'warmup_ratio': training_args.warmup_ratio,
                'weight_decay': training_args.weight_decay,
                'lr_scheduler_type': training_args.lr_scheduler_type,
            },
        )
        wandb.watch(model, log='all')

    # Download the dataset
    # See https://huggingface.co/docs/datasets/v1.2.1/package_reference/loading_methods.html#datasets.load_dataset
    assert data_args.pre_training_dataset_paths is not None and data_args.pre_training_dataset_names is not None, (
        'Must specify both dataset_paths and dataset_names'
    )
    assert (
        len(data_args.pre_training_dataset_paths.split(',')) == len(data_args.pre_training_dataset_names.split(','))), (
        'Must specify the same number of dataset_paths and dataset_names'
    )
    dataset_paths = data_args.pre_training_dataset_paths.split(',')  # Passed in as a comma-separated list
    dataset_names = data_args.pre_training_dataset_names.split(',')  # Passed in as a comma-separated list

    tokenized_dataset_name = f'{"_".join(dataset_paths)}_tokenized'
    tokenized_dataset_dir = data_args.tokenized_dataset_dir
    tokenized_dataset_path = os.path.join(tokenized_dataset_dir, tokenized_dataset_name)

    if not os.path.exists(tokenized_dataset_dir):
        accelerator.print(f'Creating directory {tokenized_dataset_dir}')
        os.makedirs(tokenized_dataset_dir)

    if tokenized_dataset_name in os.listdir(tokenized_dataset_dir):
        accelerator.print('Loading tokenized dataset...')
        accelerator.print(f'dataset name: {tokenized_dataset_name}, dataset path: {tokenized_dataset_path}')
        tokenized_dataset = datasets_lib.load_from_disk(tokenized_dataset_path)

    else:  # Tokenize the dataset
        accelerator.print('Did not find tokenized dataset. Loading and tokenizing dataset...')
        datasets = []
        for dataset_path, dataset_name in zip(dataset_paths, dataset_names):

            # Determine the split name. This determines which portion/how many examples of the dataset to load.
            if data_args.percent_of_dataset:
                split_name = f'train[:{data_args.percent_of_dataset}%]'
            elif data_args.num_examples:
                split_name = f'train[:{data_args.num_examples}]'
            else:
                split_name = 'train'

            # Names specify the dataset version (e.g., '20210301.en', the English version of Wikipedia from
            # March 1, 2021). Paths specify the dataset type (e.g., 'wikipedia', the Wikipedia dataset).
            name = dataset_name if dataset_name != '' else None
            if name is None:
                dataset = datasets_lib.load_dataset(
                    path=dataset_path,
                    split=split_name,
                    # streaming=True,  # TODO: Fix pre-training with streaming datasets
                )
            else:
                dataset = datasets_lib.load_dataset(
                    path=dataset_path,
                    name=dataset_name,
                    split=split_name,
                    # streaming=True,  # TODO: Fix pre-training with streaming datasets
                )
            # TODO: Consider removing non text columns from each dataset. Might save on memory. See example below:
            # wiki = wiki.remove_columns([col for col in wiki.column_names if col != "text"])

            # TODO: Add a flag to specify whether to save the dataset before tokenizing it. Not recommended for large
            #  datasets.
            # accelerator.print(
            #     f'Loaded dataset {dataset_path}/{dataset_name} '
            #     f'and saved to {os.path.join(tokenized_dataset_dir, dataset_path)}'
            # )
            # dataset.save_to_disk(os.path.join(tokenized_dataset_dir, dataset_path))
            datasets.append(dataset)
        dataset = datasets_lib.interleave_datasets(datasets)
        accelerator.print(f'Tokenizing dataset interleaved from: {dataset_paths}')

        # Tokenize the dataset
        t5_pre_training_tokenizer_function = lambda examples: tokenizer_lib.tokenizer_function_t5_pre_training(
            examples=examples,
            tokenizer=tokenizer,
            text_column_name=data_args.text_column_name,
        )
        tokenized_dataset = dataset.map(
            t5_pre_training_tokenizer_function,
            batched=True,
            batch_size=training_args.per_device_train_batch_size,
        )
        accelerator.print(f'Saving tokenized dataset to {tokenized_dataset_path}')
        tokenized_dataset.save_to_disk(tokenized_dataset_path)

    # Split the dataset into training and validation sets
    if isinstance(tokenized_dataset, datasets_lib.Dataset):
        dataset_dict = tokenized_dataset.train_test_split(
            test_size=data_args.validation_split_percentage / 100,
            shuffle=True,
            seed=training_args.seed,
        )
        training_set = dataset_dict['train']
        validation_set = dataset_dict['test']
    elif isinstance(tokenized_dataset, datasets_lib.DatasetDict):
        training_set = tokenized_dataset['train']
        validation_set = tokenized_dataset['test']
    elif isinstance(tokenized_dataset, datasets_lib.IterableDataset):

        # Split the dataset into training and validation sets
        # TODO: Support splitting IterableDatasets via percentage, not just number of examples.
        num_train_examples = data_args.num_train_examples
        num_validation_examples = data_args.num_validation_examples

        accelerator.print(f'Number of training examples: {num_train_examples}. '
                          f'Number of validation examples: {num_validation_examples}')
        device_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
        num_steps = (
                (num_train_examples * training_args.num_train_epochs) //  # Number of examples to train on.
                (device_count * training_args.per_device_train_batch_size)  # Number of examples per step.
        )
        accelerator.print(f'Number of training steps: {num_steps}')
        training_set = tokenized_dataset.take(data_args.num_train_examples)
        validation_set = tokenized_dataset.skip(data_args.num_train_examples)

    else:
        raise RuntimeError(f'Unsupported dataset type {type(tokenized_dataset)}')

    accelerator.print('Initializing trainer...')
    trainer_arguments = transformers.Seq2SeqTrainingArguments(
        # Set up directories
        output_dir=training_args.output_dir,
        logging_dir=training_args.logging_dir,
        deepspeed=training_args.deepspeed_config,

        # Optimization parameters
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        per_device_eval_batch_size=training_args.per_device_eval_batch_size,
        optim=training_args.optimizer,
        learning_rate=training_args.learning_rate,
        lr_scheduler_type=training_args.lr_scheduler_type,
        # TODO: Uncomment the following line when the training progress bar adapts to the batch size.
        # auto_find_batch_size=True,  # Automatically find the batch size that fits on the GPU
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
        metric_for_best_model='eval/loss',
        greater_is_better=False,

        # Frequency of training callbacks (logging, evaluation, checkpointing, etc.)
        # max_steps=num_steps,
        logging_steps=training_args.logging_steps,
        eval_steps=training_args.eval_steps,
        save_steps=training_args.save_steps,
        save_total_limit=3,  # Maintain a finite number of checkpoints # TODO: Make this a flag
    )

    accelerator.print('Training...')

    data_collator = data_collator_t5.T5DataCollator(
        tokenizer=tokenizer,
        noise_density=data_args.mlm_probability,
        mean_noise_span_length=data_args.mean_noise_span_length,
        input_length=data_args.input_seq_length,
        target_length=data_args.target_seq_length,
        pad_token_id=model.config.pad_token_id,
        decoder_start_token_id=model.config.decoder_start_token_id,
        seed=training_args.seed,
    )

    compute_metrics = lambda eval_pred: metrics_lib.compute_metrics(
        eval_pred=eval_pred,
        metric_names=['accuracy'],
        padding_token=tokenizer.pad_token_id,
    )

    trainer = transformers.Seq2SeqTrainer(
        model=model,
        args=trainer_arguments,
        # Add with_format('torch') to avoid a ValueError.
        train_dataset=training_set.with_format('torch'),
        eval_dataset=validation_set.with_format('torch'),
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=metrics_lib.preprocess_logits_for_metrics,
        data_collator=data_collator,
        callbacks=[
            transformers.trainer_callback.EarlyStoppingCallback(
                early_stopping_patience=training_args.patience)
        ],
    )
    # TODO: Investigate why training freezes during evaluation loop when using IterableDataset.
    trainer.train()


if __name__ == '__main__':
    main()
