import os
import random
from constants import DATASET_VALS
import accelerate
import transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration, HfArgumentParser
from datasets import load_dataset
import torch
import flags
import numpy as np
import wandb
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
    model = T5ForConditionalGeneration.from_pretrained(model_args.model_name_or_path)

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

    # Load the appropriate dataset
    dataset = load_dataset(data_args.benchmark, data_args.dataset_name)

    if 'prefix' in DATASET_VALS[data_args.benchmark][data_args.dataset_name].keys():
        wrapped_tokenizer_function = lambda examples: tokenizer_lib.tokenizer_function_one_input(
            examples=examples,
            tokenizer=tokenizer,
            prefix=DATASET_VALS[data_args.benchmark][data_args.dataset_name]['prefix'],
            text_column_name=DATASET_VALS[data_args.benchmark][data_args.dataset_name]['text_column_name'],
            label_column_name=DATASET_VALS[data_args.benchmark][data_args.dataset_name]['label_column_name'],
            label_names=DATASET_VALS[data_args.benchmark][data_args.dataset_name]['labels'],
        )

    elif (
            'prefix_1' in DATASET_VALS[data_args.benchmark][data_args.dataset_name].keys() and
            'prefix_2' in DATASET_VALS[data_args.benchmark][data_args.dataset_name].keys()
    ):
        is_regression = data_args.dataset_name == 'stsb'
        wrapped_tokenizer_function = lambda examples: tokenizer_lib.tokenizer_function_two_input(
            examples=examples,
            tokenizer=tokenizer,
            prefix_1=DATASET_VALS[data_args.benchmark][data_args.dataset_name]['prefix_1'],
            prefix_2=DATASET_VALS[data_args.benchmark][data_args.dataset_name]['prefix_2'],
            text_column_name_1=DATASET_VALS[data_args.benchmark][data_args.dataset_name]['text_column_name_1'],
            text_column_name_2=DATASET_VALS[data_args.benchmark][data_args.dataset_name]['text_column_name_2'],
            label_column_name=DATASET_VALS[data_args.benchmark][data_args.dataset_name]['label_column_name'],
            label_names=DATASET_VALS[data_args.benchmark][data_args.dataset_name]['labels'],
            is_regression=is_regression,
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
    )

    # Define the training parameters
    trainer_arguments = transformers.Seq2SeqTrainingArguments(
        # Set up directories
        output_dir=training_args.output_dir,
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
        greater_is_better=True,  # TODO: In the case of GLUE, this is always true. Need to configure later.

        # Frequency of training callbacks (logging, evaluation, checkpointing, etc.)
        logging_steps=training_args.logging_steps,
        eval_steps=training_args.eval_steps,
        save_steps=training_args.save_steps,
        save_total_limit=3,  # Maintain a finite number of checkpoints # TODO: Make this a flag
    )

    compute_metrics = lambda eval_pred: metrics_lib.compute_metrics(
        eval_pred=eval_pred,
        metric_names=DATASET_VALS[data_args.benchmark][data_args.dataset_name]['metric_names'],
        padding_token=tokenizer.pad_token_id,
        eos_token=tokenizer.eos_token_id,
    )
    # Train the model
    trainer = transformers.Seq2SeqTrainer(
        model=model,
        args=trainer_arguments,
        compute_metrics=compute_metrics,
        train_dataset=encoded_dataset['train'],
        eval_dataset=encoded_dataset['validation'],
        preprocess_logits_for_metrics=metrics_lib.preprocess_logits_for_metrics,
        callbacks=[
            transformers.trainer_callback.EarlyStoppingCallback(
                early_stopping_patience=training_args.patience)
        ],
    )

    if model_args.model_name_or_path is not None:
        if os.path.isdir(model_args.model_name_or_path):
            resume_from_checkpoint = model_args.model_name_or_path
        else:
            resume_from_checkpoint = None
    else:
        resume_from_checkpoint = None

    trainer.train(
        resume_from_checkpoint=resume_from_checkpoint,
    )
    accelerator.print(f"Training completed. Saving model to {training_args.output_dir}")

    # Evaluate the model
    validation_metrics = trainer.evaluate(
        eval_dataset=encoded_dataset['validation'],
        metric_key_prefix='validation',
    )
    accelerator.print(f"Validation metrics: {validation_metrics}")
    test_metrics = trainer.evaluate(
        eval_dataset=encoded_dataset['test'],
        metric_key_prefix='test',
    )
    accelerator.print(f"Test metrics: {test_metrics}")
