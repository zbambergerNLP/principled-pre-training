# Import necessary libraries
import os
import random

import accelerate
import transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration, HfArgumentParser
from datasets import load_dataset
from typing import Dict
import torch
import flags
import numpy as np
import wandb
import sklearn


# Preprocess the data
def tokenizer_function(
        examples: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Preprocess the SST2 examples for the T5 model.

    Args:
        examples: A dictionary of torch.Tensor with the input data.

    Returns:
        A dictionary of torch.Tensor with the preprocessed data.
    """
    # T5 expects the task to be in the input so prepend 'sst2 sentence: ' to each example
    inputs = ['sst2 sentence: ' + sentence for sentence in examples['sentence']]
    results = {'input_ids': tokenizer(
        inputs,
        padding='max_length',
        max_length=512,
        truncation=True,
        return_tensors="pt",
    )['input_ids']}

    # Labels are not preprocessed for the T5 model. model_inputs are returned as is
    outputs = ['positive' if example else 'negative' for example in examples['label']]
    results['labels'] = tokenizer(
        outputs,
        padding='max_length',
        max_length=512,
        truncation=True,
        return_tensors="pt",
    )['input_ids']
    return results


def compute_metrics(eval_pred: transformers.EvalPrediction):
    """Compute the accuracy of the model.

    Args:
        eval_pred: A namedtuple containing the model predictions and labels.

    Returns:
        A dictionary containing the accuracy of the model.
    """
    predictions, labels = eval_pred
    predictions: np.ndarray
    labels: np.ndarray

    # Flatten the predictions and labels
    predictions = predictions.flatten()
    labels = labels.flatten()

    metrics = {
        "accuracy": sklearn.metrics.accuracy_score(
            y_true=labels,
            y_pred=predictions,
        ),
        # TODO: Add the metrics below, but note that predictions are multi-class since the model is seq2seq.
        # "precision": sklearn.metrics.precision_score(
        #     y_true=labels,
        #     y_pred=predictions,
        # ),
        # "recall": sklearn.metrics.recall_score(
        #     y_true=labels,
        #     y_pred=predictions,
        # ),
        # "f1": sklearn.metrics.f1_score(
        #     y_true=labels,
        #     y_pred=predictions,
        # ),
    }
    return metrics


def preprocess_logits_for_metrics(
        logits: torch.Tensor,
        labels: torch.Tensor,
) -> torch.Tensor:
    """
    Original Trainer may have a memory leak.

    This is a workaround to avoid storing too many tensors that are not needed.
    """
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    return logits.argmax(dim=-1)


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

    # Preprocess the datasets
    encoded_dataset = dataset.map(
        tokenizer_function,  # Tokenizes the dataset
        batch_size=training_args.per_device_train_batch_size,
        batched=True,
    )

    # Calculate warmup steps from warmup ratio
    warmup_steps = (
        int(
            training_args.warmup_ratio *
            training_args.num_train_epochs *
            len(encoded_dataset['train']) /
            training_args.per_device_train_batch_size
        )
    )

    # TODO: Ignore padding tokens when computing metrics and loss
    # Define the training parameters
    trainer_arguments = transformers.Seq2SeqTrainingArguments(
        # Set up directories
        output_dir=training_args.output_dir,
        logging_dir=training_args.logging_dir,

        # Optimization parameters
        optim=training_args.optimizer,
        learning_rate=training_args.learning_rate,
        lr_scheduler_type=training_args.lr_scheduler_type,
        auto_find_batch_size=True,  # Automatically find the batch size that fits on the GPU
        warmup_steps=warmup_steps,
        weight_decay=training_args.weight_decay,
        gradient_accumulation_steps=training_args.training_accumulation_steps,
        eval_accumulation_steps=training_args.eval_accumulation_steps,

        # Training strategy to adopt
        logging_strategy="steps",
        evaluation_strategy="steps",
        save_strategy="steps",
        num_train_epochs=training_args.num_train_epochs,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,

        # Frequency of training callbacks (logging, evaluation, checkpointing, etc.)
        logging_steps=training_args.logging_steps,
        eval_steps=training_args.eval_steps,
        save_steps=training_args.save_steps,
    )

    # Train the model
    trainer = transformers.Seq2SeqTrainer(
        model=model,
        args=trainer_arguments,
        compute_metrics=compute_metrics,
        train_dataset=encoded_dataset['train'],
        eval_dataset=encoded_dataset['validation'],
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
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

    # Evaluate the model
    trainer.evaluate(
        eval_dataset=encoded_dataset['validation'],
        metric_key_prefix='validation',
    )
    trainer.evaluate(
        eval_dataset=encoded_dataset['test'],
        metric_key_prefix='test',
    )
