# Import necessary libraries
import os
import random
import typing
from constants import DATASET_VALS
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
def tokenizer_function_one_input(
        examples: Dict[str, typing.Any],
        label_names: typing.Dict[int, str],
        prefix: str,
        text_column_name: str = 'sentence',
        label_column_name: str = 'label',
) -> Dict[str, torch.Tensor]:
    """
    Tokenizes batches of examples with only a single textual input for an encoder-decoder model.

    Args:
        examples: A batch in the form of a dictionary mapping, mapping column names to their respective values.
        label_names: A dictionary mapping from the integer representation of the label to the string representation.
        prefix: The string prefix prepended to each textual example. (This is task specific)
        text_column_name: Name of the column within the input dictionary that contains the text which will be tokenized.
        label_column_name: Name of the column within the input dictionary that contains the labels which will be
            tokenized.

    Returns:
        A dictionary containing the original mappings, as well as the mapping between model input names (e.g.,
            `input_ids`) and model input values (e.g., the tensor corresponding to the input IDs of the model).
    """
    inputs = [f"{prefix}{sentence}" for sentence in examples[text_column_name]]
    results = {'input_ids': tokenizer(
        inputs,
        padding='max_length',
        max_length=512,
        truncation=True,
        return_tensors="pt",
    )['input_ids']}

    # Labels are not preprocessed for the T5 model. model_inputs are returned as is
    outputs = [label_names[example] for example in examples[label_column_name]]
    labels = tokenizer(
        outputs,
        padding='max_length',
        max_length=512,
        truncation=True,
        return_tensors="pt",
    )['input_ids']

    # Replace the padding token with -100 to ignore it for loss computation
    labels[labels == tokenizer.pad_token_id] = -100
    results['labels'] = labels
    return results


def compute_metrics(
        eval_pred: transformers.EvalPrediction,
) -> Dict[str, float]:
    """Compute the accuracy of the model.

    Args:
        eval_pred: A namedtuple containing the model predictions and labels.

    Returns:
        A dictionary containing the accuracy of the model.
    """
    predictions, labels = eval_pred
    predictions: np.ndarray
    labels: np.ndarray

    # Flatten the predictions and labels. Ignore the padding tokens (-100)
    # TODO: Ignore EOS tokens as well
    predictions = predictions[labels != -100].flatten()
    labels = labels[labels != -100].flatten()

    metrics = {
        "eval_accuracy": sklearn.metrics.accuracy_score(
            y_true=labels,
            y_pred=predictions,
        ),
        "precision": sklearn.metrics.precision_score(
            y_true=labels,
            y_pred=predictions,
            average='micro',
        ),
        "recall": sklearn.metrics.recall_score(
            y_true=labels,
            y_pred=predictions,
            average='micro',
        ),
        "f1": sklearn.metrics.f1_score(
            y_true=labels,
            y_pred=predictions,
            average='micro',
        ),
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

    wrapped_tokenizer_function_one_input = lambda examples: tokenizer_function_one_input(
        examples=examples,
        prefix=DATASET_VALS[data_args.benchmark][data_args.dataset_name]['prefix'],
        label_names=DATASET_VALS[data_args.benchmark][data_args.dataset_name]['labels'])

    # Preprocess the datasets
    encoded_dataset = dataset.map(
        wrapped_tokenizer_function_one_input,  # Tokenizes the dataset
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
        auto_find_batch_size=True,  # Automatically find the batch size that fits on the GPU
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
        metric_for_best_model="accuracy",
        greater_is_better=True,

        # Frequency of training callbacks (logging, evaluation, checkpointing, etc.)
        logging_steps=training_args.logging_steps,
        eval_steps=training_args.eval_steps,
        save_steps=training_args.save_steps,
        save_total_limit=3,  # Maintain a finite number of checkpoints # TODO: Make this a flag
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
