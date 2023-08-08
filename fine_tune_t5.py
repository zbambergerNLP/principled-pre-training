# Import necessary libraries
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_dataset
from typing import Dict
import torch

# Load the T5 model and tokenizer
model_name = 'google/t5-v1_1-small'
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Load the SST2 dataset
dataset = load_dataset('glue', 'sst2')


# Preprocess the data
def preprocess_function(examples: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
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
        return_tensors="pt")['input_ids']}

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


# Preprocess the datasets
encoded_dataset = dataset.map(preprocess_function, batch_size=4, batched=True)

# Define the training parameters
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

# Train the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset['train'],
    eval_dataset=encoded_dataset['validation'],
)
trainer.train()

# Evaluate the model
trainer.evaluate()
