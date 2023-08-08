import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset


# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['sentence'], padding='max_length', truncation=True, return_tensors="pt")


if __name__ == "__main__":
    dataset = load_dataset('glue', 'sst2')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    print(tokenized_datasets)
    for name, dataset in tokenized_datasets.items():
        dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)  # SST2 has 2 classes: positive and negative
    training_args = TrainingArguments(
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        evaluation_strategy="epoch",
        logging_dir='./logs',
        logging_steps=10,
        do_train=True,
        do_eval=True,
        output_dir='./results',
        save_total_limit=2,
        remove_unused_columns=False  # Important for not removing 'label' column
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer
    )

    trainer.train()
    results = trainer.evaluate()
    print(results)




