# Create flags in order to run fine_tune_bert.py and fine_tune_t5.py

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainingArguments:
    run_name: Optional[str] = field(
        default="model_evaluation",
        metadata={"help": "The name of the run. Used for logging and saving checkpoints."},
    )
    local_checkpoint_path: Optional[str] = field(
        default=None,
        metadata={"help": "The path to a local checkpoint. Used to resume training."},
    )
    output_dir: Optional[str] = field(
        default="./outputs",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    logging_dir: Optional[str] = field(
        default="./logs",
        metadata={"help": "The output directory where the logs will be written."},
    )
    last_run_id: Optional[str] = field(
        default=None,
        metadata={"help": "The ID of the last run. Used to resume training."},
    )
    per_device_train_batch_size: int = field(
        default=4, metadata={"help": "Batch size per device during training."}
    )
    per_device_eval_batch_size: int = field(
        default=64, metadata={"help": "Batch size per device for evaluation."}
    )
    optimizer: Optional[str] = field(
        default="adamw_torch",
        metadata={
            "help": "The optimizer to use. "
                    "Can be one of ['adamw_hf', 'adamw_torch', 'adamw_torch_fused', 'adamw_torch_xla', "
                    "'adamw_apex_fused', 'adafactor', 'adamw_bnb_8bit', 'adamw_anyprecision', 'sgd', 'adagrad']"
        }
    )
    lr_scheduler_type: Optional[str] = field(
        default="linear", metadata={"help": "The LR scheduler to use. Can be 'linear' or 'cosine'."}
    )
    num_train_epochs: int = field(
        default=3, metadata={"help": "Total number of training epochs to perform."}
    )
    learning_rate: float = field(default=3e-4, metadata={"help": "The initial learning rate for AdamW."})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."})
    warmup_ratio: float = field(default=0.1, metadata={"help": "The ratio of warmup steps to total training steps."})
    patience: int = field(default=3, metadata={"help": "The number of epochs to wait for the validation loss to"
                                                       " improve before early stopping."})
    logging_steps: int = field(default=50, metadata={"help": "Log every X updates steps."})
    eval_accumulation_steps: int = field(
        default=4, metadata={"help": "Number of eval steps to accumulate before performing backward pass."}
    )
    training_accumulation_steps: int = field(
        default=4, metadata={"help": "Number of training steps to accumulate before performing backward pass."}
    )
    eval_steps: int = field(
        default=200,
        metadata={"help": "Number of eval steps to perform before logging metrics."}
    )
    save_steps: int = field(default=1_000, metadata={"help": "Save checkpoint every X updates steps."})

    seed: int = field(
        default=42, metadata={"help": "The seed to use for reproducible training."}
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default="google/t5-v1_1-small",
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    tokenizer_name: Optional[str] = field(
        default="google/t5-v1_1-small",
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    dtype: Optional[str] = field(
        default="float16",
        metadata={
            "help": (
                "Floating-point format in which the model weights should be initialized and trained. Choose one of"
                " `[float32, float16, bfloat16]`."
            )
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    benchmark: Optional[str] = field(
        default="glue",
        metadata={
            "help": "The name of the benchmark to use. "
                    "Can be one of ['squad', 'glue', 'super_glue', 'cnn_dailymail', 'xsum']."}
    )
    dataset_name: Optional[str] = field(
        default="sst2",
        metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    num_train_examples: Optional[int] = field(
        default=None, metadata={"help": "The number of training examples to use during training."}
    )
    num_validation_examples: Optional[int] = field(
        default=None, metadata={"help": "The number of validation examples to use during training."}
    )
    text_column_name: Optional[str] = field(
        default="text", metadata={"help": "The name of the text column in the selected dataset"}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization and masking. Sequences longer than this"
                " will be truncated. Default to the max input length of the model."
            )
        },
    )
