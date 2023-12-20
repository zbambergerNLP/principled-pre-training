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
        default=4, metadata={"help": "Batch size per device for evaluation."}
    )
    optimizer: Optional[str] = field(
        default="adamw_torch",
        metadata={
            "help": "The optimizers to use. "
                    "Can be one of ['adamw_hf', 'adamw_torch', 'adamw_torch_fused', 'adamw_torch_xla', "
                    "'adamw_apex_fused', 'adafactor', 'adamw_bnb_8bit', 'adamw_anyprecision', 'sgd', 'adagrad']"
        }
    )
    lr_scheduler_type: Optional[str] = field(
        default="linear", metadata={
            "help": "The LR scheduler to use. Can be 'linear', 'constant', or 'cosine'.\n"
                    "Note that the original T5 paper fine-tunes with a 'constant' LR scheduler, and pre-trains"
                    "with an inverse square ratio scheduler."
        }
    )
    num_train_epochs: int = field(
        default=50, metadata={"help": "Total number of training epochs to perform."}
    )
    learning_rate: float = field(default=1e-3, metadata={
        "help": "The initial learning rate for the optimizer."
                "In T5, the learning rate is set to 1e-3 for both pre-training and fine-tuning.."
    })
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."})

    warmup_ratio: float = field(default=0.1, metadata={"help": "The ratio of warmup steps to total training steps."})
    early_stopping_patience: int = field(default=3, metadata={
        "help": "The number of epochs to wait for the validation loss to improve before early stopping."})
    early_stopping_threshold: float = field(default=0.001, metadata={
        "help": "The threshold for the validation loss to improve by in order to reset the early stopping counter."})
    logging_steps: int = field(default=20, metadata={"help": "Log every X updates steps."})
    eval_accumulation_steps: int = field(
        default=4, metadata={"help": "Number of eval steps to accumulate before performing backward pass."}
    )
    training_accumulation_steps: int = field(
        default=4, metadata={"help": "Number of training steps to accumulate before performing backward pass."}
    )
    training_padding_token_id: int = field(
        default=-100, metadata={"help": "The token ID used to pad the training labels."}
    )
    # TODO: Change the default value of eval_steps to None, when this value is set to None have it infer from
    #  the number of steps when to perform an evaluation.
    eval_steps: int = field(
        default=500,
        metadata={"help": "Number of eval steps to perform before logging metrics."}
    )
    eval_with_teacher_forcing: bool = field(
        default=False,
        metadata={"help": "Whether to use teacher forcing during evaluation."}
    )
    eval_with_beam_search: bool = field(
        default=False,
        metadata={"help": "Whether to use beam search during evaluation."}
    )
    beam_search_num_beams: int = field(
        default=4,
        metadata={"help": "The number of beams to use during beam search decoding."}
    )
    beam_search_max_length: int = field(
        default=64,
        metadata={"help": "The maximum length of the decoded sequence during beam search decoding."}
    )
    beam_search_length_penalty: float = field(
        default=0.6,
        metadata={"help": "The length penalty to use during beam search decoding."}
    )
    save_steps: int = field(default=1_000, metadata={"help": "Save checkpoint every X updates steps."})

    seed: int = field(
        default=42, metadata={"help": "The seed to use for reproducible training."}
    )
    deepspeed: bool = field(
        default=True, metadata={"help": "Whether to use deepspeed for training."}
    )
    deepspeed_config: Optional[str] = field(
        default="zero_stage2_config.json", metadata={"help": "The path to the deepspeed config file."}
    )
    local_rank: int = field(default=-1, metadata={"help": "Local rank for distributed training on GPUs."})
    checkpoint_origin: Optional[str] = field(
        default='pretrained',
        metadata={
            "help": "One of {pretrained, continuous_pretraining, discourse}. Pre-trained checkpoints are "
                    "downloaded from HuggingFace. Continuous pre-training checkpoints are downloaded from local "
                    "storage. Discourse checkpoints are also downloaded from local storage (obtained by in-house"
                    "pre-training)."
        },
    )
    save_total_limit: Optional[int] = field(
        default=None,
        metadata={"help": "If a value is passed, will limit the total amount of checkpoints. Deletes the older "
                          "checkpoints in output_dir. Default to unlimited checkpoints."},
    ),
    pmi: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not to use PMI-Masking"}
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default="google/t5-v1_1-base",
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    tokenizer_name: Optional[str] = field(
        default="google/t5-v1_1-base",
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    dtype: Optional[str] = field(
        default="bfloat16",
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
        default="OfekGlick/DiscoEval",
        metadata={
            "help": "The name of the benchmark to use. "
                    "Can be one of ['squad', 'glue', 'super_glue', 'cnn_dailymail', 'xsum']."}
    )
    dataset_name: Optional[str] = field(
        default="SParxiv",
        metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    excluded_datasets: Optional[str] = field(
        default='ax',
        metadata={"help": "A comma seperated list of datasets within the selected benchmark that are to be ignored "
                          "(if the 'all' option is selected under the 'dataset_name' flag."
                  }
    )
    num_train_examples: Optional[int] = field(
        default=5_000_000, metadata={"help": "The number of training examples to use during training."}
    )
    num_validation_examples: Optional[int] = field(
        default=10_000, metadata={"help": "The number of validation examples to use during training."}
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
        default=512,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization and masking. Sequences longer than this"
                " will be truncated. Default to the max input length of the model."
            )
        },
    )
    tokenized_dataset_dir: Optional[str] = field(
        default="./tokenized_datasets",
        metadata={"help": "The directory where the tokenized datasets will be saved."},
    )
    percent_of_dataset: Optional[int] = field(
        default=100,
        metadata={
            "help": "The percentage of the dataset to use for training. Between 0 and 100. Useful for debugging."
        },
    )
    num_examples: Optional[int] = field(
        default=None,
        metadata={
            "help": "The number of examples to use for training. Useful for debugging. "
                    "An alternative to percent_of_dataset."
        },
    )
    input_seq_length: Optional[int] = field(
        default=512,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization and masking. Sequences longer than this"
                " will be truncated. Default to the max input length of the model."
            )
        },
    )
    target_seq_length: Optional[int] = field(
        default=64,
        metadata={
            "help": (
                "The maximum total target sequence length after tokenization and masking. Sequences longer than this"
                " will be truncated. Default to the max input length of the model."
            )
        },
    )
    pre_training_dataset_paths: Optional[str] = field(
        default="wikipedia,bookcorpus",
        metadata={"help": "The name of the dataset to use for pre-training (via the datasets library)."}
    )
    pre_training_dataset_names: Optional[str] = field(
        default="20220301.en,",  # Only wikipedia has a name, bookcorpus is just a path.
        metadata={"help": "The name of the dataset to use for pre-training (via the datasets library)."}
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for span masked language modeling loss"}
    )
    mean_noise_span_length: float = field(
        default=3.0,
        metadata={"help": "Mean span length of masked tokens"},
    ),
    pmi_vocab_path: Optional[str] = field(
        default=r"pmi_vocab/pmi-wiki-bc.txt",
        metadata={"help": "The path to the PMI vocabulary, should contain a path to a text file, where each row is a "
                          "PMI phrase"}
    )



@dataclass
class AWSArguments:
    aws_config_file_path: Optional[str] = field(
        default=None,
        metadata={"help": "The path to the AWS config file. Used to configure AWS credentials."},
    )
    aws_region: Optional[str] = field(
        default="us-east-1",
        metadata={"help": "The AWS region to use for the SageMaker training job."},
    )
    aws_access_key_id: Optional[str] = field(
        default=None,
        metadata={"help": "The AWS access key ID to use for the SageMaker training job."},
    )
    aws_secret_access_key: Optional[str] = field(
        default=None,
        metadata={"help": "The AWS secret access key to use for the SageMaker training job."},
    )
    aws_role: Optional[str] = field(
        default="SageMaker-Researcher",
        metadata={"help": "The name of the AWS IAM role to use for the SageMaker training job."},
    )
