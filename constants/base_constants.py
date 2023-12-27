import enum
from dataclasses import dataclass
import typing


class SpecialTokens(enum.Enum):
    """
    Special tokens used by transformer models
    """
    PAD = '<pad>'
    UNK = '<unk>'
    BOS = '<s>'
    EOS = '</s>'
    MASK = '<mask>'
    EOT = '</t>'
    EOL = '</l>'
    SEP = '</p>'
    SPACE = '▁'


class RawExampleColumnNames(enum.Enum):
    """
    Column names for processed examples.
    """
    IDX = 'idx'
    INPUTS = 'processed_inputs'
    OUTPUTS = 'processed_outputs'


class TokenizedExampleColumnNames(enum.Enum):
    """
    Column names for tokenized examples.
    """
    INPUT_IDS = 'input_ids'
    ATTENTION_MASK = 'attention_mask'
    # NOTE: LABELS is used for generative tasks and LABEL is used for classification tasks.
    LABEL = 'label'
    LABELS = 'labels'


@dataclass
class TokenizerConstants:
    """
    Tokenizer constants. These are used to tokenize the inputs and outputs of the T5 model.
    """
    TOKENIZER: str = 'tokenizer'
    IN_LENGTH: str = 'in_length'
    OUT_LENGTH: str = 'out_length'
    T5_START_TOKEN: str = '</s>'
    T5_PAD_TOKEN: str = '<pad>'
    T5_SPACE_TOKEN: str = "▁"


@dataclass
class RawTrainingExampleConstants:
    PREPROCESSED_COLUMN_NAMES: typing.Tuple[str] = ('idx', 'processed_inputs', 'processed_outputs')
    TEXT_COLUMN_NAME: str = 'text'


@dataclass
class TokenizedTrainingExampleConstants:
    TOKENIZED_COLUMN_NAMES: typing.Tuple[str] = ('input_ids', 'attention_mask', 'labels')
    INPUT_IDS: str = 'input_ids'
    TOKEN_TYPE_IDS: str = 'token_type_ids'
    ATTENTION_MASK: str = 'attention_mask'
    LABEL: str = 'label'
    LABELS: str = 'labels'


# TODO: Make this an enum
@dataclass
class MetricConstants:
    # Metric names
    PRECISION: str = 'precision'
    RECALL: str = 'recall'
    F1: str = 'f1'
    ACCURACY: str = 'accuracy'
    MCC: str = 'mcc'
    SPEARMAN: str = 'spearman'
    PEARSON: str = 'pearson'


@dataclass
class ExampleMetricConstants(MetricConstants):
    # Example-level metric names
    EXAMPLE_ACCURACY: str = 'example_accuracy'
    EXAMPLE_F1: str = 'example_f1'
    EXAMPLE_PRECISION: str = 'example_precision'
    EXAMPLE_RECALL: str = 'example_recall'
    EXAMPLE_MCC: str = 'example_mcc'
    EXAMPLE_SPEARMAN: str = 'example_spearman'
    EXAMPLE_PEARSON: str = 'example_pearson'


@dataclass
class TokenMetricConstants(MetricConstants):
    # Token-level metric names
    TOKEN_ACCURACY: str = 'token_accuracy'
    TOKEN_F1: str = 'token_f1'
    TOKEN_PRECISION: str = 'token_precision'
    Token_RECAll: str = 'token_recall'
    TOKEN_MCC: str = 'token_mcc'
    TOKEN_SPEARMAN: str = 'token_spearman'
    TOKEN_PEARSON: str = 'token_pearson'


@dataclass
class SplitConstants:
    # Split Names
    TRAIN: str = 'train'
    VALIDATION: str = 'validation'
    TEST: str = 'test'


@dataclass
class TrainingConstants:
    # Training Parameters
    STEPS = 'steps'
    EPOCHS = 'epochs'
    BATCH_SIZE = 'batch_size'
    LEARNING_RATE = 'learning_rate'
    WARMUP_STEPS = 'warmup_steps'
    WEIGHT_DECAY = 'weight_decay'
    ADAM_EPSILON = 'adam_epsilon'
    MAX_GRAD_NORM = 'max_grad_norm'
    LOGGING_STEPS = 'logging_steps'
    SAVE_STEPS = 'save_steps'
    SAVE_TOTAL_LIMIT = 'save_total_limit'
    EVAL_STEPS = 'eval_steps'
    EVAL_TOTAL_LIMIT = 'eval_total_limit'
    EVAL_ACCUMULATION_STEPS = 'eval_accumulation_steps'
    EVAL_PREDICTION_LOSS_ONLY = 'eval_prediction_loss_only'
    DETERMINISTIC = 'deterministic'
    MAX_SEQUENCE_LENGTH = 'max_sequence_length'


@dataclass
class T5ModelConstants:
    # T5 Model Implementations
    HF_T5 = 'hf_t5'
    LOCAL_T5 = 'local_t5'

    # T5 Model Names
    T5_SMALL = 'google/t5-v1_1-small'  # 60M parameters
    T5_BASE = 'google/t5-v1_1-base'  # 220M parameters
    T5_LARGE = 'google/t5-v1_1-large'  # 770M parameters
    T5_3B = 'google/t5-v1_1-3B'  # 2.8B parameters
    T5_11B = 'google/t5-v1_1-11B'  # 11B parameters


@dataclass
class OptimizerConstants:
    # Optimizers
    ADAMW = 'adamw'
    ADAMWSCALE = 'adamwscale'
    ADAFACTOR = 'adafactor'


class SchedulerConstants:
    NO_DECAY: typing.Tuple[str] = ("bias", "LayerNorm", "layernorm", "layer_norm", "ln")
