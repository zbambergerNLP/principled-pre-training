import dataclasses
import enum
import typing


class MonitoringPlatform(enum.Enum):
    """Monitoring platform."""
    WANDB = "wandb"
    NEPTUNE = "neptune"

class Device(enum.Enum):
    """Device to train on."""
    CPU = "cpu"
    GPU = "gpu_cluster"

class TrainingPhase(enum.Enum):
    """Training phase."""
    FT = "ft"
    PT = "pt"

class NumericalPrecision(enum.Enum):
    """Numerical precision."""
    FP32 = "fp32"
    BF16 = "bf16"

class ModelImplementation(enum.Enum):
    """Model implementations."""
    LOCAL_T5 = "local_t5"
    HUGGINGFACE_T5 = "hf_t5"

class EnvironmentVariable(enum.Enum):
    """Environment variables."""
    SLURM_JOB_ID = "SLURM_JOB_ID"

class DatasetSplit(enum.Enum):
    """Dataset split."""
    TRAIN = "train"
    TEST = "test"
    VALIDATION = "validation"

class Optimizer(enum.Enum):
    """Optimizer constants."""
    ADAMW: str = 'adamw'
    ADAMWSCALE: str = 'adamwscale'
    ADAFACTOR: str = 'adafactor'

class Scheduler(enum.Enum):
    """Scheduler constants."""
    CONSTANT: str = 'constant'
    COSINE: str = 'cosine'
    LEGACY: str = 'legacy'  # The legacy scheduler from the original T5 paper.

class Metric(enum.Enum):
    """Metrics."""
    ACCURACY: str = 'accuracy'
    LOSS: str = 'loss'
    PRECISION: str = 'precision'
    RECALL: str = 'recall'
    MCC: str = 'mcc'
    SPEARMAN: str = 'spearman'
    PEARSON: str = 'pearson'
    ROUGE: str = 'rouge'
    ROUGE_L: str = 'rougeL'
    GRAD_L2: str = 'grad_l2'
    WEIGHTS_L2: str = 'weights_l2'
    LR: str = 'lr'
    SECONDS_PER_STEP: str = 'seconds_per_step'
    TIME: str = 'time'


@dataclasses.dataclass
class TokenizerConstants:
    """
    Tokenizer constants. These are used to tokenize the inputs and outputs of the T5 model.
    """
    TOKENIZER = 'tokenizer'
    IN_LENGTH = 'in_length'
    OUT_LENGTH = 'out_length'

@dataclasses.dataclass
class T5TokenizerConstants(TokenizerConstants):
    START_TOKEN: str = '</s>'
    PAD_TOKEN: str = '<pad>'
    SPACE_TOKEN: str = "▁"
    PAD_TOKEN_ID: int = -100

@dataclasses.dataclass
class OptimizerConstants:
    """
    Optimizer constants.
    """
    PARAMS: str = 'params'
    WEIGHT_DECAY: str = 'weight_decay'
    STEP: str = 'step'
    EXP_AVG: str = 'exp_avg'
    EXP_AVG_SQ: str = 'exp_avg_sq'
    LR: str = 'lr'
    CORRECT_BIAS: str = 'correct_bias'
    EPS: str = 'eps'


@dataclasses.dataclass
class SchedulerConstants:
    NO_DECAY: typing.Tuple[str] = ("bias", "LayerNorm", "layernorm", "layer_norm", "ln")

