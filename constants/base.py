from dataclasses import dataclass
import typing

T5_START_TOKEN = '</s>'
T5_PAD_TOKEN = '<pad>'
T5_SPACE_TOKEN = "▁"

@dataclass
class BasicConstants:
    PREPROCESSED_COLUMN_NAMES: typing.Tuple[str] = ('idx', 'processed_inputs', 'processed_outputs')
    TOKENIZED_COLUMN_NAMES: typing.Tuple[str] = ('input_ids', 'attention_mask', 'labels')
    INPUT_IDS: str = 'input_ids'
    TOKEN_TYPE_IDS: str = 'token_type_ids'
    ATTENTION_MASK: str = 'attention_mask'
    LABEL: str = 'label'

    # Metric names
    PRECISION: str = 'precision'
    RECALL: str = 'recall'
    F1: str = 'f1'
    ACCURACY: str = 'accuracy'
    MCC: str = 'mcc'
    SPEARMAN: str = 'spearman'
    PEARSON: str = 'pearson'

    # Example-level metric names
    EXAMPLE_ACCURACY: str = 'example_accuracy'
    EXAMPLE_F1: str = 'example_f1'
    EXAMPLE_PRECISION: str = 'example_precision'
    EXAMPLE_RECALL: str = 'example_recall'
    EXAMPLE_MCC: str = 'example_mcc'
    EXAMPLE_SPEARMAN: str = 'example_spearman'
    EXAMPLE_PEARSON: str = 'example_pearson'

    # Token-level metric names
    TOKEN_ACCURACY: str = 'token_accuracy'
    TOKEN_F1: str = 'token_f1'
    TOKEN_PRECISION: str = 'token_precision'
    Token_RECAll: str = 'token_recall'
    TOKEN_MCC: str = 'token_mcc'
    TOKEN_SPEARMAN: str = 'token_spearman'
    TOKEN_PEARSON: str = 'token_pearson'

    # Split Names
    TRAIN: str = 'train'
    VALIDATION: str = 'validation'
    TEST: str = 'test'

    # Training Parameters
    STEPS = 'steps'

