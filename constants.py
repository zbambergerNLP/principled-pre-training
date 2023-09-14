GLUE_PREPROCESSED_COLUMN_NAMES = ['idx', 'processed_inputs', 'processed_outputs']
GLUE_TOKENIZED_COLUMN_NAMES = ['input_ids', 'attention_mask', 'labels']

# Label names
NEGATIVE = 'negative'
POSITIVE = 'positive'
UNACCEPTABLE = 'unacceptable'
ACCEPTABLE = 'acceptable'
ENTAILMENT = 'entailment'
NOT_ENTAILMENT = 'not_entailment'
CONTRADICTION = 'contradiction'
NEUTRAL = 'neutral'
NOT_DUPLICATE = 'not_duplicate'
DUPLICATE = 'duplicate'
NOT_EQUIVALENT = 'not_equivalent'
EQUIVALENT = 'equivalent'
OTHER = 'other'

# Encoder-Decoder Feature names
INPUT_IDS = 'input_ids'
TOKEN_TYPE_IDS = 'token_type_ids'
ATTENTION_MASK = 'attention_mask'
LABEL = 'label'

# Metric names
PRECISION = 'precision'
RECALL = 'recall'
F1 = 'f1'
ACCURACY = 'accuracy'
MCC = 'mcc'
SPEARMAN = 'spearman'
PEARSON = 'pearson'

# Example-level metric names
EXAMPLE_ACCURACY = 'example_accuracy'
EXAMPLE_F1 = 'example_f1'
EXAMPLE_PRECISION = 'example_precision'
EXAMPLE_RECALL = 'example_recall'
EXAMPLE_MCC = 'example_mcc'
EXAMPLE_SPEARMAN = 'example_spearman'
EXAMPLE_PEARSON = 'example_pearson'

# Token-level metric names
TOKEN_ACCURACY = 'token_accuracy'
TOKEN_F1 = 'token_f1'
TOKEN_PRECISION = 'token_precision'
Token_RECAll = 'token_recall'
TOKEN_MCC = 'token_mcc'
TOKEN_SPEARMAN = 'token_spearman'
TOKEN_PEARSON = 'token_pearson'

# Text column names
SENTENCE_1 = 'sentence1'
SENTENCE_2 = 'sentence2'
SENTENCE = 'sentence'
QUESTION = 'question'
HYPOTHESIS = 'hypothesis'
PREMISE = 'premise'
TEXT = 'text'
SENTENCES = 'sentences'
QUESTION_1 = 'question1'
QUESTION_2 = 'question2'

# Split Names
TRAIN = 'train'
VALIDATION = 'validation'
TEST = 'test'

# Benchmark Names
GLUE = 'glue'
SUPER_GLUE = 'super_glue'
DISCO_EVAL = 'disco_eval'

# Dataset Names
SST2 = 'sst2'
COLA = 'cola'
RTE = 'rte'
MNLI = 'mnli'
QNLI = 'qnli'
MRPC = 'mrpc'
QQP = 'qqp'
STSB = 'stsb'
MNLI_MATCHED = 'mnli_matched'
MNLI_MISMATCHED = 'mnli_mismatched'
WNLI = 'wnli'
AX = 'ax'
ALL = 'all'

# Dataset Descriptors
PREFIX = 'prefix'
PREFIX_1 = 'prefix_1'
PREFIX_2 = 'prefix_2'
TEXT_COLUMN_NAME = 'text_column_name'
TEXT_COLUMN_NAME_1 = 'text_column_name_1'
TEXT_COLUMN_NAME_2 = 'text_column_name_2'
LABEL_COLUMN_NAME = 'label_column_name'
METRIC_TO_OPTIMIZE = 'metric_to_optimize'
GREATER_IS_BETTER = 'greater_is_better'
METRIC_NAMES = 'metric_names'
LABELS = 'labels'

# Training Parameters
STEPS = 'steps'


DATASET_VALS = {
    GLUE:
        {
            SST2:
                {
                    PREFIX: f'{SST2} {SENTENCE}: ',
                    TEXT_COLUMN_NAME: SENTENCE,
                    LABEL_COLUMN_NAME: LABEL,
                    METRIC_TO_OPTIMIZE: EXAMPLE_ACCURACY,
                    GREATER_IS_BETTER: True,
                    METRIC_NAMES: [ACCURACY, F1, PRECISION, RECALL],
                    LABELS: {
                        0: NEGATIVE,
                        1: POSITIVE,
                        -1: OTHER,
                    },
                },

            COLA:
                {
                    PREFIX: f'{COLA} {SENTENCE}: ',
                    TEXT_COLUMN_NAME: SENTENCE,
                    LABEL_COLUMN_NAME: LABEL,
                    METRIC_TO_OPTIMIZE: EXAMPLE_MCC,
                    GREATER_IS_BETTER: True,
                    METRIC_NAMES: [ACCURACY, F1, PRECISION, RECALL, MCC],
                    LABELS: {
                        0: UNACCEPTABLE,
                        1: ACCEPTABLE,
                        -1: OTHER,
                    },
                },
            RTE:
                {
                    PREFIX_1: f'{RTE} {SENTENCE_1}: ',
                    PREFIX_2: f'{SENTENCE_2}: ',
                    TEXT_COLUMN_NAME_1: SENTENCE_1,
                    TEXT_COLUMN_NAME_2: SENTENCE_2,
                    LABEL_COLUMN_NAME: LABEL,
                    METRIC_TO_OPTIMIZE: EXAMPLE_ACCURACY,
                    GREATER_IS_BETTER: True,
                    METRIC_NAMES: [ACCURACY, F1, PRECISION, RECALL],
                    LABELS: {
                        0: ENTAILMENT,
                        1: NOT_ENTAILMENT,
                        -1: OTHER,
                    },
                },

            MNLI:
                {
                    PREFIX_1: f'{MNLI} {HYPOTHESIS}: ',
                    PREFIX_2: f'{PREMISE}: ',
                    TEXT_COLUMN_NAME_1: PREMISE,
                    TEXT_COLUMN_NAME_2: HYPOTHESIS,
                    LABEL_COLUMN_NAME: LABEL,
                    METRIC_TO_OPTIMIZE: EXAMPLE_ACCURACY,
                    GREATER_IS_BETTER: True,
                    METRIC_NAMES: [ACCURACY, F1, PRECISION, RECALL],
                    LABELS: {
                        2: CONTRADICTION,
                        0: ENTAILMENT,
                        1: NEUTRAL,
                        -1: OTHER,
                    },
                },
            QNLI:
                {
                    PREFIX_1: f'{QNLI} {QUESTION}: ',
                    PREFIX_2: f'{SENTENCE}: ',
                    TEXT_COLUMN_NAME_1: QUESTION,
                    TEXT_COLUMN_NAME_2: SENTENCE,
                    LABEL_COLUMN_NAME: LABEL,
                    METRIC_TO_OPTIMIZE: EXAMPLE_ACCURACY,
                    GREATER_IS_BETTER: True,
                    METRIC_NAMES: [ACCURACY, F1, PRECISION, RECALL],
                    LABELS: {
                        1: NOT_ENTAILMENT,
                        0: ENTAILMENT,
                        -1: OTHER,
                    },
                },
            MRPC:
                {
                    PREFIX_1: f'{MRPC} {SENTENCE_1}: ',
                    PREFIX_2: f'{SENTENCE_2}: ',
                    TEXT_COLUMN_NAME_1: SENTENCE_1,
                    TEXT_COLUMN_NAME_2: SENTENCE_2,
                    LABEL_COLUMN_NAME: LABEL,
                    METRIC_TO_OPTIMIZE: EXAMPLE_F1,
                    GREATER_IS_BETTER: True,
                    METRIC_NAMES: [ACCURACY, F1, PRECISION, RECALL],
                    LABELS: {
                        0: NOT_EQUIVALENT,
                        1: EQUIVALENT,
                        -1: OTHER,
                    },
                },
            QQP:
                {
                    PREFIX_1: f'{QQP} {QUESTION_1}: ',
                    PREFIX_2: f'{QUESTION_2}: ',
                    TEXT_COLUMN_NAME_1: QUESTION_1,
                    TEXT_COLUMN_NAME_2: QUESTION_2,
                    LABEL_COLUMN_NAME: LABEL,
                    METRIC_TO_OPTIMIZE: EXAMPLE_ACCURACY,
                    GREATER_IS_BETTER: True,
                    METRIC_NAMES: [ACCURACY, F1, PRECISION, RECALL],
                    LABELS: {
                        0: NOT_DUPLICATE,
                        1: DUPLICATE,
                        -1: OTHER,
                    },
                },
            STSB:
                {
                    PREFIX_1: f'{STSB} {SENTENCE_1}: ',
                    PREFIX_2: f'{SENTENCE_2}: ',
                    TEXT_COLUMN_NAME_1: SENTENCE_1,
                    TEXT_COLUMN_NAME_2: SENTENCE_2,
                    LABEL_COLUMN_NAME: LABEL,
                    METRIC_TO_OPTIMIZE: EXAMPLE_SPEARMAN,
                    GREATER_IS_BETTER: True,
                    METRIC_NAMES: [SPEARMAN, PEARSON, ACCURACY, F1, PRECISION, RECALL],
                    LABELS: None,
                },
            MNLI_MATCHED:
                {
                    PREFIX_1: f'{MNLI} {HYPOTHESIS}: ',
                    PREFIX_2: f'{PREMISE}: ',
                    TEXT_COLUMN_NAME_1: PREMISE,
                    TEXT_COLUMN_NAME_2: HYPOTHESIS,
                    LABEL_COLUMN_NAME: LABEL,
                    METRIC_TO_OPTIMIZE: EXAMPLE_ACCURACY,
                    GREATER_IS_BETTER: True,
                    METRIC_NAMES: [ACCURACY, F1, PRECISION, RECALL],
                    LABELS: {
                        2: CONTRADICTION,
                        0: ENTAILMENT,
                        1: NEUTRAL,
                        -1: OTHER,
                    }
                },
            MNLI_MISMATCHED:
                {
                    PREFIX_1: f'{MNLI} {HYPOTHESIS}: ',
                    PREFIX_2: f'{PREMISE}: ',
                    TEXT_COLUMN_NAME_1: PREMISE,
                    TEXT_COLUMN_NAME_2: HYPOTHESIS,
                    LABEL_COLUMN_NAME: LABEL,
                    METRIC_TO_OPTIMIZE: EXAMPLE_ACCURACY,
                    GREATER_IS_BETTER: True,
                    METRIC_NAMES: [ACCURACY, F1, PRECISION, RECALL],
                    LABELS: {
                        2: CONTRADICTION,
                        0: ENTAILMENT,
                        1: NEUTRAL,
                        -1: OTHER,
                    },
                },
            WNLI:
                {
                    PREFIX_1: f'{WNLI} {SENTENCE_1}: ',
                    PREFIX_2: f'{SENTENCE_2}: ',
                    TEXT_COLUMN_NAME_1: SENTENCE_1,
                    TEXT_COLUMN_NAME_2: SENTENCE_2,
                    LABEL_COLUMN_NAME: LABEL,
                    METRIC_TO_OPTIMIZE: EXAMPLE_ACCURACY,
                    GREATER_IS_BETTER: True,
                    METRIC_NAMES: [ACCURACY, F1, PRECISION, RECALL],
                    LABELS: {
                        0: NOT_ENTAILMENT,
                        1: ENTAILMENT,
                        -1: OTHER,
                    },
                },
            AX:
                {
                    PREFIX_1: f'{AX} {PREMISE}: ',
                    PREFIX_2: f'{HYPOTHESIS}: ',
                    TEXT_COLUMN_NAME_1: PREMISE,
                    TEXT_COLUMN_NAME_2: HYPOTHESIS,
                    LABEL_COLUMN_NAME: LABEL,
                    METRIC_TO_OPTIMIZE: EXAMPLE_ACCURACY,
                    GREATER_IS_BETTER: True,
                    METRIC_NAMES: [ACCURACY, F1, PRECISION, RECALL],
                    LABELS: {
                        2: CONTRADICTION,
                        0: ENTAILMENT,
                        1: NEUTRAL,
                        -1: OTHER,
                    },
                },
            ALL:
                {
                    # Tokenize each dataset separately, and consider the union of all metrics.
                    METRIC_TO_OPTIMIZE: EXAMPLE_ACCURACY,
                    GREATER_IS_BETTER: True,
                    METRIC_NAMES: [ACCURACY, F1, PRECISION, RECALL, MCC, SPEARMAN, PEARSON],
                }
        },
}
