from dataclasses import dataclass
from .base import BasicConstants
import typing


@dataclass(frozen=True)
class TaskConfigOneInput:
    NAME: str
    PREFIX: str
    TEXT_COLUMN_NAME: str
    LABEL_COLUMN_NAME: str
    METRIC_TO_OPTIMIZE: str
    GREATER_IS_BETTER: bool
    METRIC_NAMES: typing.List[str]
    LABELS: dict


@dataclass(frozen=True)
class TaskConfigTwoInput:
    NAME: str
    PREFIX_1: str
    PREFIX_2: str
    TEXT_COLUMN_NAME_1: str
    TEXT_COLUMN_NAME_2: str
    LABEL_COLUMN_NAME: str
    METRIC_TO_OPTIMIZE: str
    GREATER_IS_BETTER: bool
    METRIC_NAMES: typing.List[str]
    LABELS: dict


@dataclass
class GlueConstants(BasicConstants):
    # Label names
    NEGATIVE: str = 'negative'
    POSITIVE: str = 'positive'
    UNACCEPTABLE: str = 'unacceptable'
    ACCEPTABLE: str = 'acceptable'
    ENTAILMENT: str = 'entailment'
    NOT_ENTAILMENT: str = 'not_entailment'
    CONTRADICTION: str = 'contradiction'
    NEUTRAL: str = 'neutral'
    NOT_DUPLICATE: str = 'not_duplicate'
    DUPLICATE: str = 'duplicate'
    NOT_EQUIVALENT: str = 'not_equivalent'
    EQUIVALENT: str = 'equivalent'
    OTHER: str = 'other'

    # GLUE Dataset Names
    SST2: str = 'sst2'
    COLA: str = 'cola'
    RTE: str = 'rte'
    MNLI: str = 'mnli'
    QNLI: str = 'qnli'
    MRPC: str = 'mrpc'
    QQP: str = 'qqp'
    STSB: str = 'stsb'
    MNLI_MATCHED: str = 'mnli_matched'
    MNLI_MISMATCHED: str = 'mnli_mismatched'
    WNLI: str = 'wnli'
    AX: str = 'ax'
    ALL: str = 'all'

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
    TASKS = [
        SST2,
        COLA,
        RTE,
        MNLI,
        QNLI,
        MRPC,
        QQP,
        STSB,
        MNLI_MISMATCHED,
        MNLI_MATCHED,
        WNLI,
        AX,
    ]

    def __post_init__(self):
        self.SST2_TASK = TaskConfigOneInput(
            NAME=self.SST2,
            PREFIX=f'{self.SST2}: {self.SENTENCE}: ',
            TEXT_COLUMN_NAME=self.SENTENCE,
            LABEL_COLUMN_NAME=self.LABEL,
            METRIC_TO_OPTIMIZE=self.EXAMPLE_ACCURACY,
            GREATER_IS_BETTER=True,
            METRIC_NAMES=[self.ACCURACY, self.F1, self.PRECISION, self.RECALL],
            LABELS={
                0: self.NEGATIVE,
                1: self.POSITIVE,
                -1: self.OTHER,
            }
        )

        self.COLA_TASK = TaskConfigOneInput(
            NAME=self.COLA,
            PREFIX=f'{self.COLA}: {self.SENTENCE}: ',
            TEXT_COLUMN_NAME=self.SENTENCE,
            LABEL_COLUMN_NAME=self.LABEL,
            METRIC_TO_OPTIMIZE=self.EXAMPLE_ACCURACY,
            GREATER_IS_BETTER=True,
            METRIC_NAMES=[self.ACCURACY, self.F1, self.PRECISION, self.RECALL, self.MCC],
            LABELS={
                0: self.UNACCEPTABLE,
                1: self.ACCEPTABLE,
                -1: self.OTHER,
            }
        )

        self.RTE_TASK = TaskConfigTwoInput(
            NAME=self.RTE,
            PREFIX_1=f'{self.RTE}: {self.SENTENCE_1}: ',
            PREFIX_2=f'{self.SENTENCE_2}',
            TEXT_COLUMN_NAME_1=self.SENTENCE_1,
            TEXT_COLUMN_NAME_2=self.SENTENCE_2,
            LABEL_COLUMN_NAME=self.LABEL,
            METRIC_TO_OPTIMIZE=self.EXAMPLE_ACCURACY,
            GREATER_IS_BETTER=True,
            METRIC_NAMES=[self.ACCURACY, self.F1, self.PRECISION, self.RECALL],
            LABELS={
                0: self.NEGATIVE,
                1: self.POSITIVE,
                -1: self.OTHER,
            }
        )
        self.MNLI_TASK = TaskConfigTwoInput(
            NAME=self.MNLI,
            PREFIX_1=f'{self.MNLI} {self.HYPOTHESIS}: ',
            PREFIX_2=f'{self.PREMISE}: ',
            TEXT_COLUMN_NAME_1=self.PREMISE,
            TEXT_COLUMN_NAME_2=self.HYPOTHESIS,
            LABEL_COLUMN_NAME=self.LABEL,
            METRIC_TO_OPTIMIZE=self.EXAMPLE_ACCURACY,
            GREATER_IS_BETTER=True,
            METRIC_NAMES=[self.ACCURACY, self.F1, self.PRECISION, self.RECALL],
            LABELS={
                2: self.CONTRADICTION,
                0: self.ENTAILMENT,
                1: self.NEUTRAL,
                -1: self.OTHER,
            }
        )

        self.QNLI_TASK = TaskConfigTwoInput(
            NAME=self.QNLI,
            PREFIX_1=f'{self.QNLI} {self.QUESTION}: ',
            PREFIX_2=f'{self.SENTENCE}: ',
            TEXT_COLUMN_NAME_1=self.QUESTION,
            TEXT_COLUMN_NAME_2=self.SENTENCE,
            LABEL_COLUMN_NAME=self.LABEL,
            METRIC_TO_OPTIMIZE=self.EXAMPLE_ACCURACY,
            GREATER_IS_BETTER=True,
            METRIC_NAMES=[self.ACCURACY, self.F1, self.PRECISION, self.RECALL],
            LABELS={
                1: self.NOT_ENTAILMENT,
                0: self.ENTAILMENT,
                -1: self.OTHER,
            }
        )

        self.MRPC_TASK = TaskConfigTwoInput(
            NAME=self.MRPC,
            PREFIX_1=f'{self.MRPC} {self.SENTENCE_1}: ',
            PREFIX_2=f'{self.SENTENCE_2}: ',
            TEXT_COLUMN_NAME_1=self.SENTENCE_1,
            TEXT_COLUMN_NAME_2=self.SENTENCE_2,
            LABEL_COLUMN_NAME=self.LABEL,
            METRIC_TO_OPTIMIZE=self.EXAMPLE_F1,
            GREATER_IS_BETTER=True,
            METRIC_NAMES=[self.ACCURACY, self.F1, self.PRECISION, self.RECALL],
            LABELS={
                0: self.NOT_EQUIVALENT,
                1: self.EQUIVALENT,
                -1: self.OTHER,
            }
        )

        self.QQP_TASK = TaskConfigTwoInput(
            NAME=self.QQP,
            PREFIX_1=f'{self.QQP} {self.QUESTION_1}: ',
            PREFIX_2=f'{self.QUESTION_2}: ',
            TEXT_COLUMN_NAME_1=self.QUESTION_1,
            TEXT_COLUMN_NAME_2=self.QUESTION_2,
            LABEL_COLUMN_NAME=self.LABEL,
            METRIC_TO_OPTIMIZE=self.EXAMPLE_ACCURACY,
            GREATER_IS_BETTER=True,
            METRIC_NAMES=[self.ACCURACY, self.F1, self.PRECISION, self.RECALL],
            LABELS={
                0: self.NOT_DUPLICATE,
                1: self.DUPLICATE,
                -1: self.OTHER,
            }
        )

        self.STSB_TASK = TaskConfigTwoInput(
            NAME=self.STSB,
            PREFIX_1=f'{self.STSB} {self.SENTENCE_1}: ',
            PREFIX_2=f'{self.SENTENCE_2}: ',
            TEXT_COLUMN_NAME_1=self.SENTENCE_1,
            TEXT_COLUMN_NAME_2=self.SENTENCE_2,
            LABEL_COLUMN_NAME=self.LABEL,
            METRIC_TO_OPTIMIZE=self.EXAMPLE_SPEARMAN,
            GREATER_IS_BETTER=True,
            METRIC_NAMES=[self.SPEARMAN, self.PEARSON, self.ACCURACY, self.F1, self.PRECISION, self.RECALL],
            LABELS=None
        )

        self.MNLI_MATCHED_TASK = TaskConfigTwoInput(
            NAME=self.MNLI_MATCHED,
            PREFIX_1=f'{self.MNLI_MATCHED} {self.HYPOTHESIS}: ',
            PREFIX_2=f'{self.PREMISE}: ',
            TEXT_COLUMN_NAME_1=self.PREMISE,
            TEXT_COLUMN_NAME_2=self.HYPOTHESIS,
            LABEL_COLUMN_NAME=self.LABEL,
            METRIC_TO_OPTIMIZE=self.EXAMPLE_ACCURACY,
            GREATER_IS_BETTER=True,
            METRIC_NAMES=[self.ACCURACY, self.F1, self.PRECISION, self.RECALL],
            LABELS={
                2: self.CONTRADICTION,
                0: self.ENTAILMENT,
                1: self.NEUTRAL,
                -1: self.OTHER,
            }
        )

        self.MNLI_MISMATCHED_TASK = TaskConfigTwoInput(
            NAME=self.MNLI_MISMATCHED,
            PREFIX_1=f'{self.MNLI_MISMATCHED} {self.HYPOTHESIS}: ',
            PREFIX_2=f'{self.PREMISE}: ',
            TEXT_COLUMN_NAME_1=self.PREMISE,
            TEXT_COLUMN_NAME_2=self.HYPOTHESIS,
            LABEL_COLUMN_NAME=self.LABEL,
            METRIC_TO_OPTIMIZE=self.EXAMPLE_ACCURACY,
            GREATER_IS_BETTER=True,
            METRIC_NAMES=[self.ACCURACY, self.F1, self.PRECISION, self.RECALL],
            LABELS={
                2: self.CONTRADICTION,
                0: self.ENTAILMENT,
                1: self.NEUTRAL,
                -1: self.OTHER,
            }
        )

        self.WNLI_TASK = TaskConfigTwoInput(
            NAME=self.WNLI,
            PREFIX_1=f'{self.WNLI} {self.SENTENCE_1}: ',
            PREFIX_2=f'{self.SENTENCE_2}: ',
            TEXT_COLUMN_NAME_1=self.SENTENCE_1,
            TEXT_COLUMN_NAME_2=self.SENTENCE_2,
            LABEL_COLUMN_NAME=self.LABEL,
            METRIC_TO_OPTIMIZE=self.EXAMPLE_ACCURACY,
            GREATER_IS_BETTER=True,
            METRIC_NAMES=[self.ACCURACY, self.F1, self.PRECISION, self.RECALL],
            LABELS={
                0: self.NOT_ENTAILMENT,
                1: self.ENTAILMENT,
                -1: self.OTHER,
            }
        )

        self.AX_TASK = TaskConfigTwoInput(
            NAME=self.AX,
            PREFIX_1=f'{self.AX} {self.PREMISE}: ',
            PREFIX_2=f'{self.HYPOTHESIS}: ',
            TEXT_COLUMN_NAME_1=self.PREMISE,
            TEXT_COLUMN_NAME_2=self.HYPOTHESIS,
            LABEL_COLUMN_NAME=self.LABEL,
            METRIC_TO_OPTIMIZE=self.EXAMPLE_ACCURACY,
            GREATER_IS_BETTER=True,
            METRIC_NAMES=[self.ACCURACY, self.F1, self.PRECISION, self.RECALL],
            LABELS={
                2: self.CONTRADICTION,
                0: self.ENTAILMENT,
                1: self.NEUTRAL,
                -1: self.OTHER,
            }
        )

    def __getitem__(self, item):
        if item == self.SST2:
            return self.SST2_TASK
        elif item == self.COLA:
            return self.COLA_TASK
        elif item == self.RTE:
            return self.RTE_TASK
        elif item == self.MNLI:
            return self.MNLI_TASK
        elif item == self.QNLI:
            return self.QNLI_TASK
        elif item == self.MRPC:
            return self.MRPC_TASK
        elif item == self.QQP:
            return self.QQP_TASK
        elif item == self.STSB:
            return self.STSB_TASK
        elif item == self.MNLI_MATCHED:
            return self.MNLI_MATCHED_TASK
        elif item == self.MNLI_MISMATCHED:
            return self.MNLI_MISMATCHED_TASK
        elif item == self.WNLI:
            return self.WNLI_TASK
        elif item == self.AX:
            return self.AX_TASK
        else:
            raise KeyError(f'{item} is not a valid GLUE task')
