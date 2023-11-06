from .base import BasicConstants
from dataclasses import dataclass

import typing


@dataclass(frozen=True)
class DiscoEvalTaskConfig:
    """
    A dataclass for a specific DiscoEval task.
    Prefix: The prefix to add to the text columns
    Text Column Amount: The amount of text columns in the dataset task
    Label Column Name: The name of the label column
    Metric to Optimize: The metric to optimize for the task
    Greater is Better: Whether a higher value of the metric is better
    Metric Names: The names of the metrics to calculate
    Labels: The labels of the task
    """
    PREFIX: str
    TEXT_COLUMN_AMOUNT: int
    LABEL_COLUMN_NAME: str
    METRIC_TO_OPTIMIZE: str
    GREATER_IS_BETTER: bool
    METRIC_NAMES: typing.List[str]
    LABELS: dict


@dataclass
class DiscoEvalConstants(BasicConstants):
    """
    A dataclass for DiscoEval constants. Inherits from BasicConstants.
    """
    DISCO_EVAL: str = 'OfekGlick/DiscoEval'
    # DiscoEval Task Names
    SSPABS = 'SSPabs'
    PDTB_I = 'PDTB_I'
    PDTB_E = 'PDTB_E'
    SPARXIV = 'SParxiv'
    SPROCSTORY = 'SProcstory'
    SPWIKI = 'SPwiki'
    BSOARXIV = 'BSOarxiv'
    BSOROCSTORY = 'BSOrocstory'
    BSOWIKI = 'BSOwiki'
    DCCHAT = 'DCchat'
    DCWIKI = 'DCwiki'
    RST = 'RST'

    numbers = {
        1: "first",
        2: "second",
        3: "third",
        4: "fourth",
        5: "fifth",
        6: "sixth",
        7: "seventh",
        8: "eighth",
        9: "ninth"

    }

    TEXT_COLUMN_AMOUNT = 'text_column_amount'
    RST_TEXT_COLUMNS = 2
    DC_TEXT_COLUMNS = 6
    BSO_TEXT_COLUMNS = 2
    SP_TEXT_COLUMNS = 5
    PDTB_E_TEXT_COLUMNS = 2
    PDTB_I_TEXT_COLUMNS = 2
    SSPABS_TEXT_COLUMNS = 1

    TASKS = [
        SSPABS,
        PDTB_I,
        PDTB_E,
        SPARXIV,
        SPROCSTORY,
        SPWIKI,
        BSOARXIV,
        BSOROCSTORY,
        BSOWIKI,
        DCCHAT,
        DCWIKI,
        RST,
    ]

    SP_LABELS = {
        "0": 'First sentence',
        "1": 'Second sentence',
        "2": 'Third sentence',
        "3": "Fourth sentence",
        "4": "Fifth sentence",
    }
    BSO_LABELS = {
        "0": 'Incorrect order',
        "1": 'Correct order',
    }
    DC_LABELS = {
        "0": "Incoherent",
        "1": "Coherent",
    }
    SSPABS_LABELS = {
        "0": "Does not belong to abstract",
        "1": "Belongs to abstract",
    }

    RST_LABELS = [
        'NS-Explanation',
        'NS-Evaluation',
        'NN-Condition',
        'NS-Summary',
        'SN-Cause',
        'SN-Background',
        'NS-Background',
        'SN-Summary',
        'NS-Topic-Change',
        'NN-Explanation',
        'SN-Topic-Comment',
        'NS-Elaboration',
        'SN-Attribution',
        'SN-Manner-Means',
        'NN-Evaluation',
        'NS-Comparison',
        'NS-Contrast',
        'SN-Condition',
        'NS-Temporal',
        'NS-Enablement',
        'SN-Evaluation',
        'NN-Topic-Comment',
        'NN-Temporal',
        'NN-Textual-organization',
        'NN-Same-unit',
        'NN-Comparison',
        'NN-Topic-Change',
        'SN-Temporal',
        'NN-Joint',
        'SN-Enablement',
        'SN-Explanation',
        'NN-Contrast',
        'NN-Cause',
        'SN-Contrast',
        'NS-Attribution',
        'NS-Topic-Comment',
        'SN-Elaboration',
        'SN-Comparison',
        'NS-Cause',
        'NS-Condition',
        'NS-Manner-Means'
    ]

    PDTB_E_LABELS = [
        'Comparison.Concession',
        'Comparison.Contrast',
        'Contingency.Cause',
        'Contingency.Condition',
        'Contingency.Pragmatic condition',
        'Expansion.Alternative',
        'Expansion.Conjunction',
        'Expansion.Instantiation',
        'Expansion.List',
        'Expansion.Restatement',
        'Temporal.Asynchronous',
        'Temporal.Synchrony',
    ]

    PDTB_I_LABELS = [
        'Comparison.Concession',
        'Comparison.Contrast',
        'Contingency.Cause',
        'Contingency.Pragmatic cause',
        'Expansion.Alternative',
        'Expansion.Conjunction',
        'Expansion.Instantiation',
        'Expansion.List',
        'Expansion.Restatement',
        'Temporal.Asynchronous',
        'Temporal.Synchrony',
    ]

    def __post_init__(self):
        self.TEXT_COLUMN_NAMES = [f"{self.numbers[i]}_sentence" for i in range(1, 10)]
        self.PDTB_E_LABELS = {str(i): self.PDTB_E_LABELS[i] for i in range(len(self.PDTB_E_LABELS))}
        self.RST_LABELS = {str(i): self.RST_LABELS[i] for i in range(len(self.RST_LABELS))}
        self.PDTB_I_LABELS = {str(i): self.PDTB_I_LABELS[i] for i in range(len(self.PDTB_I_LABELS))}
        self.SPARXIV_TASK = DiscoEvalTaskConfig(
            PREFIX=f'{self.SPARXIV}: ',
            TEXT_COLUMN_AMOUNT=self.SP_TEXT_COLUMNS,
            LABEL_COLUMN_NAME=self.LABEL,
            METRIC_TO_OPTIMIZE=self.EXAMPLE_ACCURACY,
            GREATER_IS_BETTER=True,
            METRIC_NAMES=[self.ACCURACY, self.F1, self.PRECISION, self.RECALL],
            LABELS=self.SP_LABELS
        )
        self.SPROCSTORY_TASK = DiscoEvalTaskConfig(
            PREFIX=f'{self.SPROCSTORY}: ',
            TEXT_COLUMN_AMOUNT=self.SP_TEXT_COLUMNS,
            LABEL_COLUMN_NAME=self.LABEL,
            METRIC_TO_OPTIMIZE=self.EXAMPLE_ACCURACY,
            GREATER_IS_BETTER=True,
            METRIC_NAMES=[self.ACCURACY, self.F1, self.PRECISION, self.RECALL],
            LABELS=self.SP_LABELS,
        )
        self.SPWIKI_TASK = DiscoEvalTaskConfig(
            PREFIX=f'{self.SPWIKI}: ',
            TEXT_COLUMN_AMOUNT=self.SP_TEXT_COLUMNS,
            LABEL_COLUMN_NAME=self.LABEL,
            METRIC_TO_OPTIMIZE=self.EXAMPLE_ACCURACY,
            GREATER_IS_BETTER=True,
            METRIC_NAMES=[self.ACCURACY, self.F1, self.PRECISION, self.RECALL],
            LABELS=self.SP_LABELS,
        )
        self.BSOARXIV_TASK = DiscoEvalTaskConfig(
            PREFIX=f'{self.BSOARXIV}: ',
            TEXT_COLUMN_AMOUNT=self.BSO_TEXT_COLUMNS,
            LABEL_COLUMN_NAME=self.LABEL,
            METRIC_TO_OPTIMIZE=self.EXAMPLE_ACCURACY,
            GREATER_IS_BETTER=True,
            METRIC_NAMES=[self.ACCURACY, self.F1, self.PRECISION, self.RECALL],
            LABELS=self.BSO_LABELS
        )
        self.BSOROCSTORY_TASK = DiscoEvalTaskConfig(
            PREFIX=f'{self.BSOROCSTORY}: ',
            TEXT_COLUMN_AMOUNT=self.BSO_TEXT_COLUMNS,
            LABEL_COLUMN_NAME=self.LABEL,
            METRIC_TO_OPTIMIZE=self.EXAMPLE_ACCURACY,
            GREATER_IS_BETTER=True,
            METRIC_NAMES=[self.ACCURACY, self.F1, self.PRECISION, self.RECALL],
            LABELS=self.BSO_LABELS
        )
        self.BSOWIKI_TASK = DiscoEvalTaskConfig(
            PREFIX=f'{self.BSOWIKI}: ',
            TEXT_COLUMN_AMOUNT=self.BSO_TEXT_COLUMNS,
            LABEL_COLUMN_NAME=self.LABEL,
            METRIC_TO_OPTIMIZE=self.EXAMPLE_ACCURACY,
            GREATER_IS_BETTER=True,
            METRIC_NAMES=[self.ACCURACY, self.F1, self.PRECISION, self.RECALL],
            LABELS=self.BSO_LABELS
        )
        self.DCCHAT_TASK = DiscoEvalTaskConfig(
            PREFIX=f'{self.DCCHAT}: ',
            TEXT_COLUMN_AMOUNT=self.DC_TEXT_COLUMNS,
            LABEL_COLUMN_NAME=self.LABEL,
            METRIC_TO_OPTIMIZE=self.EXAMPLE_ACCURACY,
            GREATER_IS_BETTER=True,
            METRIC_NAMES=[self.ACCURACY, self.F1, self.PRECISION, self.RECALL],
            LABELS=self.DC_LABELS
        )
        self.DCWIKI_TASK = DiscoEvalTaskConfig(
            PREFIX=f'{self.DCWIKI}: ',
            TEXT_COLUMN_AMOUNT=self.DC_TEXT_COLUMNS,
            LABEL_COLUMN_NAME=self.LABEL,
            METRIC_TO_OPTIMIZE=self.EXAMPLE_ACCURACY,
            GREATER_IS_BETTER=True,
            METRIC_NAMES=[self.ACCURACY, self.F1, self.PRECISION, self.RECALL],
            LABELS=self.DC_LABELS
        )
        self.RST_TASK = DiscoEvalTaskConfig(
            PREFIX=f'{self.RST}: ',
            TEXT_COLUMN_AMOUNT=self.RST_TEXT_COLUMNS,
            LABEL_COLUMN_NAME=self.LABEL,
            METRIC_TO_OPTIMIZE=self.EXAMPLE_ACCURACY,
            GREATER_IS_BETTER=True,
            METRIC_NAMES=[self.ACCURACY, self.F1, self.PRECISION, self.RECALL],
            LABELS=self.RST_LABELS
        )
        self.PDTB_E_TASK = DiscoEvalTaskConfig(
            PREFIX=f'{self.PDTB_E}: ',
            TEXT_COLUMN_AMOUNT=self.PDTB_E_TEXT_COLUMNS,
            LABEL_COLUMN_NAME=self.LABEL,
            METRIC_TO_OPTIMIZE=self.EXAMPLE_ACCURACY,
            GREATER_IS_BETTER=True,
            METRIC_NAMES=[self.ACCURACY, self.F1, self.PRECISION, self.RECALL],
            LABELS=self.PDTB_E_LABELS
        )
        self.PDTB_I_TASK = DiscoEvalTaskConfig(
            PREFIX=f'{self.PDTB_I}: ',
            TEXT_COLUMN_AMOUNT=self.PDTB_I_TEXT_COLUMNS,
            LABEL_COLUMN_NAME=self.LABEL,
            METRIC_TO_OPTIMIZE=self.EXAMPLE_ACCURACY,
            GREATER_IS_BETTER=True,
            METRIC_NAMES=[self.ACCURACY, self.F1, self.PRECISION, self.RECALL],
            LABELS=self.PDTB_I_LABELS
        )
        self.SSPABS_TASK = DiscoEvalTaskConfig(
            PREFIX=f'{self.SSPABS}: ',
            TEXT_COLUMN_AMOUNT=self.SSPABS_TEXT_COLUMNS,
            LABEL_COLUMN_NAME=self.LABEL,
            METRIC_TO_OPTIMIZE=self.EXAMPLE_ACCURACY,
            GREATER_IS_BETTER=True,
            METRIC_NAMES=[self.ACCURACY, self.F1, self.PRECISION, self.RECALL],
            LABELS=self.SSPABS_LABELS
        )

    def __getitem__(self, item):
        if item == self.SPARXIV:
            return self.SPARXIV_TASK
        elif item == self.SPROCSTORY:
            return self.SPROCSTORY_TASK
        elif item == self.SPWIKI:
            return self.SPWIKI_TASK
        elif item == self.BSOARXIV:
            return self.BSOARXIV_TASK
        elif item == self.BSOROCSTORY:
            return self.BSOROCSTORY_TASK
        elif item == self.BSOWIKI:
            return self.BSOWIKI_TASK
        elif item == self.DCCHAT:
            return self.DCCHAT_TASK
        elif item == self.DCWIKI:
            return self.DCWIKI_TASK
        elif item == self.RST:
            return self.RST_TASK
        elif item == self.PDTB_E:
            return self.PDTB_E_TASK
        elif item == self.PDTB_I:
            return self.PDTB_I_TASK
        elif item == self.SSPABS:
            return self.SSPABS_TASK
        else:
            raise KeyError(f'{item} is not a valid DiscoEval task')
