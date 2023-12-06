import os

DISCO_EVAL: str = 'OfekGlick/DiscoEval'

SSPABS = 'SSPabs'
PDTB_I = 'PDTB-I'
PDTB_E = 'PDTB-E'
SPARXIV = 'SParxiv'
SPROCSTORY = 'SProcstory'
SPWIKI = 'SPwiki'
BSOARXIV = 'BSOarxiv'
BSOROCSTORY = 'BSOrocstory'
BSOWIKI = 'BSOwiki'
DCCHAT = 'DCchat'
DCWIKI = 'DCwiki'
RST = 'RST'

DE_TASKS = [
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

GLUE_TASKS = [
    SST2,
    COLA,
    RTE,
    MNLI,
    QNLI,
    MRPC,
    QQP,
    STSB,
    MNLI_MATCHED,
    MNLI_MISMATCHED,
    WNLI,
    AX,
]
for task in DE_TASKS:
    os.system(f"fine_tune_t5.py --benchmark OfekGlick/DiscoEval --dataset_name {task}")

for task in GLUE_TASKS:
    os.system(f"fine_tune_t5.py --benchmark glue --dataset_name {task}")