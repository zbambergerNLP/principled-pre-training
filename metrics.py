import typing
import numpy as np
import scipy.stats as stats
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef
import transformers

METRIC_NAME_TO_FUNC = {
    'accuracy': accuracy_score,
    'f1': lambda labels, prediction: f1_score(
        y_true=labels,
        y_pred=prediction,
        average='micro',
    ),
    'precision': lambda labels, prediction: precision_score(
        y_true=labels,
        y_pred=prediction,
        average='micro',
    ),
    'recall': lambda labels, prediction: recall_score(
        y_true=labels,
        y_pred=prediction,
        average='micro',
    ),
    'mcc': matthews_corrcoef,
    'pearson': stats.pearsonr,
    'spearman': stats.spearmanr,
}


def compute_metrics(
        eval_pred: transformers.EvalPrediction,
        metric_names: typing.List[str],
        padding_token: int = 0,
        eos_token: int = 1,
) -> typing.Dict[str, float]:
    """Compute the accuracy of the model.

    Args:
        eval_pred: A namedtuple containing the model predictions and labels.
        metric_names: The names of the metrics to be used for evaluation on a benchmark task.
        padding_token: Token_id of the padding token.
        eos_token: Token_id of the end of sentence token.

    Returns:
        A dictionary containing the accuracy of the model.
    """
    predictions, labels = eval_pred
    predictions: np.ndarray  # Shape is [batch_size, target_sequence_length]
    labels: np.ndarray       # Shape is [batch_size, target_sequence_length]

    # Convert padding tokens from -100 to the tokenizer's padding token ID (typically 0)
    labels[labels == -100] = padding_token

    if predictions[:, 0].max() == padding_token:  # Check if the first token in the predictions is the padding token
        # Skip the first token in the predictions (i.e., the decoder start token), and add a padding token at the end
        predictions = np.concatenate(
            [predictions[:, 1:], np.full((predictions.shape[0], 1), padding_token)],
            axis=1)

    # Flatten the predictions and labels. Ignore the padding tokens and the end of sentence tokens.
    predictions = predictions[(labels != padding_token) & (labels != eos_token)].flatten()
    labels = labels[(labels != padding_token) & (labels != eos_token)].flatten()

    # Get the metrics!
    metrics = {}
    for metric_name in metric_names:
        # Metrics from scipy return `statistic` and `pvalue`, but we are only interested in the statistic.
        if metric_name == 'pearson' or metric_name == 'spearman':
            # Get the statistic (not the pvalue)
            metrics[metric_name] = METRIC_NAME_TO_FUNC[metric_name](labels, predictions)[0]
        # Multiply mcc by 100 to remain consistent with the original T5 implementation:
        # https://github.com/google-research/text-to-text-transfer-transformer/blob/main/t5/data/glue_utils.py#L140
        elif metric_name == 'mcc':
            metrics[metric_name] = METRIC_NAME_TO_FUNC[metric_name](labels, predictions) * 100
        else:
            metrics[metric_name] = METRIC_NAME_TO_FUNC[metric_name](labels, predictions)
    return metrics

def preprocess_logits_for_metrics(
        logits: torch.Tensor,  # Shape is [batch_size, target_sequence_length, vocab_size]
        labels: torch.Tensor,  # Shape is [batch_size, target_sequence_length]
) -> torch.Tensor:
    """
    Original Trainer may have a memory leak.

    This is a workaround to avoid storing too many tensors that are not needed (which may cause a memory leak).

    Args:
        logits: The logits output by the model.
        labels: The labels for the model.

    Returns:
        The predictions of the model (i.e., the argmax of the logits). Shape is [batch_size, target_sequence_length].
    """
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]

    return logits.argmax(dim=-1)
