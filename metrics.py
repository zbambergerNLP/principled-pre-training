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
        tokenizer: transformers.PreTrainedTokenizer,
) -> typing.Dict[str, float]:
    """Compute the accuracy of the model.

    Args:
        eval_pred: A namedtuple containing the model predictions and labels.
        metric_names: The names of the metrics to be used for evaluation on a benchmark task.
        tokenizer: The tokenizer used to encode the inputs and labels.

    Returns:
        A dictionary containing the accuracy of the model.
    """
    predictions, labels = eval_pred
    predictions: np.ndarray  # Shape is [batch_size, target_sequence_length]
    labels: np.ndarray       # Shape is [batch_size, target_sequence_length]
    metrics = {}
    labels[labels == -100] = tokenizer.pad_token_id

    if predictions[:, 0].max() == tokenizer.pad_token_id:  # Check if the first token in the predictions is the padding token
        # Skip the first token in the predictions (i.e., the decoder start token), and add a padding token at the end
        predictions = np.concatenate(
            [predictions[:, 1:],
             np.full(
                 (predictions.shape[0], 1),
                 tokenizer.pad_token_id)
             ],
            axis=1,
        )

    is_correct = np.equal(predictions, labels)
    num_correct_per_example = is_correct.sum(axis=1)
    ideal_num_correct_per_example = np.ones_like(num_correct_per_example) * labels.shape[1]
    example_is_correct = np.equal(num_correct_per_example, ideal_num_correct_per_example)

    predictions = predictions[(labels != tokenizer.pad_token_id) & (labels != tokenizer.eos_token_id)]
    labels = labels[(labels != tokenizer.pad_token_type_id) & (labels != tokenizer.eos_token_id)]

    # Get the metrics!
    for metric_name in metric_names:
        # Metrics from scipy return `statistic` and `pvalue`, but we are only interested in the statistic.
        if metric_name == 'pearson' or metric_name == 'spearman':
            # Get the statistic (not the pvalue)
            metrics[f'token_{metric_name}'] = METRIC_NAME_TO_FUNC[metric_name](labels, predictions)[0]
            metrics[f'example_{metric_name}'] = METRIC_NAME_TO_FUNC[metric_name](
                example_is_correct, np.ones_like(example_is_correct))[0]
        # Multiply mcc by 100 to remain consistent with the original T5 implementation:
        # https://github.com/google-research/text-to-text-transfer-transformer/blob/main/t5/data/glue_utils.py#L140
        elif metric_name == 'mcc':
            metrics[f'token_{metric_name}'] = METRIC_NAME_TO_FUNC[metric_name](labels, predictions) * 100
            metrics[f'example_{metric_name}'] = METRIC_NAME_TO_FUNC[metric_name](
                example_is_correct, np.ones_like(example_is_correct)) * 100
        else:
            metrics[f'token_{metric_name}'] = METRIC_NAME_TO_FUNC[metric_name](labels, predictions)
            metrics[f'example_{metric_name}'] = METRIC_NAME_TO_FUNC[metric_name](
                example_is_correct, np.ones_like(example_is_correct))
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
