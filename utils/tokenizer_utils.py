from utils import logging_utils
import omegaconf
import transformers
import typing
import numpy as np
from constants import base_constants


def get_tokenizer(
        args: omegaconf.DictConfig,
        logger: logging_utils.Logger,
) -> transformers.AutoTokenizer:
    """
    Get the tokenizer. This is used to tokenize the input data.
    :param args: The omegaconf configuration used to generate the tokenizer.
    :param logger: The logger. See `logging_utils.py` for more details.
    :return: The tokenizer.
    """
    # TODO: Enable custom tokenizer
    logger.log_message(f'Loading tokenizer')
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model.name,
        use_fast=True
    )
    tokenizer.model_max_length = int(1e9)

    return tokenizer


def tokenize_function(
        examples: typing.Dict[str, typing.List[str]],
        tokenizer: transformers.PreTrainedTokenizer,
        in_length: int,
) -> typing.Dict[str, np.ndarray]:
    """
    Tokenize the inputs and outputs of the T5 model.
    :param examples: Examples are a dictionary of lists of strings. The keys are the column names and the values are
        the list of strings in that column. Concretely, the columns are "text",
        'processed_outputs'.
    :param tokenizer: The tokenizer to use to tokenize the inputs and outputs.
    :param in_length: The length of the input sequence.
    :return: A dictionary of numpy arrays. The keys are the column names and the values are the numpy arrays of the
        tokenized inputs and outputs.
    """
    tokenizer_out = tokenizer(
        text=examples[base_constants.RawTrainingExampleConstants.TEXT_COLUMN_NAME],
        return_attention_mask=False,
    )

    input_ids = tokenizer_out[base_constants.TokenizedTrainingExampleConstants.INPUT_IDS]

    concatenated_ids = np.concatenate(input_ids)

    total_length = concatenated_ids.shape[0]
    total_length = (total_length // in_length) * in_length

    concatenated_ids = concatenated_ids[:total_length].reshape(-1, in_length)
    result = {base_constants.TokenizedTrainingExampleConstants.INPUT_IDS: concatenated_ids}

    return result