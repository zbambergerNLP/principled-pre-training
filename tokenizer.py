import typing
import torch
import transformers
from transformers import T5Tokenizer


def tokenize_function(
        examples: typing.Dict[str, typing.Any],
        tokenizer: transformers.PreTrainedTokenizer,
        input_column_name: str = 'sentence',
        target_column_name: str = 'label',
        input_max_length: int = 512,
        target_max_length: int = 512,
) -> typing.Dict[str, torch.Tensor]:
    """
    Tokenizes batches of examples for an encoder-decoder model.

    Args:
        examples: A batch in the form of a dictionary mapping, mapping column names to their respective values.
        tokenizer: A function which converts string tokens into input_ids and other model inputs.
        input_column_name: Name of the column within the input dictionary that contains the text which will be
            tokenized.
        target_column_name: Name of the column within the input dictionary that contains the labels which will be
            tokenized.
        input_max_length: The maximum length of the input sequence.
        target_max_length: The maximum length of the target sequence.

    Returns:
        A dictionary containing the original mappings, as well as the mapping between model input names (e.g.,
            `input_ids`) and model input values (e.g., the tensor corresponding to the input IDs of the model).
    """
    inputs = examples[input_column_name]
    encoding = tokenizer(
        inputs,
        padding='max_length',
        max_length=input_max_length,
        truncation=True,
        return_tensors="pt",
    )
    results = {'input_ids': encoding.input_ids, 'attention_mask': encoding.attention_mask}

    # Labels are not preprocessed for the T5 model. model_inputs are returned as is
    outputs = examples[target_column_name]
    labels = tokenizer(
        outputs,
        padding='max_length',
        max_length=target_max_length,
        truncation=True,
        return_tensors="pt",
    )['input_ids']

    # Replace the padding token with -100 to ignore it for loss computation
    labels[labels == tokenizer.pad_token_id] = -100
    results['labels'] = labels
    return results


def tokenize_function_sentence_positioning(
        examples: typing.Dict[str, typing.Any],
        tokenizer: transformers.PreTrainedTokenizer,
        input_column_name: str = 'sentence',
        target_column_name: str = 'label',
        input_max_length: int = 512,
        target_max_length: int = 512,
) -> typing.Dict[str, torch.Tensor]:
    """
    Tokenizes batches of examples for an encoder-decoder model.

    Args:
        examples: A batch in the form of a dictionary mapping, mapping column names to their respective values.
        tokenizer: A function which converts string tokens into input_ids and other model inputs.
        input_column_name: Name of the column within the input dictionary that contains the text which will be
            tokenized.
        target_column_name: Name of the column within the input dictionary that contains the labels which will be
            tokenized.
        input_max_length: The maximum length of the input sequence.
        target_max_length: The maximum length of the target sequence.

    Returns:
        A dictionary containing the original mappings, as well as the mapping between model input names (e.g.,
            `input_ids`) and model input values (e.g., the tensor corresponding to the input IDs of the model).
    """
    from transformers import BatchEncoding

    def combine_batch_encodings(batch_encodings_list: list[BatchEncoding]) -> BatchEncoding:
        # Initialize an empty dictionary for combined encodings
        combined_encodings = {}

        # Iterate over each BatchEncoding object
        for encodings in batch_encodings_list:
            for key, value in encodings.items():
                if key not in combined_encodings:
                    combined_encodings[key] = []
                combined_encodings[key].extend(value)

        return BatchEncoding(combined_encodings)


    inputs = examples[input_column_name]
    encodings = []
    for sample in inputs:
        encodings.append(
            tokenizer(
                sample,
                padding='max_length',
                max_length=input_max_length,
                truncation=True,
                return_tensors="pt",
            )
        )
    encoding = combine_batch_encodings(encodings)
    results = {'input_ids': encoding.input_ids, 'attention_mask': encoding.attention_mask}

    # Labels are not preprocessed for the T5 model. model_inputs are returned as is
    outputs = examples[target_column_name]
    labels = tokenizer(
        outputs,
        padding='max_length',
        max_length=target_max_length,
        truncation=True,
        return_tensors="pt",
    )['input_ids']

    # Replace the padding token with -100 to ignore it for loss computation
    labels[labels == tokenizer.pad_token_id] = -100
    results['labels'] = labels
    return results


def tokenizer_function_one_input(
        examples: typing.Dict[str, typing.Any],
        tokenizer: T5Tokenizer,
        label_names: typing.Dict[int, str],
        prefix: str,
        text_column_name: str = 'sentence',
        label_column_name: str = 'label',
        input_max_length: int = 512,
        target_max_length: int = 512,
) -> typing.Dict[str, torch.Tensor]:
    """
    Tokenizes batches of examples with only a single textual input for an encoder-decoder model.

    Args:
        examples: A batch in the form of a dictionary mapping, mapping column names to their respective values.
        tokenizer: A function which converts string tokens into input_ids and other model inputs.
        label_names: A dictionary mapping from the integer representation of the label to the string representation.
        prefix: The string prefix prepended to each textual example. (This is task specific)
        text_column_name: Name of the column within the input dictionary that contains the text which will be tokenized.
        label_column_name: Name of the column within the input dictionary that contains the labels which will be
            tokenized.
        input_max_length: The maximum length of the input sequence.
        target_max_length: The maximum length of the target sequence.

    Returns:
        A dictionary containing the original mappings, as well as the mapping between model input names (e.g.,
            `input_ids`) and model input values (e.g., the tensor corresponding to the input IDs of the model).
    """
    inputs = [f"{prefix}{sentence}" for sentence in examples[text_column_name]]
    encoding = tokenizer(
        inputs,
        padding='max_length',
        max_length=input_max_length,
        truncation=True,
        return_tensors="pt",
    )
    results = {'input_ids': encoding.input_ids, 'attention_mask': encoding.attention_mask}

    # Labels are not preprocessed for the T5 model. model_inputs are returned as is
    outputs = [label_names[label] for label in examples[label_column_name]]
    labels = tokenizer(
        outputs,
        padding='max_length',
        max_length=target_max_length,
        truncation=True,
        return_tensors="pt",
    )['input_ids']

    # Replace the padding token with -100 to ignore it for loss computation
    labels[labels == tokenizer.pad_token_id] = -100
    results['labels'] = labels
    return results


def tokenizer_function_two_input(
        examples: typing.Dict[str, typing.Any],
        tokenizer: T5Tokenizer,
        label_names: typing.Dict[int, str],
        prefix_1: str,
        prefix_2: str,
        text_column_name_1: str = 'sentence1',
        text_column_name_2: str = 'sentence2',
        label_column_name: str = 'label',
        is_regression: bool = False,
        input_max_length: int = 512,
        target_max_length: int = 512,
) -> typing.Dict[str, torch.Tensor]:
    """
    Tokenizes batches of examples with only a single textual input for an encoder-decoder model.

    This tokenizer function merges two sentences along with their corresponding prefixes. For example, given the first
    sentence "I love NLP." and the second sentence "You too?", the combination would be:

    "stsb sentence1: I love NLP. sentence2: You too?"

    Args:
        examples: A batch in the form of a dictionary mapping, mapping column names to their respective values.
        tokenizer: A function which converts string tokens into input_ids and other model inputs.
        label_names: A dictionary mapping from the integer representation of the label to the string representation.
        prefix_1: The first string prefix prepended to each textual example. (This is task specific)
        prefix_2: The second string prefix prepended to each textual example.
        text_column_name_1: Name of the first column within the input dictionary that contains the text which will be
            tokenized.
        text_column_name_2: Name of the second column within the input dictionary that contains the text which will be
            tokenized.
        label_column_name: Name of the column within the input dictionary that contains the labels which will be
            tokenized.
        is_regression: True if task is a regression task, False if task is classification task.
        input_max_length: The maximum length of the input sequence.
        target_max_length: The maximum length of the target sequence.
    Returns:
        A dictionary containing the original mappings, as well as the mapping between model input names (e.g.,
            `input_ids`) and model input values (e.g., the tensor corresponding to the input IDs of the model).
    """
    # TODO: Add max length for inputs and targets as parameters.
    inputs_1 = [f"{prefix_1}{sentence}" for sentence in examples[text_column_name_1]]
    inputs_2 = [f"{prefix_2}{sentence}" for sentence in examples[text_column_name_2]]
    inputs = [f"{sent1} {sent2}" for sent1, sent2 in zip(inputs_1, inputs_2)]
    encoding = tokenizer(
        inputs,
        padding='max_length',
        max_length=input_max_length,
        truncation=True,
        return_tensors="pt",
    )
    results = {'input_ids': encoding.input_ids, 'attention_mask': encoding.attention_mask}

    if is_regression:  # Training task involves predicting continuous values
        outputs = [str(round(example, 1)) for example in examples[label_column_name]]
    else:  # Training task involves predicting a label from a predefined set of possible labels.
        outputs = [label_names[example] for example in examples[label_column_name]]

    # Seq2seq models expect labels in the form of tokenized text (multi-class prediction).
    labels = tokenizer(
        outputs,
        padding='max_length',
        max_length=target_max_length,
        truncation=True,
        return_tensors="pt",
    )['input_ids']

    # Replace the padding token with -100 to ignore it for loss computation
    labels[labels == tokenizer.pad_token_id] = -100
    results['labels'] = labels
    return results


def tokenizer_function_t5_pre_training(
        examples: typing.Dict[str, typing.List[str]],
        tokenizer: T5Tokenizer,
        text_column_name: str = 'text',
) -> transformers.tokenization_utils_base.BatchEncoding:
    """
    Tokenizes batches of examples for pre-training a T5 model.

    Args:
        examples: A batch in the form of a dictionary mapping, mapping column names to their respective values.
        tokenizer: A function which converts string tokens into input_ids and other model inputs.
        text_column_name: Name of the column within the input dictionary that contains the text which will be
            tokenized.

    Returns:
        A dictionary containing the original mappings, as well as the mapping between model input names (e.g.,
            `input_ids`) and model input values (e.g., the tensor corresponding to the input IDs of the model).
    """
    batch_encoding = tokenizer(
        text=examples[text_column_name],
        max_length=tokenizer.model_max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt',
    )
    return batch_encoding
