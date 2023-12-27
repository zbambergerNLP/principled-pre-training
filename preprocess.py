from constants.disco_eval_constants import DiscoEvalConstants
from constants.glue_constants import GlueConstants
import constants.glue_constants as glue_constants
import typing
from constants.base_constants import *

dec = DiscoEvalConstants()
def preprocess_function_n_inputs(
        examples: typing.Dict[str, typing.Any],
        label_names: typing.Dict[int, str],
        task_name: str,
        label_column_name: str,
) -> typing.Dict[str, typing.List[str]]:
    """
    Pre-processes batches of examples with two textual inputs for an encoder-decoder model.

    Args:
        examples: A batch in the form of a dictionary mapping, mapping column names to their respective values.
        label_names: A dictionary mapping from the integer representation of the label to the string representation.
        task_name: The name of the task (i.e. SPArxib/RST/etc.).
        label_column_name: Name of the column within the input dictionary that contains the labels text.
    Returns:
        A dictionary containing the original mappings, as well as mappings to processed inputs and outputs.
    """

    outputs = [label_names[str(example)] for example in examples[label_column_name]]
    examples.pop(label_column_name)
    examples_values = dict(examples).values()  # takes the values for each text column of the dataset
    transposed_values = list(zip(*examples_values))  # transposes a matrix (list of lists)
    inputs = [[f"{dec.TEXT_COLUMN_NAMES[i]}: {sent}" for i, sent in enumerate(exmple)] for exmple in transposed_values]
    inputs = ["\t".join(exmple) for exmple in inputs]
    inputs = [f"{task_name}: {exmple}" for exmple in inputs]
    result = {'processed_inputs': inputs, 'processed_outputs': outputs}
    return result


def preprocess_function_one_input(
        examples: typing.Dict[str, typing.Any],
        label_names: typing.Dict[int, str],
        prefix: str,
        text_column_name: str = GlueConstants.SENTENCE,
        label_column_name: str = TokenizedExampleColumnNames.LABEL.value,
) -> typing.Dict[str, typing.List[str]]:
    """
    Pre-processes batches of examples with only a single textual input for an encoder-decoder model.

    Args:
        examples: A batch in the form of a dictionary mapping, mapping column names to their respective values.
        label_names: A dictionary mapping from the integer representation of the label to the string representation.
        prefix: The string prefix prepended to each textual example. (This is task specific)
        text_column_name: Name of the column within the input dictionary that contains the text.
        label_column_name: Name of the column within the input dictionary that contains the labels text.

    Returns:
        A dictionary containing the original mappings, as well as mappings to processed inputs and outputs.
    """
    inputs = [f"{prefix}{sentence}" for sentence in examples[text_column_name]]
    outputs = [label_names[example] for example in examples[label_column_name]]
    result = {'processed_inputs': inputs, 'processed_outputs': outputs}
    return result


def preprocess_function_two_inputs(
        examples: typing.Dict[str, typing.Any],
        label_names: typing.Dict[int, str],
        prefix_1: str,
        prefix_2: str,
        text_column_name_1: str,
        text_column_name_2: str,
        label_column_name: str,
        is_regression: bool = False,
) -> typing.Dict[str, typing.List[str]]:
    """
    Pre-processes batches of examples with two textual inputs for an encoder-decoder model.

    Args:
        examples: A batch in the form of a dictionary mapping, mapping column names to their respective values.
        label_names: A dictionary mapping from the integer representation of the label to the string representation.
        prefix_1: The string prefix prepended to the first textual example. (This is task specific)
        prefix_2: The string prefix prepended to the second textual example.
        text_column_name_1: Name of the first column within the input dictionary that contains the text.
        text_column_name_2: Name of the second column within the input dictionary that contains the text.
        label_column_name: Name of the column within the input dictionary that contains the labels text.
        is_regression: Whether the task is a regression task or not.

    Returns:
        A dictionary containing the original mappings, as well as mappings to processed inputs and outputs.
    """
    inputs_1 = [f"{prefix_1}{sentence}" for sentence in examples[text_column_name_1]]
    inputs_2 = [f"{prefix_2}{sentence}" for sentence in examples[text_column_name_2]]
    inputs = [f"{sent1} {sent2}" for sent1, sent2 in zip(inputs_1, inputs_2)]

    if is_regression:  # Training task involves predicting continuous values
        outputs = [str(round(example, 1)) for example in examples[label_column_name]]
    else:  # Training task involves predicting a label from a predefined set of possible labels.
        outputs = [label_names[example] for example in examples[label_column_name]]
    result = {'processed_inputs': inputs, 'processed_outputs': outputs}
    return result


def create_preprocess_function_one_input(
        label_names: typing.Dict[int, str],
        prefix: str,
        text_column_name,
        label_column_name,
) -> typing.Callable[[typing.Dict[str, typing.Any]], typing.Dict[str, typing.List[str]]]:
    """
    Creates a pre-processing function for batches of examples with only a single textual input for an encoder-decoder
    model.

    Args:
        label_names: A dictionary mapping from the integer representation of the label to the string representation.
        prefix: The string prefix prepended to each textual example. (This is task specific)
        text_column_name: Name of the column within the input dictionary that contains the text.
        label_column_name: Name of the column within the input dictionary that contains the labels text.

    Returns:
        A pre-processing function for batches of examples with only a single textual input for an encoder-decoder model.
    """

    def preprocess_function(examples: typing.Dict[str, typing.Any]) -> typing.Dict[str, typing.List[str]]:
        return preprocess_function_one_input(
            examples=examples,
            label_names=label_names,
            prefix=prefix,
            text_column_name=text_column_name,
            label_column_name=label_column_name,
        )

    return preprocess_function


def create_preprocess_function_two_inputs(
        label_names: typing.Dict[int, str],
        prefix_1: str,
        prefix_2: str,
        text_column_name_1,
        text_column_name_2,
        label_column_name,
        is_regression: bool = False,
) -> typing.Callable[[typing.Dict[str, typing.Any]], typing.Dict[str, typing.List[str]]]:
    """
    Creates a pre-processing function for batches of examples with two textual inputs for an encoder-decoder model.

    Args:
        label_names: A dictionary mapping from the integer representation of the label to the string representation.
        prefix_1: The string prefix prepended to the first textual example. (This is task specific)
        prefix_2: The string prefix prepended to the second textual example.
        text_column_name_1: Name of the first column within the input dictionary that contains the text.
        text_column_name_2: Name of the second column within the input dictionary that contains the text.
        label_column_name: Name of the column within the input dictionary that contains the labels text.
        is_regression: Whether the task is a regression task or not.

    Returns:
        A pre-processing function for batches of examples with two textual inputs for an encoder-decoder model.
    """

    def preprocess_function(examples: typing.Dict[str, typing.Any]) -> typing.Dict[str, typing.List[str]]:
        return preprocess_function_two_inputs(
            examples=examples,
            label_names=label_names,
            prefix_1=prefix_1,
            prefix_2=prefix_2,
            text_column_name_1=text_column_name_1,
            text_column_name_2=text_column_name_2,
            label_column_name=label_column_name,
            is_regression=is_regression,
        )

    return preprocess_function


def create_preprocess_function_n_inputs(
        label_names: typing.Dict[int, str],
        task_name: str,
        label_column_name,
) -> typing.Callable[[typing.Dict[str, typing.Any]], typing.Dict[str, typing.List[str]]]:
    def preprocess_function(examples: typing.Dict[str, typing.Any]) -> typing.Dict[str, typing.List[str]]:
        return preprocess_function_n_inputs(
            examples=examples,
            label_names=label_names,
            task_name=task_name,
            label_column_name=label_column_name,
        )

    return preprocess_function


def create_preprocess_function(
        dataset_info: typing.Union[glue_constants.TaskConfigOneInput, glue_constants.TaskConfigTwoInput],
        dataset_name: str,
        is_regression: bool = False,
) -> typing.Callable[
    [typing.Dict[str, typing.Any]],
    typing.Dict[str, typing.List[str]]
]:
    """
    Create a function to pre-process the examples within the specified dataset.

    Preprocessing often involves the following steps:
        1. Adding prefixes to the input/s (still represented as strings, yet to be tokenized)
        2. Converting the label from a numerical value to the predetermined string equivalent. For example, in SST2,
            the label 0 corresponds with 'negative' and the label '1' corresponds with 'positive'.

    Args:
        dataset_info: A dictionary representation of the dataset's metadata. Includes a mapping between integer labels
            and their corresponding names, the prefixes to prepend to textual inputs, and the names of the input and
            label text columns.
        dataset_name: The name of the dataset that is processed by this function.
        is_regression: Whether the task is a regression task or not.

    Returns:
        A function that takes in a batch of input examples, and returns a dictionary with the processed inputs and
        labels. Note that the original batch of input example might include additional columns.

    Raises:
        RuntimeError if the dataset information is not formatted correctly.
    """
    label_names = dataset_info.LABELS
    label_column_name = dataset_info.LABEL_COLUMN_NAME
    if dataset_name in GlueConstants.TASKS:  # This refers to the GLUE and SUPERGLUE benchmarks.
        if isinstance(dataset_info, glue_constants.TaskConfigOneInput):
            return create_preprocess_function_one_input(
                label_names=label_names,
                label_column_name=label_column_name,
                prefix=dataset_info.PREFIX,
                text_column_name=dataset_info.TEXT_COLUMN_NAME,
            )
        elif isinstance(dataset_info, glue_constants.TaskConfigTwoInput):
            return create_preprocess_function_two_inputs(
                label_names=label_names,
                label_column_name=label_column_name,
                prefix_1=dataset_info.PREFIX_1,
                prefix_2=dataset_info.PREFIX_2,
                text_column_name_1=dataset_info.TEXT_COLUMN_NAME_1,
                text_column_name_2=dataset_info.TEXT_COLUMN_NAME_2,
                is_regression=(is_regression or dataset_name == 'stsb')
            )
        else:
            raise RuntimeError(
                "Unsupported prefix structure. Must contain either `prefix` for single input tasks or `prefix_1` and "
                "`prefix_2` for two input tasks"
            )
    elif dataset_name in DiscoEvalConstants.TASKS:  # This refers to the DiscoEval benchmark.
        return create_preprocess_function_n_inputs(
            label_names=label_names,
            task_name=dataset_name,
            label_column_name=label_column_name,
        )
