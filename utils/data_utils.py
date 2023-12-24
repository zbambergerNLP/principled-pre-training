import typing
from dataclasses import dataclass
import omegaconf
import transformers
import datasets
import numpy as np
import torch

from constants import (
    base_constants,
)

from utils import (
    logging_utils,
    tokenizer_utils,
)

# TODO: Create a unit test suite for functions and classes in this file.


# TODO: Create an alternative function for alternative pre-training objectives/mixtures. For example, UL2, DEPTH, and
#  other pre-training objectives have different approaches for computing the input and target lengths.
def compute_input_and_target_lengths(
        inputs_length: int,
        noise_density: float,
        mean_noise_span_length: float,
) -> typing.Tuple[int, int]:
    """This function is copy of `random_spans_helper:
    https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2466>

    Copied from:
    https://github.com/huggingface/transformers/blob/main/examples/flax/language-modeling/run_t5_mlm_flax.py

    Training parameters to avoid padding with random_spans_noise_mask. When training a model with
    random_spans_noise_mask, we would like to set the other training hyperparmeters in a way that avoids padding. This
    function helps us compute these hyperparameters. We assume that each noise span in the input is replaced by
    extra_tokens_per_span_inputs sentinel tokens, and each non-noise span in the targets is replaced by
    extra_tokens_per_span_targets sentinel tokens. This function tells us the required number of tokens in the
    raw example (for split_tokens()) as well as the length of the encoded targets. Note that this function assumes the
    inputs and targets will have EOS appended and includes that in the reported length.

    :param inputs_length: an integer - desired length of the tokenized inputs sequence
    :param noise_density: a float - approximate density of output mask
    :param mean_noise_span_length: a float - average length of a span of masked tokens
    :return: A tuple of integers (inputs_length, targets_length) where:
        tokens_length: length of original text in tokens
        targets_length: an integer - length in tokens of encoded targets sequence
    """

    def _tokens_length_to_inputs_length_targets_length(
            tokens_length: int,
    ) -> typing.Tuple[int, int]:
        """Computes the inputs_length and targets_length for a given tokens_length.

        :param tokens_length: an integer - desired length of the tokenized inputs sequence
        :return: inputs_length: length of original text in tokens
        """
        num_noise_tokens = int(round(tokens_length * noise_density))
        num_nonnoise_tokens = tokens_length - num_noise_tokens
        num_noise_spans = int(round(num_noise_tokens / mean_noise_span_length))
        # inputs contain all nonnoise tokens, sentinels for all noise spans
        # and one EOS token.
        _input_length = num_nonnoise_tokens + num_noise_spans + 1
        _output_length = num_noise_tokens + num_noise_spans + 1
        return _input_length, _output_length

    tokens_length = inputs_length

    while _tokens_length_to_inputs_length_targets_length(tokens_length + 1)[0] <= inputs_length:
        tokens_length += 1

    inputs_length, targets_length = _tokens_length_to_inputs_length_targets_length(tokens_length)

    # minor hack to get the targets length to be equal to inputs length
    # which is more likely to have been set to a nice round number.
    if noise_density == 0.5 and targets_length > inputs_length:
        tokens_length -= 1
        targets_length -= 1
    return tokens_length, targets_length


@dataclass
class DataCollatorForT5MLM:
    """
    [Copied from https://github.com/huggingface/transformers/blob/main/examples/flax/language-modeling/run_t5_mlm_flax.py]
    Data collator used for T5 span-masked language modeling.
    It is made sure that after masking the inputs are of length `data_args.max_seq_length` and targets are also of fixed length.
    For more information on how T5 span-masked language modeling works, one can take a look
    at the `official paper <https://arxiv.org/pdf/1910.10683.pdf>`__
    or the `official code for preprocessing <https://github.com/google-research/text-to-text-transfer-transformer/blob/master/t5/data/preprocessors.py>`__ .
    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        noise_density (:obj:`float`):
            The probability with which to (randomly) mask tokens in the input.
        mean_noise_span_length (:obj:`float`):
            The average span length of the masked tokens.
        input_length (:obj:`int`):
            The expected input length after masking.
        target_length (:obj:`int`):
            The expected target length after masking.
        pad_token_id: (:obj:`int`):
            The pad token id of the model
        decoder_start_token_id: (:obj:`int):
            The decoder start token id of the model
    """

    tokenizer: transformers.AutoTokenizer
    noise_density: float
    mean_noise_span_length: float
    input_length: int
    target_length: int
    pad_token_id: int

    def __call__(
            self,
            examples: typing.List[typing.Dict[str, np.ndarray]]
    ) -> transformers.BatchEncoding:
        """
        Create a batch suitable for T5 span-masked language modeling. The batch will be created from a list of
        dictionaries like ``[{'input_ids': ..., 'labels': ...}, ...]``. The ``input_ids`` key will be used as the
        input sequence and the ``labels`` key will be used as the target sequence. All other keys will be ignored.
        The input sequence will be masked with a span-mask and the target sequence will be masked with a
        consecutive-mask. The input sequence will be padded to ``self.input_length`` and the target sequence will be
        padded to ``self.target_length``.

        :param examples:  A list of dictionaries like ``[{'input_ids': ..., 'labels': ...}, ...]``. The ``input_ids``
            key will be used as the input sequence and the ``labels`` key will be used as the target sequence. All
            other keys will be ignored.
        :type examples: A list of dictionaries, where keys are strings and values are numpy arrays. The keys should
            contain ``input_ids`` and ``labels``. The values of ``input_ids`` should be of type ``np.ndarray`` and
            the values of ``labels`` should be of type ``np.ndarray``.
        :return: A batch of data in the form of a ``transformers.BatchEncoding``. The batch will contain the following
            keys: ``input_ids``, ``labels``, ``decoder_input_ids``, ``decoder_attention_mask``. The values of
            ``input_ids`` and ``labels`` will be of type ``torch.Tensor`` and the values of ``decoder_input_ids`` and
            ``decoder_attention_mask`` will be of type ``torch.Tensor``. The shape of ``input_ids`` will be
            ``(batch_size, self.input_length)``. The shape of ``labels`` will be ``(batch_size, self.target_length)``.
                    """
        # convert list to dict and tensorize input
        batch = transformers.BatchEncoding(
            {
                k: np.array([examples[i][k] for i in range(len(examples))])
                for k, v in examples[0].items()
            }
        )

        input_ids = batch[base_constants.TokenizedTrainingExampleConstants.INPUT_IDS]
        batch_size, expandend_input_length = input_ids.shape

        mask_indices = np.asarray(
            [
                self.random_spans_noise_mask(expandend_input_length)
                for _ in range(batch_size)
            ]
        )
        labels_mask = ~mask_indices

        input_ids_sentinel = self.create_sentinel_ids(mask_indices=mask_indices.astype(np.int8))
        labels_sentinel = self.create_sentinel_ids(mask_indices=labels_mask.astype(np.int8))

        batch[base_constants.TokenizedTrainingExampleConstants.INPUT_IDS] = (
            self.filter_input_ids(input_ids=input_ids, sentinel_ids=input_ids_sentinel)
        )
        batch[base_constants.TokenizedTrainingExampleConstants.LABELS] = (
            self.filter_input_ids(input_ids=input_ids, sentinel_ids=labels_sentinel)
        )

        if batch[base_constants.TokenizedTrainingExampleConstants.INPUT_IDS].shape[-1] != self.input_length:
            raise ValueError(
                "`input_ids` are incorrectly preprocessed. `input_ids` length is "
                f"{batch[base_constants.TokenizedTrainingExampleConstants.INPUT_IDS].shape[-1]}, but should be "
                f"{self.input_length}."
            )

        if batch[base_constants.TokenizedTrainingExampleConstants.LABELS].shape[-1] != self.target_length:
            raise ValueError(
                "`labels` are incorrectly preprocessed. `labels` length is "
                f"{batch[base_constants.TokenizedTrainingExampleConstants.LABELS].shape[-1]}, but should be"
                f" {self.target_length}."
            )

        batch = {k: torch.from_numpy(v) for k, v in batch.items()}
        return batch

    def create_sentinel_ids(
            self,
            mask_indices: np.ndarray, # shape: (batch_size, expanded_inputs_length)
    ) -> np.ndarray:
        """
        Sentinel ids creation given the indices that should be masked.
        The start indices of each mask are replaced by the sentinel ids in increasing
        order. Consecutive mask indices to be deleted are replaced with `-1`.
        """
        start_indices = mask_indices - np.roll(a=mask_indices, shift=1, axis=-1) * mask_indices
        start_indices[:, 0] = mask_indices[:, 0]

        sentinel_ids = np.where(
            start_indices != 0,
            np.cumsum(a=start_indices, axis=-1),
            start_indices,
        )
        sentinel_ids = np.where(
            sentinel_ids != 0,
            (len(self.tokenizer) - sentinel_ids),
            0,
        )
        sentinel_ids -= mask_indices - start_indices

        return sentinel_ids

    def filter_input_ids(
            self,
            input_ids: np.ndarray,  # shape: (batch_size, expanded_inputs_length)
            sentinel_ids: np.ndarray,  # shape: (batch_size, expanded_inputs_length)
    ):
        """
        Puts sentinel mask on `input_ids` and fuse consecutive mask tokens into a single mask token by deleting.
        This will reduce the sequence length from `expanded_inputs_length` to `input_length`.
        """
        batch_size = input_ids.shape[0]

        input_ids_full = np.where(
            sentinel_ids != 0,
            sentinel_ids,
            input_ids,
        )
        # input_ids tokens and sentinel tokens are >= 0, tokens < 0 are
        # masked tokens coming after sentinel tokens and should be removed
        input_ids = input_ids_full[input_ids_full >= 0].reshape((batch_size, -1))
        input_ids = np.concatenate(
            [
                input_ids,
                np.full(
                    shape=(batch_size, 1),
                    fill_value=self.tokenizer.eos_token_id,
                    dtype=np.int32,
                ),
            ],
            axis=-1,
        )
        return input_ids

    def random_spans_noise_mask(self, length: int):
        """This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2682>`__ .

        Noise mask consisting of random spans of noise tokens.
        The number of noise tokens and the number of noise spans and non-noise spans
        are determined deterministically as follows:
        num_noise_tokens = round(length * noise_density)
        num_nonnoise_spans = num_noise_spans = round(num_noise_tokens / mean_noise_span_length)
        Spans alternate between non-noise and noise, beginning with non-noise.
        Subject to the above restrictions, all masks are equally likely.

        :param length: an int32 scalar (length of the incoming token sequence)
        :param noise_density: a float - approximate density of output mask
        :param mean_noise_span_length: a number

        :returns: a boolean tensor with shape [length]
        """

        orig_length = length

        num_noise_tokens = int(np.round(length * self.noise_density))
        # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
        num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
        num_noise_spans = int(np.round(num_noise_tokens / self.mean_noise_span_length))

        # avoid degeneracy by ensuring positive number of noise spans
        num_noise_spans = max(num_noise_spans, 1)
        num_nonnoise_tokens = length - num_noise_tokens

        # pick the lengths of the noise spans and the non-noise spans
        def _random_segmentation(num_items: int, num_segments: int) -> np.ndarray:
            """Partition a sequence of items randomly into non-empty segments.
            Args:
                num_items: an integer scalar > 0. The number of items to be partitioned. This is the sum of the lengths
                    of the segments.
                num_segments: an integer scalar in [1, num_items]. If 1, the sequence is not split. If num_items is
                    greater than 1, the sequence is split into num_segments segments (where the length of each segment
                    is at least 1).
            Returns:
                a Tensor with shape [num_segments] containing positive integers that add up to num_items
            """
            assert num_items > 0, f"Expected num_items > 0, got {num_items}"
            assert 1 <= num_segments <= num_items, f"Expected 1 <= num_segments <= num_items, got {num_segments}"

            # mask_indices is a boolean vector indicating which items will be masked. We want to place the mask
            # boundaries at the start of segments of consecutive Trues. We do this by padding the mask_indices
            # vector with a False at the beginning and finding the cumulative sum. The False at the beginning
            # ensures that the cumulative sum starts at 0.
            mask_indices = np.arange(stop=(num_items - 1)) < (num_segments - 1)
            np.random.shuffle(mask_indices)
            first_in_segment = np.pad(array=mask_indices, pad_width=[[1, 0]])
            segment_id = np.cumsum(a=first_in_segment)
            # count length of sub segments assuming that list is sorted
            _, segment_length = np.unique(ar=segment_id, return_counts=True)
            return segment_length

        # We want spans to be more than one token on average. If the mean is less than two tokens round up to two tokens
        # with probability equal to the amount that the mean is less than two.
        noise_span_lengths = _random_segmentation(num_items=num_noise_tokens, num_segments=num_noise_spans)
        nonnoise_span_lengths = _random_segmentation(num_items=num_nonnoise_tokens, num_segments=num_noise_spans)

        interleaved_span_lengths = np.reshape(
            a=np.stack(
                arrays=[nonnoise_span_lengths, noise_span_lengths],
                axis=1,
            ),
            newshape=[num_noise_spans * 2],
        )
        span_starts = np.cumsum(a=interleaved_span_lengths)[:-1]
        span_start_indicator = np.zeros(shape=(length,), dtype=np.int8)
        span_start_indicator[span_starts] = True
        span_num = np.cumsum(a=span_start_indicator)
        is_noise = np.equal(span_num % 2, 1)

        return is_noise[:orig_length]

def load_dataset_splits(
        args: omegaconf.DictConfig,
        logger: logging_utils.Logger,
) -> typing.Dict[str, datasets.Dataset]:
    """
    Load the splits of the dataset (e.g., train, test, validation).

    :param args: The omegaconf configuration used to generate the dataset splits.
    :param logger: A logging_utils.Logger object. See `logging_utils.py` for more details.
    :return: A dictionary of the dataset splits.
    """
    logger.log_message(f'Loading dataset {args.dataset.dataset_name} from {args.dataset.dataset_path}')
    dataset = datasets.load_dataset(
        'c4',
        'en',
        streaming=True,
    )
    # dataset = datasets.load_dataset(
    #     path=args.dataset.dataset_path,
    #     name=args.dataset.dataset_name,
    #     streaming=args.dataset.streaming,
    #     # If we are using a streaming dataset, then we want to use the entire dataset. Otherwise, we want to use a
    #     # fraction of the dataset.
    #     split=None if args.dataset.streaming else f'train[:{args.dataset.dataset_fraction}%]',
    # )
    dataset = dataset.remove_columns(args.dataset.columns_to_remove)
    dataset_splits = {
        base_constants.SplitConstants.TRAIN: dataset[base_constants.SplitConstants.TRAIN],
        base_constants.SplitConstants.TEST: dataset[base_constants.SplitConstants.VALIDATION],
    }

    assert (
            dataset['train'].n_shards == 1024  # TODO: Make this a flag
    ), "We want to have many shards for efficient processing with num_workes in PyTorch dataloader"

    return dataset_splits


def process_dataset(
        dataset_splits: typing.Dict[str, datasets.Dataset],
        args: omegaconf.DictConfig,
        tokenizer: transformers.AutoTokenizer,
        logger: logging_utils.Logger,
) -> typing.Dict[str, datasets.Dataset]:
    """
    Process the dataset splits (e.g., tokenize the inputs and outputs).

    :param dataset_splits: A dictionary of the dataset splits. The keys are the split names (e.g., train, test,
        validation) and the values are the dataset splits (i.e., a HuggingFace Dataset object).
    :param args: The omegaconf configuration used to process the dataset splits.
    :param tokenizer: The tokenizer used to tokenize the inputs and outputs.
    :param logger: A logging_utils.Logger object. See `logging_utils.py` for more details.
    :return: A dictionary of the processed dataset splits.
    """
    logger.log_message('Processing dataset splits')
    final_datasets = {}

    for split, dataset_split in dataset_splits.items():
        # We increase the input_length, because instead of masking tokens T5 replaces
        # masked spans with a single token, therefore to avoid padding we need to have
        # longer sequences at the start, before masking
        before_mask_input_length, target_length = compute_input_and_target_lengths(
            inputs_length=args.data.input_length,
            noise_density=args.data.mlm_probability,
            mean_noise_span_length=args.data.mean_noise_span_length,
        )

        with omegaconf.open_dict(args):
            args.data.before_mask_input_length = before_mask_input_length
            args.data.target_length = target_length

        dataset_split = dataset_split.map(
            tokenizer_utils.tokenize_function,
            batched=True,
            fn_kwargs={
                base_constants.TokenizerConstants.TOKENIZER: tokenizer,
                base_constants.TokenizerConstants.IN_LENGTH: before_mask_input_length,
            },
            remove_columns=[base_constants.RawTrainingExampleConstants.TEXT_COLUMN_NAME],
        )

        dataset_split = dataset_split.shuffle(buffer_size=10_000, seed=args.seed)
        final_datasets[split] = dataset_split

    return final_datasets


# TODO: Replace this function with the data collator from data_collator_t5.py.
def get_data_collator(
        tokenizer: transformers.AutoTokenizer,
        config: transformers.AutoConfig,
        args: omegaconf.DictConfig,
        logger: logging_utils.Logger,
) -> DataCollatorForT5MLM:
    """
    Get the data collator. This is used to collate the data into batches.

    :param tokenizer: The tokenizer used to tokenize the inputs and outputs.
    :param config: The model configuration. See `get_config` for more details.
    :param args: The omegaconf configuration used to generate the data collator.
    :param logger: The logger. See `logging_utils.py` for more details.
    :return: The data collator.
    """
    logger.log_message('Creating data collator')
    # TODO: Enable custom data collator. Condition the data collator on the model type (e.g., T5 vs. UL2  vs. DEPTH).
    data_collator = DataCollatorForT5MLM(
        tokenizer=tokenizer,
        noise_density=args.data.mlm_probability,
        mean_noise_span_length=args.data.mean_noise_span_length,
        input_length=args.data.input_length,
        target_length=args.data.target_length,
        pad_token_id=config.pad_token_id,
    )
    return data_collator



def get_dataloaders(
        tokenizer: transformers.AutoTokenizer,
        config: transformers.AutoConfig,
        args: omegaconf.DictConfig,
        logger: logging_utils.Logger,
) -> typing.Tuple[datasets.Dataset, datasets.Dataset]:
    """
    Get the dataloaders. This is used to load the data into batches.
    :param tokenizer: The tokenizer used to tokenize the inputs and outputs.
    :param config: The model configuration. See `get_config` for more details.
    :param args: The omegaconf configuration used to generate the dataloaders.
    :param logger: The logger. See `logging_utils.py` for more details.
    :return: The dataloaders. The first element is the training dataloader and the second element is the
        test dataloader.
    """
    logger.log_message('Loading dataset splits')
    dataset_splits = load_dataset_splits(args, logger=logger)
    dataset = process_dataset(dataset_splits=dataset_splits, args=args, tokenizer=tokenizer, logger=logger)
    data_collator = get_data_collator(tokenizer=tokenizer, config=config, args=args, logger=logger)
    is_iterable = isinstance(dataset[base_constants.SplitConstants.TRAIN], torch.utils.data.IterableDataset)
    logger.log_message(f'Is dataset iterable: {is_iterable}')

    dataloaders = {}

    for split in [base_constants.SplitConstants.TRAIN, base_constants.SplitConstants.TEST]:
        batch_size = args.optim.batch_size // args.optim.grad_acc

        shuffle = (split == base_constants.SplitConstants.TRAIN) and not is_iterable
        logger.log_message(f'\tShuffle {split} data: {shuffle}')

        dataloaders[split] = torch.utils.data.DataLoader(
            dataset[split],
            shuffle=shuffle,
            collate_fn=data_collator,
            batch_size=batch_size,
            num_workers=args.data.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    # Add & Check args about data loaders
    with omegaconf.open_dict(args):
        if not is_iterable:
            args.data.train_batches = len(dataloaders[base_constants.SplitConstants.TRAIN])
            args.data.test_batches = len(dataloaders[base_constants.SplitConstants.TEST])

        if args.optim.epochs > 0:
            assert not is_iterable
            args.optim.total_steps = (
                    (len(dataloaders[base_constants.SplitConstants.TRAIN]) // args.optim.grad_acc) * args.optim.epochs
            )

        args.evaluate.corrected_steps = args.evaluate.steps

    return dataloaders[base_constants.SplitConstants.TRAIN], dataloaders[base_constants.SplitConstants.TEST]

