import copy

import torch
import numpy as np
from typing import (
    Dict,
    List,
    Union,
    Tuple,
    Set,
)
import string
import constants.base as const
from transformers import BatchEncoding
import re
import random
import transformers

def shift_tokens_right(
        input_ids: np.array,
        pad_token_id: int,
        decoder_start_token_id: int,
) -> np.ndarray:
    """
    Shift input ids one token to the right.

    Taken from:
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_flax_t5.py#L61

    Args:
        input_ids: An integer tensor of shape [batch_size, sequence_length].
        pad_token_id: The pad token id.
        decoder_start_token_id: The decoder start token id.

    Returns:
        An integer tensor of shape [batch_size, sequence_length].
    """
    shifted_input_ids = np.zeros_like(input_ids)
    shifted_input_ids[:, 1:] = input_ids[:, :-1]
    shifted_input_ids[:, 0] = decoder_start_token_id

    shifted_input_ids = np.where(shifted_input_ids == -100, pad_token_id, shifted_input_ids)
    return shifted_input_ids


def merge_segments(
        offset: int,
        n_grams: int,
        covered_indices: Set[int],
        whole_words: str,
        start_inds: List[int],
        segments_to_merge: List[List[int]],
        ngrams_vocab_set: Set[str],
):
    """
    Merge segments of length n_grams into a single segment if they are in the ngrams_vocab_set.
    Args:
        offset: The offset of the current segment in the whole sequence.
        n_grams: The length of the segment to merge.
        covered_indices: A set of indices that were already merged.
        whole_words: The whole sequence.
        start_inds: The indices of the start of each word in the whole sequence.
        segments_to_merge: A list of segments to merge.
        ngrams_vocab_set: The set of ngrams to use for PMI-based corruption.
    """
    # TODO: Vectorize this function.
    possible_merges = []
    for i in range(len(start_inds) - n_grams):
        segment = whole_words[start_inds[i]:start_inds[i + n_grams] - 1]
        if segment in ngrams_vocab_set:
            possible_merges.append(list(range(offset + i, offset + i + n_grams)))

    random.shuffle(possible_merges)
    for seg_inds in possible_merges:
        if len(set(seg_inds).intersection(covered_indices)) > 0:
            continue
        covered_indices.update(seg_inds)
        segments_to_merge.append(seg_inds)


def pmi_word_mask(
        input_tokens: List[str],
        pmi_vocab: Set[str],
        max_predictions=512,
        mlm_probability=0.5,
) -> List[int]:
    """
    Create a mask for PMI-based corruption for a sample.
    Initially, we map which tokens are part of the same word, and then we mask the entire word.
    Next, we mask ngrams that are in the ngrams_vocab_set, from 5 to 2 grams, while avoiding overlapping ngrams.
    Args:
        input_tokens: A tensor of tokens.
        pmi_vocab: The set of ngrams to use for PMI-based corruption.
        max_predictions: The maximum number of tokens to mask.
        mlm_probability: The probability of masking a token.
    Returns:
        A list of 0/1 in the length of the input tokens, 1 means the token should be masked.
    """
    # TODO: Vectorize this function.
    whole_words_indexes = []
    whole_words_lists = [[]]
    whole_words = whole_words_lists[0]
    for (i, token) in enumerate(input_tokens):
        if token == const.T5_START_TOKEN or token == const.T5_PAD_TOKEN:
            whole_words_lists.append(
                []  # to separate parts as we don't want them to be considered part of an ngram
            )
            whole_words = whole_words_lists[-1]
            continue
        # now, we mark the indices of token that start a whole word that can be masked
        if len(whole_words_indexes) >= 1 and not token.startswith(
                const.T5_SPACE_TOKEN) and token not in string.punctuation:
            whole_words_indexes[-1].append(i)
            whole_words[-1] = whole_words[-1] + token.strip(const.T5_SPACE_TOKEN).lower()
        else:
            whole_words_indexes.append([i])
            whole_words.append(token.strip(const.T5_SPACE_TOKEN).lower())
    offset = 0
    covered_indices = set()
    segments_to_merge = []
    for whole_words in whole_words_lists:
        if len(whole_words) == 0:
            continue
        added_offset = len(whole_words)
        whole_words = ' '.join(whole_words)
        start_inds = [0] + [m.start() + 1 for m in re.finditer(' ', whole_words)] + [len(whole_words) + 1]
        for n_grams in range(5, 1, -1):
            merge_segments(
                offset,
                n_grams,
                covered_indices,
                whole_words,
                start_inds,
                segments_to_merge,
                pmi_vocab,
            )
        offset += added_offset
    segments_to_merge.extend([i] for i in set(range(len(whole_words_indexes))).difference(covered_indices))

    candidates = []
    for seg_to_merge in segments_to_merge:
        candidates.append(sum([whole_words_indexes[i] for i in seg_to_merge], []))

    # whole_words_indexes is a list of lists of ints, each list is the
    # indices part of the whole word to be masked, i.e. the segment to be considered for masking
    random.shuffle(candidates)
    candidates = sorted(candidates, reverse=True, key=lambda x: len(x))
    num_to_predict = min(max_predictions, max(1, int(round(len(input_tokens) * mlm_probability))))
    masked_lms = []

    # aux list that index every token to the parent segment index
    # to make sure later the entire segment is masked correctly
    indexer = list(range(len(input_tokens)))
    covered_indexes = set()

    # list of 0/1 in the length of the input tokens, 1 means the token should be masked
    mask_labels = [0] * len(input_tokens)
    for index_set in candidates:
        if len(masked_lms) >= num_to_predict:
            break
        # If adding a whole-word mask would exceed the maximum number of
        # predictions, then just skip these candidates.
        if len(covered_indexes) + len(index_set) > num_to_predict:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:  # not sure how is it possible, the sets should be disjoint
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        head_index = min(index_set)
        mask_labels[head_index] = 1
        for index in index_set:
            covered_indexes.add(index)
            indexer[index] = head_index
            mask_labels[index] = 1

    return mask_labels


def pmi_noise_mask(
        examples: transformers.BatchEncoding,
        pmi_vocab: Set[str],
        tokenizer: transformers.T5Tokenizer,
) -> torch.Tensor:
    """
    Create a mask for PMI-based corruption.

    Args:
        examples: A BatchEncoding containing the input sequences, expected shapr [batch_size, input_length].
        pmi_vocab: The set of ngrams to use for PMI-based corruption.
        tokenizer: The tokenizer to use for PMI-based corruption.
    Returns:
        Returns a mask tensor (0/1) of shape [batch_size, input_length] where 1 means the token should be masked.
    """
    mask_labels = []
    for e in examples['input_ids']:
        ref_tokens = []
        for input_id in e:
            token = tokenizer._convert_id_to_token(input_id.item())
            ref_tokens.append(token)
        mask_labels_for_sample = pmi_word_mask(ref_tokens, pmi_vocab)
        mask_labels.append(mask_labels_for_sample)
    mask_labels = torch.tensor(mask_labels)
    return mask_labels


def random_spans_noise_mask(
        sequence_length: int,
        maximum_length: int,
        noise_density: float,
        mean_noise_span_length: float = 3.0,
        random_roll: bool = True):
    """Initialize spans to mask tokens from input text as part of pre-training.

    Noise mask consisting of random spans of noise tokens. The number of noise tokens and the number of noise spans
    and non-noise spans are determined deterministically as follows:
        - num_noise_tokens = round(sequence_length * noise_density)
        - num_nonnoise_spans = num_noise_spans = round(num_noise_tokens / mean_noise_span_length)
    Spans alternate between non-noise and noise, beginning with non-noise. Subject to the above restrictions, all masks
    are equally likely.

    Adopted from:
    https://github.com/google-research/text-to-text-transfer-transformer/blob/2ce1574a0c2f5ed65a08e87cc38ad8ceb222b239/t5/data/preprocessors.py#L2895

    Args:
        sequence_length: an int32 scalar (length of the incoming token sequence)
        maximum_length: an int32 scalar (length of the resulting padded or truncated mask).
        noise_density: a float - approximate density of output mask
        mean_noise_span_length: a number
        random_roll: bool, whether to roll the mask by a random integer offset in [0, sequence_length). Set random_roll
            to True to get a more uniform distribution of masked positions. Specifically, when random_roll is False
            (default) and a single span is enough to satisfy the noise density requirement, this fuction masks only the
            last few positions.

    Returns:
        a boolean tensor with shape [sequence_length] denoting the location of masked spans. True denotes a mask on the
        corresponding token while False denotes that the corresponding token is unmasked.
    """
    if noise_density == 0.0:
        return np.zeros(sequence_length, np.bool)

    orig_length = sequence_length

    # increase length to avoid degeneracy
    sequence_length = np.maximum(sequence_length, 2)

    def to_int(x: np.ndarray):
        return x.astype(np.int32)

    def to_float(x: np.ndarray):
        return x.astype(np.float32)

    # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
    num_noise_tokens = to_int(np.round(to_float(sequence_length) * noise_density))
    num_noise_tokens = np.minimum(np.maximum(num_noise_tokens, 1), sequence_length - 1)
    num_noise_spans = to_int(
        np.round(to_float(num_noise_tokens) / mean_noise_span_length))

    # avoid degeneracy by ensuring positive number of noise spans
    num_noise_spans = np.maximum(num_noise_spans, 1)
    num_nonnoise_tokens = sequence_length - num_noise_tokens

    # pick the lengths of the noise spans and the non-noise spans
    def _random_segmentation(
            num_items: int,
            num_segments: int,
    ) -> np.ndarray:
        """Partition a sequence of items randomly into non-empty segments.

        Precondition: in order to ensure determinism, set the numpy seed before calling this function.

        Args:
            num_items: an integer scalar > 0
            num_segments: an integer scalar in [1, num_items]

        Returns:
            A tensor with shape [num_segments] containing positive integers that add up to num_items.
        """
        first_in_segment = to_int(np.less(np.arange(num_items - 1), num_segments - 1))
        np.random.shuffle(first_in_segment)
        first_in_segment = np.pad(first_in_segment, [[1, 0]])
        segment_id = np.cumsum(first_in_segment)
        _, segment_length = np.unique(segment_id, return_counts=True)
        return segment_length

    noise_span_lengths = _random_segmentation(
        num_noise_tokens,
        num_noise_spans,
    )
    nonnoise_span_lengths = _random_segmentation(
        num_nonnoise_tokens,
        num_noise_spans,
    )

    # Identify the indices of the beginning of masked spans.
    interleaved_span_lengths = np.reshape(
        np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1),
        [num_noise_spans * 2])
    span_starts = np.cumsum(interleaved_span_lengths)[:-1]

    span_start_indicator = np.zeros([sequence_length], dtype=np.int32)
    np.put_along_axis(span_start_indicator, span_starts, values=1, axis=0)
    span_num = np.cumsum(span_start_indicator)
    is_noise = np.equal(span_num % 2, 1)
    mask = is_noise[:orig_length]

    if random_roll:
        # TODO: Explore the benefit or using 4 different seeds following the logic below.
        # roll_seed = (seeds[0][0] + seeds[1][1], seeds[0][1] - seeds[1][0])  # new seed.
        # Roll the mask by a random offset e.g. for offset=2: [1,2,3,4] => [3,4,1,2]
        # np.random.seed(roll_seed[0])
        offset = np.random.uniform(low=0, high=sequence_length, size=[1]).astype(np.int32)
        mask = np.roll(mask, shift=offset, axis=0)

    # Pad to a consistent length
    if sequence_length < maximum_length:
        num_values_to_add = maximum_length - sequence_length
        mask = np.concatenate(
            [mask, np.zeros([num_values_to_add], dtype=bool)],
            axis=0,
        )

    return mask


def filter_input_ids_for_t5(
        vocab_size: int,
        input_ids: torch.Tensor,  # An integer tensor of shape [batch_size, input_length]
        sentinel_ids: torch.Tensor,  # An integer tensor of shape [batch_size, input_length]
        token_type_ids: torch.Tensor = None,  # An integer tensor of shape [batch_size, input_length]
) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
    """
    Puts sentinel mask on `input_ids` and fuse consecutive mask tokens into a single mask token by deleting.

    Args:
        vocab_size: An integer representing the size of the vocabulary.
        input_ids: An integer tensor of shape [batch_size, input_length].
        sentinel_ids: An integer tensor of shape [batch_size, input_length], where non-sentinels are 0s, sentinel
            continuations are -1, and sentinel starts are integers within the vocabulary's sentinel token IDs.
        token_type_ids: An integer tensor of shape [batch_size, input_length], where each sentence corresponds to a
            unique integer from 1 to k (where k is the number of sentences). Padding corresponds with type 0, and, if
            present, is at the end of an example.

    Returns:
        A tensor of shape [batch_size, input_length] in which sentinel continuations are removed.
    """
    large_num = vocab_size + 100
    input_ids_full = torch.where(torch.as_tensor(sentinel_ids != 0), sentinel_ids, input_ids)

    # Concatenate -1 to the end of each example's input IDs (i.e., along axis 1, where axis 0 is the batch dimension).
    expanded_input_ids_full = torch.cat(
        (input_ids_full,
         -torch.ones(input_ids_full.shape[0], dtype=input_ids_full.dtype).reshape(-1, 1),
         ),
        dim=1,
    )
    # Create a tensor of indices using broadcasting.
    indices = torch.zeros(expanded_input_ids_full.shape, dtype=torch.int64)
    indices += torch.arange(expanded_input_ids_full.shape[1])

    # Replace -1s within input IDs (span continuations) with a large number that is out of vocabulary.
    indices[expanded_input_ids_full == -1] = large_num

    # Ensure that input IDs that were span continuations appear at the end of the sequence
    gather_indices = torch.sort(indices).values

    # Ensure that all indices of span continuations (now at the end of 'gather_indices') correspond to the last index
    # of a sequence in input IDs (since those values must correspond to -1)
    gather_indices[gather_indices == large_num] = expanded_input_ids_full.shape[1] - 1

    # Shift all -1s within input IDs to the end of the sequence, and then replace them with 0s.
    modified_input_ids = torch.gather(expanded_input_ids_full, 1, gather_indices[:, :-1])
    modified_input_ids = torch.where(torch.eq(modified_input_ids, -1), 0, modified_input_ids)

    # If token type IDs are not provided, then we are done. Otherwise, we need to filter the token type IDs as well.
    if token_type_ids is None:
        return modified_input_ids, None

    expanded_token_type_ids = torch.cat(
        (
            token_type_ids,
            -torch.ones(token_type_ids.shape[0], dtype=token_type_ids.dtype).reshape(-1, 1)
        ),
        dim=1,
    )
    modified_token_type_ids = torch.gather(expanded_token_type_ids, 1, gather_indices[:, :-1])
    modified_token_type_ids = torch.where(torch.eq(modified_token_type_ids, -1), 0, modified_token_type_ids)
    return modified_input_ids, modified_token_type_ids


def filter_target_ids_for_t5(
        input_ids: torch.Tensor,
        input_ids_sentinel: torch.Tensor,
        vocab_size: int,
) -> torch.Tensor:
    """
    Filter the target IDs for T5.

    Args:
        input_ids: An integer tensor of shape [batch_size, input_length].
        input_ids_sentinel: An integer tensor of shape [batch_size, input_length], where non-sentinels are 0s, sentinel
            continuations are -1, and sentinel starts are integers within the vocabulary's sentinel token IDs.
        vocab_size: An integer representing the size of the vocabulary.

    Returns:
        A tensor of shape [batch_size, input_length] in which sentinel continuations are removed.
    """
    # TODO: Support target_length != input_length
    shifted_input_ids = torch.cat(
        [
            torch.zeros((input_ids.shape[0], 1), dtype=torch.int8),
            input_ids[:, :-1],
        ],
        dim=1,
    )
    shifted_input_ids_sentinel = torch.cat(
        [
            torch.zeros((input_ids_sentinel.shape[0], 1), dtype=torch.int8),
            input_ids_sentinel[:, :-1],
        ],
        dim=1,
    )
    result = copy.deepcopy(input_ids_sentinel)
    result = torch.where(
        condition=torch.as_tensor(shifted_input_ids_sentinel != 0),
        input=shifted_input_ids,
        other=result,
    )

    large_num = vocab_size + 100

    # Concatenate 0 to the end of each example's input IDs (i.e., along axis 1, where axis 0 is the batch dimension).
    expanded_result = torch.cat(
        (result,
         torch.zeros(result.shape[0], dtype=result.dtype).reshape(-1, 1),
         ),
        dim=1,
    )
    # Create a tensor of indices using broadcasting.
    indices = torch.zeros(expanded_result.shape, dtype=torch.int64)
    indices += torch.arange(expanded_result.shape[1])

    # Replace 0s within the result with a large number that is out of vocabulary.
    indices[expanded_result == 0] = large_num

    # Ensure that padding tokens within our result are moved to the end.
    gather_indices = torch.sort(indices).values

    # Ensure that all indices of 0's (now at the end of 'gather_indices') correspond to the last index
    # of a sequence in input IDs (since those values must correspond to 0)
    gather_indices[gather_indices == large_num] = expanded_result.shape[1] - 1

    # Shift all -1s within input IDs to the end of the sequence, and then replace them with 0s.
    modified_result = torch.gather(expanded_result, 1, gather_indices[:, :-1])
    return modified_result


def create_sentinel_ids_for_t5(
        mask_indices: torch.Tensor,  # An integer tensor of shape [batch_size, input_length]
        vocab_size: int,
) -> torch.Tensor:
    """
    Create sentinel ids given the indices that should be masked.

    The start indices of each mask are replaced by the sentinel ids in increasing
    order. Consecutive mask indices to be deleted are replaced with `-1`.

    Args:
        mask_indices: The tensor containing mask indices.
        vocab_size: The size of the vocabulary that is used by both the model and tokenizer.

    Returns:
        A tensor of the same size as 'mask_indices', where the beginning of masked spans are denoted by the
        ID of some sentinel token, continuations of spans are denoted with an ID of -1, and non-masked tokens are
        denoted with an ID of 0.
    """
    # Create a tensor of the same size as `mask_indices` where the first element of each mask is replaced with a '1',
    # and the rest of the elements of each mask are replaced with a '0'.
    start_indices = mask_indices - torch.roll(
        input=mask_indices,
        shifts=1,
        dims=-1,
    ) * mask_indices
    start_indices[:, 0] = mask_indices[:, 0]

    # Create a tensor of the same size as `mask_indices` where the first element of each mask is replaced with a '1',
    # the remaining elements of each mask are replaced with a '-1', and non-masked tokens are replaced with a '0'.
    sentinel_ids = torch.where(
        start_indices != 0,
        torch.cumsum(start_indices, dim=-1),
        start_indices,
    )

    # Replace the tokens at the beginning of masked spans with the sentinel ids in decreasing order.
    sentinel_ids = torch.where(
        torch.as_tensor(sentinel_ids != 0),
        (vocab_size - sentinel_ids),
        0,
    )
    sentinel_ids -= mask_indices - start_indices
    return sentinel_ids


def corrupt_for_vanilla_t5(
        examples: Union[BatchEncoding, List[Dict[str, np.ndarray]]],
        vocab_size: int,
        input_length: int,
        target_length: int,
        pad_token_id: int,
        eos_token_id: int,
        decoder_start_token_id: int,
        noise_density: float = 0.5,
        pmi: bool = False,
        ngram_vocab_set: Set[str] = None,
        tokenizer=None,
) -> BatchEncoding:
    """Apply corruption to the input examples for T5, create targets, prepare all model inputs.

    Args:
        examples: A list of dictionaries containing the input and target sequences.
        vocab_size: The size of the vocabulary that is used by both the model and tokenizer.
        input_length: The length of the input sequence.
        target_length: The length of the target sequence.
        pad_token_id: The ID of the padding token.
        eos_token_id: The ID of the end of sentence token.
        decoder_start_token_id: The ID of the decoder start token.
        noise_density: The density of the noise to be applied to the input sequence.
        pmi: Whether to use PMI-based corruption.
        ngram_vocab_set: The set of ngrams to use for PMI-based corruption.
        tokenizer: The tokenizer to use for PMI-based corruption.
    Returns:
        A dictionary containing the input and target sequences, as well as the model inputs.
    """
    # convert list to dict and tensorize input
    if isinstance(examples, list) and isinstance(examples[0], dict):
        batch = BatchEncoding(
            {k: np.array([examples[i][k] for i in range(len(examples))]) for k, v in examples[0].items()}
        )
    else:
        batch = examples

    # convert attention mask to torch tensor
    # TODO: Fix the logic below given that the attention mask consists of a list of tensors of rank 2,
    #  and not a single tensor of rank 3.
    batch['attention_mask'] = torch.asarray(batch['attention_mask'])
    input_ids = torch.asarray(batch["input_ids"])
    batch_size, expandend_input_length = input_ids.shape
    if pmi:
        mask_indices = pmi_noise_mask(examples, ngram_vocab_set, tokenizer)
    else:
        mask_indices = torch.stack(
            [
                torch.asarray(
                    random_spans_noise_mask(
                        sequence_length=expandend_input_length,
                        maximum_length=expandend_input_length,
                        noise_density=noise_density,
                    )
                ) for _ in range(batch_size)
            ],
        )

    # Ensure that padding tokens are not masked
    is_special_token = torch.isin(
        elements=input_ids,
        test_elements=torch.tensor([pad_token_id, eos_token_id]),
    )
    mask_indices = torch.masked_fill(mask_indices, is_special_token, False)
    input_ids_sentinel = create_sentinel_ids_for_t5(
        vocab_size=vocab_size,
        mask_indices=mask_indices.to(torch.int8),
    )

    batch["input_ids"] = filter_input_ids_for_t5(
        input_ids=input_ids,
        sentinel_ids=input_ids_sentinel,
        vocab_size=vocab_size,
    )[0]
    batch["input_ids"] = torch.functional.F.pad(
        input=batch["input_ids"],
        pad=(0, input_length - expandend_input_length),
        mode="constant",
        value=pad_token_id,
    )
    labels = filter_target_ids_for_t5(
        input_ids=input_ids,
        input_ids_sentinel=input_ids_sentinel,
        vocab_size=vocab_size,
    )
    labels = torch.functional.F.pad(
        input=labels,
        pad=(0, target_length - expandend_input_length),
        mode="constant",
        value=pad_token_id,
    )
    labels[labels == pad_token_id] = -100
    batch["labels"] = labels
    if batch["input_ids"].shape[-1] != input_length:
        raise ValueError(
            f"`input_ids` are incorrectly preprocessed. `input_ids` length is {batch['input_ids'].shape[-1]}, but"
            f" should be {input_length}."
        )
    if batch["labels"].shape[-1] != target_length:
        raise ValueError(
            f"`labels` are incorrectly preprocessed. `labels` length is {batch['labels'].shape[-1]}, but should be"
            f" {target_length}."
        )
    # to check that tokens are correctly preprocessed, one can run `self.tokenizer.batch_decode(input_ids)` and
    # `self.tokenizer.batch_decode(labels)` here
    batch["decoder_input_ids"] = torch.asarray(
        shift_tokens_right(
            batch["labels"], pad_token_id, decoder_start_token_id
        )
    )
    return batch
