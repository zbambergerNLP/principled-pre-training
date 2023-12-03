import typing
import unittest
import numpy as np
import torch
import transformers.models.t5
import corruption_t5 as corruption_lib
from absl.testing import parameterized


# Fields for parameterized tests
TESTCASE_NAME = 'testcase_name'
SENTENCES = 'sentences'
INPUT_LENGTH = 'input_length'
NOISE_DENSITY = 'noise_density'
MEAN_NOISE_SPAN_LENGTH = 'mean_noise_span_length'
SEEDS = 'seeds'
EXPECTED_SPAN_MASKS = 'expected_span_masks'
EXPECTED_INPUT_IDS_SENTINEL = 'expected_input_ids_sentinel'
EXPECTED_LABEL_IDS_SENTINEL = 'expected_label_ids_sentinel'

EXPECTED_MODIFIED_INPUT_IDS = 'expected_modified_input_ids'
EXPECTED_MODIFIED_LABEL_IDS = 'expected_modified_label_ids'
EXPECTED_TOKEN_TYPE_IDS = 'expected_token_type_ids'
EXPECTED_LABEL_TOKEN_TYPE_IDS = 'expected_label_token_type_ids'

# Add special tokens for the test
TOKEN_TYPE_IDS = 'token_type_ids'
PADDING_TOKEN_IDS = 'padding_token_id'
EXPECTED_SHUFFLED_SENTENCE_ORDER = 'expected_shuffled_sentence_order'
EXPECTED_SHUFFLED_SENTENCE_LENGTHS = 'expected_shuffled_sentence_lengths'
EXPECTED_SHUFFLED_SENTENCE_START_INDICES = 'expected_shuffled_sentence_start_indices'

# Test inputs
EXAMPLE_1 = 'Hello world! I am learning to use tokenizers. Did you know they are this cool?'
EXAMPLE_2 = 'His lecture was so boring... I couldn\'t help but doze off.'
EXAMPLE_3 = 'Here is a first sentence! This is a second. What about a third? Four is enough!'


class CorruptionTest(parameterized.TestCase):
    @parameterized.named_parameters(
        {TESTCASE_NAME: 'short_sequence_length',
         INPUT_LENGTH: 5,
         NOISE_DENSITY: 0.5,
         MEAN_NOISE_SPAN_LENGTH: 2,
         SEEDS: [[42, 43], [44, 45]],
         EXPECTED_SPAN_MASKS: [False, True, True, False, False]
         },
        {TESTCASE_NAME: 'medium_sequence_length',
         INPUT_LENGTH: 10,
         NOISE_DENSITY: 0.5,
         MEAN_NOISE_SPAN_LENGTH: 3,
         SEEDS: [[42, 43], [44, 45]],
         EXPECTED_SPAN_MASKS: [True, False, False, False, True, True, True, False, False, True],
         },
        {TESTCASE_NAME: 'medium_sequence_length_different_seed',
         INPUT_LENGTH: 10,
         NOISE_DENSITY: 0.5,
         MEAN_NOISE_SPAN_LENGTH: 3,
         SEEDS: [[100, 101], [102, 103]],
         EXPECTED_SPAN_MASKS: [False, False, False, True, True, True, True, False, False, True],
         },
        {TESTCASE_NAME: 'medium_sequence_length_lower_density',
         INPUT_LENGTH: 10,
         NOISE_DENSITY: 0.3,
         MEAN_NOISE_SPAN_LENGTH: 3,
         SEEDS: [[42, 43], [44, 45]],
         EXPECTED_SPAN_MASKS: [False, False, True, True, True, False, False, False, False, False],
         },
        {TESTCASE_NAME: 'medium_sequence_length_higher_density',
         INPUT_LENGTH: 10,
         NOISE_DENSITY: 0.9,
         MEAN_NOISE_SPAN_LENGTH: 9,
         SEEDS: [[42, 43], [44, 45]],
         EXPECTED_SPAN_MASKS: [True, False, True, True, True, True, True, True, True, True],
         },
    )
    def test_random_spans_noise_mask(
            self,
            input_length: int,
            noise_density: float,
            mean_noise_span_length: int,
            seeds: typing.List[typing.List[int]],
            expected_span_masks: typing.List[typing.List[bool]]):
        np.random.seed(seeds[0][0])
        span_masks = corruption_lib.random_spans_noise_mask(
            sequence_length=input_length,
            maximum_length=input_length,
            noise_density=noise_density,
            mean_noise_span_length=mean_noise_span_length,
        )
        np.testing.assert_array_equal(span_masks, expected_span_masks)

    @parameterized.named_parameters(
        {TESTCASE_NAME: 'short_sequence_length',
         INPUT_LENGTH: 5,
         NOISE_DENSITY: 0.5,
         MEAN_NOISE_SPAN_LENGTH: 2,
         SEEDS: [[42, 43], [44, 45]],
         EXPECTED_INPUT_IDS_SENTINEL: [[0, 32099, -1, 0, 0]],
         },
        {TESTCASE_NAME: 'medium_sequence_length',
         INPUT_LENGTH: 10,
         NOISE_DENSITY: 0.5,
         MEAN_NOISE_SPAN_LENGTH: 3,
         SEEDS: [[42, 43], [44, 45]],
         EXPECTED_INPUT_IDS_SENTINEL: [[32099, 0, 0, 0, 32098, -1, -1, 0, 0, 32097]],
         },
        {TESTCASE_NAME: 'medium_sequence_length_different_seed',
         INPUT_LENGTH: 10,
         NOISE_DENSITY: 0.5,
         MEAN_NOISE_SPAN_LENGTH: 3,
         SEEDS: [[100, 101], [102, 103]],
         EXPECTED_INPUT_IDS_SENTINEL: [[0, 0, 0, 32099, -1, -1, -1, 0, 0, 32098]],
         },
        {TESTCASE_NAME: 'medium_sequence_length_lower_density',
         INPUT_LENGTH: 10,
         NOISE_DENSITY: 0.3,
         MEAN_NOISE_SPAN_LENGTH: 3,
         SEEDS: [[42, 43], [44, 45]],
         EXPECTED_INPUT_IDS_SENTINEL: [[0, 0, 32099, -1, -1, 0, 0, 0, 0, 0]],
         },
        {TESTCASE_NAME: 'medium_sequence_length_higher_density',
         INPUT_LENGTH: 10,
         NOISE_DENSITY: 0.9,
         MEAN_NOISE_SPAN_LENGTH: 9,
         SEEDS: [[42, 43], [44, 45]],
         EXPECTED_INPUT_IDS_SENTINEL: [[32099, 0, 32098, -1, -1, -1, -1, -1, -1, -1]],
         },
    )
    def test_create_sentinel_ids(
            self,
            input_length: int,
            noise_density: float,
            mean_noise_span_length: int,
            seeds: typing.List[typing.List[int]],
            expected_input_ids_sentinel: typing.List[typing.List[int]],
    ):
        np.random.seed(seeds[0][0])
        tokenizer = transformers.T5Tokenizer.from_pretrained('t5-small')
        span_masks = corruption_lib.random_spans_noise_mask(
            sequence_length=input_length,
            noise_density=noise_density,
            mean_noise_span_length=mean_noise_span_length,
            maximum_length=input_length,

        )
        # print(f'span masks are: {span_masks}')
        input_ids_sentinel = corruption_lib.create_sentinel_ids_for_t5(
            torch.as_tensor([span_masks], dtype=torch.int8),
            vocab_size=len(tokenizer),
        )
        np.testing.assert_array_equal(input_ids_sentinel, expected_input_ids_sentinel)

    @parameterized.named_parameters(
        {
            'testcase_name': 'basic_test',
            'examples': [
                'Hello world!',
                'Here is an example with a longer sentence',
                'An example with multiple sentences? Might be tricky... Worth testing!',
            ],
            'input_length': 20,
            'target_length': 10,
            EXPECTED_MODIFIED_INPUT_IDS: np.array(
                [[32099, 296, 55, 1,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [947, 32099, 46, 677, 28, 3, 9, 1200, 32098, 1,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [32099, 28, 1317, 16513, 32098, 16114, 233, 16990, 2505, 32097, 1,
                  0, 0, 0, 0, 0, 0, 0, 0, 0]],
            ),
            # Recall that in the target of T5, padding tokens (usually 0) are replaced with -100.
            EXPECTED_MODIFIED_LABEL_IDS: np.array(
                [[32099, 8774, -100, -100, -100, -100, -100, -100, -100, -100],
                 [32099, 19, 32098, 7142, -100, -100, -100, -100, -100, -100],
                 [32099, 389, 677, 32098, 58, 23840, 36, 32097, 55, -100]],
            ),
        },
        {
            'testcase_name': 'truncated targets',
            'examples': [EXAMPLE_1, EXAMPLE_2, EXAMPLE_3],
            'input_length': 20,
            'target_length': 10,
            EXPECTED_MODIFIED_INPUT_IDS: np.array(
                [[32099, 296, 32098, 183, 32097, 169, 14145, 8585, 7, 5, 3963, 25, 214, 32096, 1,
                  0, 0, 0, 0, 0],
                 [32099, 47, 32098, 27, 2654, 31, 17, 199, 32097, 776, 326, 5, 1,
                  0, 0, 0, 0, 0, 0, 0],
                 [947, 19, 32099, 100, 19, 3, 9, 511, 5, 32098, 81, 32097, 1,
                  0, 0, 0, 0, 0, 0, 0]]
            ),
            EXPECTED_MODIFIED_LABEL_IDS: np.array(

                [
                    # Truncated 33, 48, 1633, which are masked by sentinel 32096
                    [32099, 8774, 32098, 55, 27, 32097, 1036, 12, 32096, 79],
                    # No truncation
                    [32099, 978, 7177, 32098, 78, 13006, 233, 32097, 68, 103],
                    # Truncated 9, 1025, 58, which are masked by sentinel 32097
                    [32099, 3, 9, 166, 7142, 55, 32098, 363, 32097, 3],
                ],
            ),
        },
    )
    def test_corrupt_for_vanilla_t5(
            self,
            examples,
            input_length,
            target_length,
            expected_modified_input_ids,
            expected_modified_label_ids,
            seed=42,
    ):
        """

        :param examples: A list (batch) of strings corresponding to the examples to be corrupted.
        :param input_length: The length of the input sequence.
        :param target_length: The length of the target sequence.
        :param expected_modified_input_ids: A tensor of shape [batch_size, input_length] corresponding to the expected
            modified input ids.
        :param expected_modified_label_ids: A tensor of shape [batch_size, target_length] corresponding to the expected
            modified label ids.
        :param seed: The seed to use for the test.
        """
        # Set seed
        np.random.seed(seed)
        torch.random.manual_seed(seed)

        tokenizer = transformers.T5TokenizerFast.from_pretrained('t5-small')
        tokenized_examples = tokenizer(
            examples,
            max_length=input_length,
            truncation='only_first',
            padding='longest',
            return_tensors='np'
        )
        batch_encoding = corruption_lib.corrupt_for_vanilla_t5(
            examples=tokenized_examples,
            vocab_size=len(tokenizer),
            input_length=input_length,
            target_length=target_length,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            decoder_start_token_id=tokenizer.pad_token_id,
        )
        torch.testing.assert_allclose(
            actual=batch_encoding['input_ids'],
            expected=expected_modified_input_ids,
            msg=f'Excepted: {expected_modified_input_ids}\nActual: {batch_encoding["input_ids"]}',
        )
        torch.testing.assert_allclose(
            actual=batch_encoding['labels'],
            expected=expected_modified_label_ids,
            msg=f'Excepted: {expected_modified_label_ids}\nActual: {batch_encoding["labels"]}',
        )


if __name__ == '__main__':
    unittest.main()
