import typing
import unittest
import numpy as np
import torch
import transformers.models.t5
import corruption_t5 as corruption_lib
from absl.testing import parameterized
import random

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

EXAMPLES = 'examples'
PMI_VOCAB = 'pmi_vocab'
TOKENIZER = 'tokenizer'
INPUT_TOKENS = 'input_tokens'
MAX_PREDICTIONS = 'max_predictions'
MLM_PROBABILITY = 'mlm_probability'
TESTCASE_NAME = 'testcase_name'

PMI_DEMO_VOCAB = {'1950 and 1951',
                  'muttering to himself',
                  'in an interview',
                  'looked back at',
                  'united states',
                  'biological and geological',
                  'sergeant at arms',
                  'greenland and iceland',
                  'plan of action',
                  'wrote several books',
                  'propaganda minister joseph',
                  "none of your damn",
                  "first woman to win",
                  "commanded by a lieutenant",
                  "tells the true story",
                  "everything that is happening",
                  "i have to tell",
                  "from 1987 to 1990",
                  "hard as a rock",
                  "journal has a 2015",
                  "job as a waitress",
                  "turn into the straight",
                  "sat at the bar",
                  "london brighton and south",
                  "ask me a question",
                  "comes to his senses",
                  "mother of two children",
                  "or by any information",
                  "school district officials did",
                  "some explaining to do",
                  "pi beta phi",
                  "jew and gentile",
                  "central business district",
                  "butter and jelly",
                  "pupil total expenditures",
                  "stained glass windows"
                  }

DEMO_TEXTS = ['The united states is a country in North America. looking back at 1950 and 1951,'
              ' the president said in an interview that he was muttering to himself.',
              'I am a sergeant at arms. I wrote several books. I am the propaganda minister joseph.',
              'My plan of action is to go to greenland and iceland. biological and geological.',
              "None of your damn business, but I have to tell you about the hard-as-a-rock butter and jelly sandwich I had for lunch.",
              "The first woman to win the prestigious award sat at the bar, sipping her drink, surrounded by stained glass windows.",
              "Commanded by a lieutenant, the military unit turned into the straight path, ready for the mission ahead.",
              "London, Brighton, and South—locations explored by the journalist in the 2015 journal, tell the true story of diverse experiences.",
              "As a waitress, I have to tell you about the time a customer at the bar asked me a question that left me puzzled.",
              "From 1987 to 1990, the school district officials did some explaining to do regarding pupil total expenditures.",
              "Journal has a 2015 entry about the central business district, where I worked a job as a waitress.",
              "Pi Beta Phi hosted an event, and everyone, from the mother of two children to the jew and gentile attendees, enjoyed it.",
              "Everything that is happening around the world makes me wonder if people will ever come to their senses.",
              "The stained glass windows in the church depicted the turn-of-the-century scenes, including the London, Brighton, and South railway.",
              "A hard-as-a-rock butter and jelly sandwich was my go-to snack during the years from 1987 to 1990.",
              "The waitress at the bar, a mother of two children, juggled her job and school district responsibilities.",
              "None of your damn excuses could justify the actions of the lieutenant who commanded the ill-fated mission.",
              "The true story of the central business district development unfolds in the 2015 journal entry.",
              "From 1987 to 1990, I attended school district events and occasionally worked as a waitress during weekends.",
              "The first woman to win the championship sat at the bar, surrounded by stained glass windows depicting her achievements.",
              "The jew and gentile communities collaborated on a project that transformed the central business district.",
              "I have to tell you about the hard-as-a-rock bread I bought at the local bakery, where I also worked a job as a waitress.",
              "Everything that is happening in the world requires individuals to come to their senses and take action.",
              "Pi Beta Phi hosted an event, and the mother of two children volunteered to help with the preparations.",
              "London, Brighton, and South were the settings for the true story I read in the 2015 journal about a waitress's journey.",
              "Ask me a question about my experiences from 1987 to 1990, and I'll gladly share the highlights of school district life.",
              "None of your damn complaints will change the fact that the lieutenant commanded the military unit with precision.",
              "The job as a waitress allowed me to meet people from different backgrounds, including jew and gentile customers.",
              "Turning into the straight path, the military unit commanded by a lieutenant embarked on a challenging mission.",
              "The stained glass windows in the church told the true story of the London, Brighton, and South railway's history.",
              "Pupil total expenditures in the school district increased during the years from 1987 to 1990.",
              "I have to tell you about the delicious butter and jelly combination that I enjoyed at the bar last night.",
              "Everything that is happening in the central business district reflects the economic changes of the past decade.",
              "Pi Beta Phi organized an event, and the first woman to win a prestigious award was a guest of honor.",
              "A mother of two children, working a job as a waitress, shared her experiences in the 2015 journal.",
              "The hard-as-a-rock bread I bought from the bakery turned into the straight talk of the town.",
              "None of your damn opinions can change the fact that the school district officials did some explaining to do.",
              "London, Brighton, and South—locations explored by the journalist in the 2015 journal entry—inspired my travel plans.",
              "Commanded by a lieutenant, the military unit's journey turned into the straight path of historical significance.",
              "The jew and gentile communities collaborated on a project that transformed the central business district landscape.",
              "I have to tell you about the hard-as-a-rock sandwich I made for lunch, inspired by my job as a waitress.",
              "Everything that is happening in the world makes the stained glass windows of our experiences more colorful.",
              "Pupil total expenditures in the school district increased during the years from 1987 to 1990, impacting education.",
              "Ask me a question about the first woman to win the award, and I'll gladly share the inspiring story.",
              "None of your damn excuses can justify the actions of the military unit commanded by a reckless lieutenant.",
              "The job as a waitress allowed me to connect with people from diverse backgrounds, including jew and gentile customers.",
              "Turning into the straight path, the military unit commanded by a lieutenant faced unexpected challenges.",
              "The true story of the central business district's growth unfolds in the 2015 journal entry.",
              "Pi Beta Phi hosted an event, and the mother of two children actively participated in organizing the activities.",
              "I have to tell you about the delicious butter and jelly combination that I enjoyed at the bar with stained glass windows.",
              "Everything that is happening in the central business district reflects the economic changes from 1987 to 1990.",
              "London, Brighton, and South were the settings for the true story I read in the 2015 journal, where I worked a job as a waitress.",
              "Ask me a question about the hard-as-a-rock experiences during my school district years, and I'll share the lessons learned.",
              "None of your damn complaints will change the fact that the lieutenant commanded the military unit with precision and expertise."]


class CorruptionTest(parameterized.TestCase):

    def setUp(self):
        self.tokenizer = transformers.T5Tokenizer.from_pretrained('google/t5-v1_1-small')
        random.seed(42)

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

    @parameterized.named_parameters(
        {
            TESTCASE_NAME: "No n-grams in PMI Vocab",
            INPUT_TOKENS: "Ofek went to Taub.",
            MAX_PREDICTIONS: 512,
            MLM_PROBABILITY: 0.5,
        },
        {
            TESTCASE_NAME: "Gibberish",
            INPUT_TOKENS: "asdvbdsasd asdvewasdf ",
            MAX_PREDICTIONS: 512,
            MLM_PROBABILITY: 0.5,
        },
        {
            TESTCASE_NAME: "Some n-grams in PMI Vocab",
            INPUT_TOKENS: "I have to tell everything that is happening, but what happens after that, i don't know",
            MAX_PREDICTIONS: 512,
            MLM_PROBABILITY: 0.5,
        },
        {
            TESTCASE_NAME: "extra punctuation",
            INPUT_TOKENS: "I want to tell you, maybe ask you? maybe yell! maybe scream & yell - then tell you.",
            MAX_PREDICTIONS: 512,
            MLM_PROBABILITY: 0.5,
        },

    )
    def test_pmi_mask_word(self,
                           input_tokens,
                           max_predictions,
                           mlm_probability,
                           ):
        """
        Test different use cases of the method "pmi_word_mask"
        :param input_tokens: input tokens to test
        :param max_predictions: max predictions to test
        :param mlm_probability: mlm probability to test
        :return: None
        """
        input_tokens = self.tokenizer(
            input_tokens,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
        )['input_ids']
        input_tokens = input_tokens.squeeze()
        ref_tokens = []
        for input_id in input_tokens:
            token = self.tokenizer._convert_id_to_token(input_id.item())
            ref_tokens.append(token)
        mask_labels_for_sample = corruption_lib.pmi_word_mask(
            ref_tokens,
            PMI_DEMO_VOCAB,
            max_predictions,
            mlm_probability, )
        self.assertIsNotNone(mask_labels_for_sample)

    @parameterized.named_parameters(
        {
            TESTCASE_NAME: f"test stand use of pmi_noise_mask",
            EXAMPLES: DEMO_TEXTS,
            PMI_VOCAB: PMI_DEMO_VOCAB,
        },
    )
    def test_pmi_noise_mask(self,
                            examples,
                            pmi_vocab,
                            ):
        """
        Test the method "pmi_noise_mask". This method will test the standard use case of the method as expected to
        happen in pre-training.
        :param examples: examples to test
        :param pmi_vocab: pmi vocab to test
        """
        tokenized_examples = self.tokenizer(
            examples,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            padding=True,
        )
        predicted_mask_labels = corruption_lib.pmi_noise_mask(
            tokenized_examples,
            pmi_vocab,
            self.tokenizer,
        )
        self.assertIsNotNone(predicted_mask_labels)


if __name__ == '__main__':
    unittest.main()
