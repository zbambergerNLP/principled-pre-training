import corruption_t5 as corruption_lib
from absl.testing import parameterized
import transformers
import random

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


class PMICorruptionTest(parameterized.TestCase):

    def setUp(self):
        self.tokenizer = transformers.T5Tokenizer.from_pretrained('google/t5-v1_1-small')
        random.seed(42)

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
            TESTCASE_NAME: f"Test PMI Noise Mask, many sentences and a large n-gram vocab",
            EXAMPLES: DEMO_TEXTS,
            PMI_VOCAB: PMI_DEMO_VOCAB,
        },
    )
    def test_pmi_noise_mask(self,
                            examples,
                            pmi_vocab,
                            ):
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
