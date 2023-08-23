import typing

import datasets
import numpy as np
import torch
import transformers
from dataclasses import dataclass

import corruption_t5


@dataclass
class T5DataCollator:
    """
    Data collator used for T5 span-masked language modeling.

    It is made sure that after masking the inputs are of length `data_args.max_seq_length` and targets are also of fixed
    length.

    For more information on how T5 span-masked language modeling works, one can take a look
    """

    def __init__(
            self,
            tokenizer: transformers.PreTrainedTokenizer,
            noise_density: float,
            mean_noise_span_length: float,
            input_length: int,
            target_length: int,
            pad_token_id: int,
            decoder_start_token_id: int,
            seed: int = 42,
    ):
        """Initialize a T5DataCollator instance.

        :param tokenizer: The tokenizer to use as part of span corruption in the data collator.
        :param noise_density: The density of noise to be added to the input sequence.
        :param mean_noise_span_length: The mean length of the noise spans.
        :param input_length: The length of the input sequence.
        :param target_length: The length of the target sequence.
        :param pad_token_id: The id of the pad token.
        :param decoder_start_token_id: The id of the decoder start token.
        :param seed: The seed to use for random number generation.
        """
        np.random.seed(seed)
        self.tokenizer = tokenizer
        self.noise_density = noise_density
        self.mean_noise_span_length = mean_noise_span_length
        self.input_length = input_length
        self.target_length = target_length
        self.pad_token_id = pad_token_id
        self.decoder_start_token_id = decoder_start_token_id

    def __call__(
            self,
            examples: typing.List[typing.Dict[str, torch.Tensor]],
    ) -> transformers.BatchEncoding:
        """Generate a dictionary of input tensors to a Vanilla T5 language model.

        :param examples:
        :return: A dictionary of input tensors to a Vanilla T5 language model.
        """
        input_ids = []
        attention_masks = []
        for example in examples:
            input_ids.append(example['input_ids'])
            attention_masks.append(example['attention_mask'])
        input_ids = np.stack(input_ids)
        attention_masks = np.stack(attention_masks)
        batch = transformers.BatchEncoding(
            data={
                'input_ids': input_ids,
                'attention_mask': attention_masks,
            },
        )
        # print(f'batch input IDs shape: {batch["input_ids"].shape}')
        batch_encoding = corruption_t5.corrupt_for_vanilla_t5(
            batch,
            self.tokenizer.vocab_size,
            self.input_length,
            self.target_length,
            self.pad_token_id,
            self.tokenizer.eos_token_id,
            self.decoder_start_token_id,
            self.noise_density,
        )
        return batch_encoding
