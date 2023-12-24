import typing

import torch
import transformers
import omegaconf
import datasets

# Local imports
from utils import (
    logging_utils,
    data_utils,
    t5_utils,
)

from constants import (
    base_constants
)


# TODO: Support other models with alternative signatures (e.g., creating T5 models based on the Megatron LM framework
#  as opposed to the HuggingFace framework). Return type will be a union (typing.Union) of the various model types.
def get_model(
        args: omegaconf.DictConfig,
        config: transformers.AutoConfig,
        logger: logging_utils.Logger,
) -> torch.nn.Module:
    """
    Either create or load a T5 model for conditional generation.

    The T5 model we use can be either a HuggingFace T5 model or a locally implemented T5 model.
    Furthermore, we support loading a model from a checkpoint, randomly initializing a model, or loading a model from
    a pretrained checkpoint (e.g., the standard T5-base weights on Huggingface).

    We also save the number of parameters in the model to the args.

    :param args: The omegaconf configuration used to generate the model.
    :param config: The model configuration. See `get_config` for more details.
    :param logger: The logger. See `logging_utils.py` for more details.
    :return: A T5 model for conditional generation.
    """

    logger.log_message('Loading model')
    model_implementation: torch.nn.Module = {
        base_constants.T5ModelConstants.HF_T5: transformers.T5ForConditionalGeneration,  # HuggingFace T5
        base_constants.T5ModelConstants.LOCAL_T5: t5_utils.MyT5,  # TODO: Consider using Megatron LM for this.
    }[args.model.model_implementation]

    # Load the model from a defined checkpoint
    if args.model.checkpoint_path:
        logger.log_message(f'Loading model from checkpoint: {args.model.checkpoint_path}')
        model = model_implementation(config)
        model.load_state_dict(torch.load(args.model.checkpoint_path))

    # Randomly initialize the model
    elif args.model.random_init:
        logger.log_message('Randomly initializing model')
        model = model_implementation(config)

    # Load the model from a pretrained checkpoint (e.g., the standard T5-base weights on Huggingface)
    else:
        assert (
            model_implementation == transformers.T5ForConditionalGeneration,
            'To load HFs weights you need to use HF model'
        )
        logger.log_message(f'Loading model from pretrained: {args.model.name}')
        model = model_implementation.from_pretrained(
            args.model.name,
            config=config,
        )

    # Save the number of parameters in the model to the args
    with omegaconf.open_dict(args):
        args.n_all_param = sum([parameter.nelement() for parameter in model.parameters()])
        logger.log_message(f'Number of parameters: {args.n_all_param.__format__("0,")}')

    return model


def get_config(
        args: omegaconf.DictConfig,
        logger: logging_utils.Logger,
) -> transformers.AutoConfig:
    """
    Get the model configuration, which is used to initialize the model.

    :param args: The omegaconf configuration used to generate the model's configuration.
    :param logger: The logger. See `logging_utils.py` for more details.
    :return: The model configuration.
    """
    logger.log_message('Loading model config')

    config = transformers.AutoConfig.from_pretrained(
        args.model.name,
    )

    # TODO: Review the following code. We may want to customize the hydra and omegaconf code to make this cleaner.
    #  Furthermore, we want to support more than just a T5 architecture (e.g., support DEPTH and UL2 in additional to
    #  the basic T5 architecture).
    if hasattr(args.model, 'overwrite'):
        for k, v in args.model.overwrite.items():
            assert hasattr(config, k), f'config does not have attribute {k}'
            setattr(config, k, v)

    if hasattr(args.model, 'add_config'):
        for k, v in args.model.add_config.items():
            assert not hasattr(config, k), f'config already has attribute {k}'
            setattr(config, k, v)

    return config
