import typing

import torch
import datasets
from torch.utils.data import DataLoader
from omegaconf import open_dict
from datasets.iterable_dataset import IterableDataset
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    AutoConfig,
)
import omegaconf
import transformers

from .copied_utils import (
    compute_input_and_target_lengths,
    DataCollatorForT5MLM,
    tokenize_function,
    DataCollatorForNI,
)
from .t5_model import MyT5
from .constants import (
    ModelImplementation,
    TrainingPhase,
    DatasetSplit,
    T5TokenizerConstants,
)
from .logging_utils import Logger

def get_model(
        args: omegaconf.DictConfig,
        config: transformers.AutoConfig,
        logger: Logger,
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
    # TODO: Review the following code. We may want to customize the hydra and omegaconf code to make this cleaner.
    #  Furthermore, we want to support more than just a T5 architecture (e.g., support DEPTH and UL2 in additional to
    #  the basic T5 architecture).
    model_implementation: torch.nn.Module = {
        ModelImplementation.HUGGINGFACE_T5.value: transformers.T5ForConditionalGeneration,  # HuggingFace T5
        ModelImplementation.LOCAL_T5.value: MyT5,  # TODO: Consider using Megatron LM for this.
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
        logger: Logger,
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

    if hasattr(args.model, 'overwrite'):
        for k, v in args.model.overwrite.items():
            assert hasattr(config, k), f'config does not have attribute {k}'
            setattr(config, k, v)

    if hasattr(args.model, 'add_config'):
        for k, v in args.model.add_config.items():
            assert not hasattr(config, k), f'config already has attribute {k}'
            setattr(config, k, v)

    return config

def get_tokenizer(
        args: omegaconf.DictConfig,
        logger: Logger,
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


def load_dataset_splits(
        args: omegaconf.DictConfig,
        logger: Logger,
) -> typing.Dict[str, datasets.Dataset]:
    """
    Load the splits of the dataset (e.g., train, test, validation).

    :param args: The omegaconf configuration used to generate the dataset splits.
    :param logger: A logging_utils.Logger object. See `logging_utils.py` for more details.
    :return: A dictionary of the dataset splits.
    """
    logger.log_message(f'Loading dataset {args.dataset.name} from {args.dataset.path}')

    if args.mode == TrainingPhase.PT.value:

        # TODO: Enable loading multiple datasets and interweaving them.
        dataset = datasets.load_dataset(
            path=args.dataset.path,
            name=args.dataset.name,
            streaming=args.dataset.streaming,
        )

        dataset = dataset.remove_columns(
            args.dataset.columns_to_remove
        )

        # We want to use the validation set as the test set
        dataset_splits = {
            DatasetSplit.TRAIN.value: dataset[DatasetSplit.TRAIN.value],
            DatasetSplit.TEST.value: dataset[DatasetSplit.VALIDATION.value],
        }

        assert (
            dataset[DatasetSplit.TRAIN.value].n_shards == args.dataset.num_shards
        ), "We want to have many shards for efficient processing with num_workes in PyTorch dataloader"

    elif args.mode == TrainingPhase.FT.value:
        dataset_splits = datasets.load_dataset(
            args.data.exec_file_path,
            data_dir=args.data.data_dir,
            task_dir=args.data.task_dir,
            max_num_instances_per_task=args.data.max_num_instances_per_task,
            max_num_instances_per_eval_task=args.data.max_num_instances_per_task
        )

    else:
        raise NotImplementedError

    return dataset_splits


def process_dataset(
        dataset_splits: typing.Dict[str, datasets.Dataset],
        args: omegaconf.DictConfig,
        tokenizer: transformers.AutoTokenizer,
        logger: Logger,
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
    if args.mode == TrainingPhase.PT.value:
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

            with open_dict(args):
                args.data.before_mask_input_length = before_mask_input_length
                args.data.target_length = target_length

            # TODO: Let users choose between simple tokenization and tokenization with example concatenation (as it
            #  is done in the original T5 paper).
            dataset_split = dataset_split.map(
                tokenize_function,
                batched=True,
                fn_kwargs={
                    T5TokenizerConstants.TOKENIZER: tokenizer,
                    T5TokenizerConstants.IN_LENGTH: before_mask_input_length,
                },
                remove_columns=[args.dataset.text_column]
            )

            dataset_split = dataset_split.shuffle(buffer_size=args.dataset.buffer_size, seed=args.seed)
            final_datasets[split] = dataset_split
    elif args.mode == TrainingPhase.FT.value:
        final_datasets = dataset_splits
    else:
        raise NotImplementedError

    return final_datasets


def get_data_collator(
        tokenizer: transformers.AutoTokenizer,
        config: transformers.AutoConfig,
        args: omegaconf.DictConfig,
) -> typing.Union[DataCollatorForT5MLM, DataCollatorForNI]:
    """
    Get the data collator. This is used to collate the data into batches.

    :param tokenizer: The tokenizer used to tokenize the inputs and outputs.
    :param config: The model configuration. See `get_config` for more details.
    :param args: The omegaconf configuration used to generate the data collator.
    :param logger: The logger. See `logging_utils.py` for more details.
    :return: The data collator.
    """
    if args.mode == TrainingPhase.PT.value:
        data_collator = DataCollatorForT5MLM(
            tokenizer=tokenizer,
            noise_density=args.data.mlm_probability,
            mean_noise_span_length=args.data.mean_noise_span_length,
            input_length=args.data.input_length,
            target_length=args.data.target_length,
            pad_token_id=config.pad_token_id,
        )
    elif args.mode == TrainingPhase.FT.value:
        data_collator = DataCollatorForNI(
            tokenizer,
            padding="longest",
            max_source_length=args.data.max_seq_len,
            max_target_length=args.data.max_target_len,
            label_pad_token_id=T5TokenizerConstants.PAD_TOKEN_ID,
            pad_to_multiple_of=8,
            add_task_name=args.data.add_task_name,
            add_task_definition=args.data.add_task_definition,
            num_pos_examples=args.data.num_pos_examples,
            num_neg_examples=args.data.num_neg_examples,
            add_explanation=args.data.add_explanation,
            tk_instruct=args.data.tk_instruct
        )
    else:
        raise NotImplementedError

    return data_collator


def get_dataloaders(
        tokenizer: transformers.AutoTokenizer,
        config: transformers.AutoConfig,
        args: omegaconf.DictConfig,
        logger: Logger,
) -> typing.Tuple[datasets.Dataset, datasets.Dataset]:
    """
    Create the dataloaders for the training and test splits.

    :param tokenizer: The tokenizer used to tokenize the inputs and outputs.
    :param config: The model configuration. See `get_config` for more details.
    :param args: The omegaconf configuration used to generate the dataloaders.
    :param logger: The logger. See `logging_utils.py` for more details.
    :return: The dataloaders. The first element is the training dataloader and the second element is the
        test dataloader.
    """
    dataset_splits = load_dataset_splits(args=args, logger=logger)
    dataset = process_dataset(dataset_splits=dataset_splits, args=args, tokenizer=tokenizer, logger=logger)
    data_collator = get_data_collator(tokenizer=tokenizer, config=config, args=args)

    is_iterable = isinstance(dataset[DatasetSplit.TRAIN.value], IterableDataset)
    logger.log_message(f'Is dataset iterable: {is_iterable}')
    dataloaders = {}

    for split in [DatasetSplit.TRAIN.value, DatasetSplit.TEST.value]:


        # TODO: Enable dynamic batch size via HuggingFace/Accelerate
        batch_size = args.optim.batch_size // args.optim.grad_acc

        shuffle = (split == DatasetSplit.TRAIN.value) and not is_iterable
        logger.log_message(f'\tShuffle {split} data: {shuffle}')

        if args.mode == TrainingPhase.FT.value and split == DatasetSplit.TRAIN.value:
            assert shuffle is True
        else:
            assert shuffle is False

        dataloaders[split] = DataLoader(
            dataset[split],
            shuffle=shuffle,
            collate_fn=data_collator,
            batch_size=batch_size,
            num_workers=args.data.num_workers,
            # num_workers=args.num_cpus,
            pin_memory=True,
            drop_last=False,
        )

    # Add & Check args about data loaders
    with open_dict(args):

        logger.log_message(f'Number of epochs: {args.optim.epochs}')
        logger.log_message(f'Number of gradient accumulation steps: {args.optim.grad_acc}')

        if not is_iterable:
            args.data.train_batches = len(dataloaders[DatasetSplit.TRAIN.value])
            args.data.test_batches = len(dataloaders[DatasetSplit.TEST.value])

            logger.log_message(f'Number of train batches: {args.data.train_batches}')
            logger.log_message(f'Number of test batches: {args.data.test_batches}')

            args.optim.total_steps = (
                (args.data.train_batches // args.optim.grad_acc) * args.optim.epochs
            )

        if args.optim.epochs > 0:

            if is_iterable:
                num_examples = args.dataset.training_set.num_examples
                logger.log_message(f"Number of examples: {num_examples}")

                logger.log_message(
                    f"Total steps: {num_examples // args.optim.grad_acc // args.optim.batch_size * args.optim.epochs}\n"
                    f"({num_examples} // {args.optim.grad_acc} // {args.optim.batch_size}) * {args.optim.epochs}"
                )
                args.optim.total_steps = (
                        (num_examples // args.optim.grad_acc // args.optim.batch_size) * args.optim.epochs
                )
            else:
                args.optim.total_steps = (
                        (len(dataloaders[DatasetSplit.TRAIN.value]) // args.optim.grad_acc) * args.optim.epochs
                )

            # Consider the number of GPU workers
            logger.log_message(
                f"Steps per worker: {args.optim.total_steps // args.num_gpus}\n"
                f"{args.optim.total_steps} // {args.num_gpus}"
            )
            args.optim.total_steps = args.optim.total_steps // args.num_gpus

        args.evaluate.corrected_steps = args.evaluate.steps // args.num_gpus

    return dataloaders[DatasetSplit.TRAIN.value], dataloaders[DatasetSplit.TEST.value]
