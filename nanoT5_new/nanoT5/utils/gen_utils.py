import torch
import os

from accelerate.utils import set_seed
import omegaconf
from hydra.utils import to_absolute_path
import accelerate

# Local imports
from .constants import (
    TrainingPhase,
    Device,
    NumericalPrecision,
    ModelImplementation,
    EnvironmentVariable,
)
from .logging_utils import Logger


def check_args_and_env(args: omegaconf.DictConfig):
    """
    Check if the arguments and environment variables are valid.

    :param args: The hydra config which contains the model, data, and training arguments. See the
        configs/default.yaml file for the default configuration.
    """
    assert args.optim.batch_size % args.optim.grad_acc == 0, \
        'Batch size must be divisible by grad_acc\n' \
        f'Batch size: {args.optim.batch_size}, grad_acc: {args.optim.grad_acc}'

    # Train log must happen before eval log
    assert args.evaluate.every_steps % args.logging.every_steps == 0

    if args.device == Device.GPU.value:
        assert (
            torch.cuda.is_available(),
            'You selected to use a GPU, but CUDA is not available on your machine.'
        )

    assert not (args.eval_only and args.predict_only), \
        'Cannot both only evaluate and only predict.'

    if args.predict_only:
        assert args.mode == TrainingPhase.FT.value, \
            'Predict only works in fine-tuning mode, but the current mode is pre-training (pt)'


def opti_flags(args: omegaconf.DictConfig):
    """
    Enable more effective cuda operations, and utilize bf16 precision if appropriate.

    :param args: The hydra config which contains the model, data, and training arguments. See the
        configs/default.yaml file for the default configuration.
    """
    # This lines reduce training step by 2.4x
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if (
            args.precision == NumericalPrecision.BF16 and
            args.device == Device.GPU.value and
            args.model.klass == ModelImplementation.LOCAL_T5.value
    ):
        args.model.add_config.is_bf16 = True


def update_args_with_env_info(args: omegaconf.DictConfig):
    """
    Update the arguments with environment variables.

    :param args: The hydra config which contains the model, data, and training arguments. See the
        configs/default.yaml file for the default configuration.
    """
    with omegaconf.open_dict(args):
        slurm_id = os.getenv(EnvironmentVariable.SLURM_JOB_ID.value)

        if slurm_id is not None:
            args.slurm_id = slurm_id
        else:
            args.slurm_id = 'none'

        args.working_dir = os.getcwd()


def update_paths(args: omegaconf.DictConfig):
    """
    Update the paths in the arguments to absolute paths.

    Specifically, update the paths of the execution file, data directory, and task directory.

    :param args: The hydra config which contains the model, data, and training arguments. See the
        configs/default.yaml file for the default configuration.
    """
    # If we are fine-tuning, we need to update the paths
    if args.mode == TrainingPhase.FT.value:
        args.data.exec_file_path = to_absolute_path(args.data.exec_file_path)
        args.data.data_dir = to_absolute_path(args.data.data_dir)
        args.data.task_dir = to_absolute_path(args.data.task_dir)


def setup_basics(accelerator: accelerate.Accelerator, args: omegaconf.DictConfig) -> Logger:
    """
    Setup the logger and accelerator.
    :param accelerator: The accelerator object which will be used to train the model.
    :param args: The hydra config which contains the model, data, and training arguments. See the
        configs/default.yaml file for the default configuration.
    :return: The logger object which will be used to log the training and evaluation results.
    """
    check_args_and_env(args)
    update_args_with_env_info(args)
    update_paths(args)
    opti_flags(args)

    if args.seed is not None:
        set_seed(args.seed)

    logger = Logger(args=args, accelerator=accelerator)

    return logger
