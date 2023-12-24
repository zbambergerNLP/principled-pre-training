import dataclasses
import logging
import os
import typing
from collections import defaultdict

# Flags
from pre_training_flags import DataTrainingArguments, ModelArguments, TrainingArguments

# Training transformers models
import accelerate
import datasets
import transformers

# Hyper-parameter configuration
import omegaconf

# Logging
import neptune



class Averager:
    def __init__(self, weight: float = 1.0 ):
        """
        Initialize an averager.

        :param weight: The weight to apply to the statistics. Default is 1.0.
        """
        self.weight = weight
        self.total = defaultdict(float)
        self.counter = defaultdict(float)

    def reset(self):
        """
        Reset the averager.
        """
        self.total = defaultdict(float)
        self.counter = defaultdict(float)

    def update(self, stats: typing.Dict[str, float]):
        """
        Update the averager with new statistics.
        :param stats: A dictionary of statistics to update the averager with.
        """
        for key, value in stats.items():
            self.total[key] = self.total[key] * self.weight + value * self.weight
            self.counter[key] = self.counter[key] * self.weight + self.weight


    def average(self) -> typing.Dict[str, float]:
        """
        Get the average of the statistics in the averager.

        :return: A dictionary of statistics averaged over the number of updates.
        """
        averaged_stats = {
            key: total / self.counter[key] for key, total in self.total.items()
        }
        self.reset()
        return averaged_stats

class Logger:
    def __init__(
            self,
            accelerator: accelerate.Accelerator,
            args: omegaconf.DictConfig,
            model_args: ModelArguments=None,
            data_args: DataTrainingArguments=None,
            training_args: TrainingArguments=None,
    ):
        self.accelerator = accelerator
        self.logger = accelerate.logging.get_logger(name='Main')

        # Make one log on every process with the configuration for debugging
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            datefmt='%m/%d/%Y %H:%M:%S',
            level=logging.INFO,
        )
        self.logger.info(accelerator.state, main_process_only=False)
        self.logger.info(f'Working directory is: {os.getcwd()}')

        if accelerator.is_local_main_process:
            datasets.utils.logging.set_verbosity_warning()
            transformers.utils.logging.set_verbosity_info()
        else:
            datasets.utils.logging.set_verbosity_error()
            transformers.utils.logging.set_verbosity_error()

        self.setup_neptune(args=args)
        # self.setup_wandb(model_args, data_args, training_args)


    def setup_wandb(
            self,
            model_args: ModelArguments,
            data_args: DataTrainingArguments,
            training_args: TrainingArguments,
    ):
        """
        Setup Weights and Biases logging.

        Create a weights and biases run and log the model arguments, data arguments, and training arguments.
        Create an experiment name based on the model name and learning rate.

        :param model_args: The model arguments. See `pre_training_flags.py`.
        :param data_args: The data arguments. See `pre_training_flags.py`.
        :param training_args: The training arguments. See `pre_training_flags.py`.
        :return: None
        """
        experiment_name = (
            f"{model_args.model_name_or_path.split('/')[-1].replace('.', '_')}_lr_{training_args.learning_rate}"
        )

        model_args_dict = dataclasses.asdict(model_args)
        data_args_dict = dataclasses.asdict(data_args)
        training_args_dict = dataclasses.asdict(training_args)


        self.accelerator.init_trackers(
            project_name='principled-pre-training',

            # Initialize config from model args, data args, and training args
            config={
                **model_args_dict,
                **data_args_dict,
                **training_args_dict,
            },

            # Initialize the run name and tags
            init_kwargs={
                "wandb": {
                    "name": experiment_name,
                }
            }
        )

    def setup_neptune(self, args):
        if args.logging.neptune:
            neptune_logger = neptune.init_run(
                project=args.logging.neptune_creds.project,
                api_token=args.logging.neptune_creds.api_token,
                tags=[str(item) for item in args.logging.neptune_creds.tags.split(",")],
            )
        else:
            neptune_logger = None

        self.neptune_logger = neptune_logger

        with omegaconf.open_dict(args):
            if neptune_logger is not None:
                args.neptune_id = neptune_logger["sys/id"].fetch()

    def log_args(self, args):
        if self.neptune_logger is not None:
            logging_args = omegaconf.OmegaConf.to_container(args, resolve=True)
            self.neptune_logger['args'] = logging_args

    def log_stats(self, stats, step, args, prefix=''):
        if self.neptune_logger is not None:
            for k, v in stats.items():
                self.neptune_logger[f'{prefix}{k}'].log(v, step=step)

        msg_start = f'[{prefix[:-1]}] Step {step} out of {args.optim.total_steps}' + ' | '
        dict_msg = ' | '.join([f'{k.capitalize()} --> {v:.3f}' for k, v in stats.items()]) + ' | '

        msg = msg_start + dict_msg

        self.log_message(msg)

    def log_message(self, msg):
        self.logger.info(msg)

    def finish(self):
        if self.neptune_logger is not None:
            self.neptune_logger.stop()



    # def log_metrics(
    #         self,
    #         metrics: typing.Dict[str, float],
    #         step: int,
    # ):
    #     """
    #     Log metrics to Weights and Biases.
    #
    #     :param metrics: A dictionary of metrics to log.
    #     :param step: The step at which the metrics were logged.
    #     """
    #     self.accelerator.log_metrics(metrics, step)
    #     self.logger.info(f"Step {step}:\n{metrics}")
    #
    #
    # def log(self, *args, **kwargs):
    #     """
    #     Log to the console.
    #
    #     :param args: The arguments to log.
    #     :param kwargs: The keyword arguments to log.
    #     """
    #     self.logger.info(*args, **kwargs)
    #
    #
    # def print(self, *args, **kwargs):
    #     """
    #     Print to the console.
    #
    #     :param args: The arguments to print.
    #     :param kwargs: The keyword arguments to print.
    #     """
    #     self.accelerator.print(*args, **kwargs)
