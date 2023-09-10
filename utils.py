import transformers
import typing
import copy
import random
import torch
import numpy as np

def set_seed(seed: int):
    """Set the seed for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    transformers.set_seed(seed)

class TrainingMetricsCallback(transformers.trainer_callback.TrainerCallback):
    """A callback which logs the training metrics to W&B."""

    def __init__(
            self,
            trainer: transformers.Trainer,
            eval_to_train_ratio: float = 2,  # TODO: Make this a flag.
    ) -> None:
        super().__init__()
        self._trainer = trainer
        self._eval_to_train_ratio = eval_to_train_ratio

    def on_epoch_end(
            self,
            args: transformers.training_args.TrainingArguments,
            state: transformers.trainer_callback.TrainerState,
            control: transformers.trainer_callback.TrainerControl,
            **kwargs: typing.Any,
    ) -> transformers.trainer_callback.TrainerControl:
        """Log the training metrics to W&B."""
        control_copy = copy.deepcopy(control)
        if state.epoch % self._eval_to_train_ratio == 0:
            self._trainer.evaluate(
                eval_dataset=self._trainer.train_dataset,
                metric_key_prefix="training",
            )
        # control_copy.should_evaluate = False
        return control_copy