import math
from typing import Iterable, Tuple
import torch
import transformers
import omegaconf

from constants import base_constants
from utils import logging_utils


class AdamWScale(torch.optim.Optimizer):
    """
    This AdamW implementation is copied from Huggingface.
    We modified it with Adagrad scaling by rms of a weight tensor

    Implements Adam algorithm with weight decay fix as introduced in [Decoupled Weight Decay
    Regularization](https://arxiv.org/abs/1711.05101).

    Parameters:
        params (`Iterable[nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (`float`, *optional*, defaults to 1e-3):
            The learning rate to use.
        betas (`Tuple[float,float]`, *optional*, defaults to (0.9, 0.999)):
            Adam's betas parameters (b1, b2).
        eps (`float`, *optional*, defaults to 1e-6):
            Adam's epsilon for numerical stability.
        weight_decay (`float`, *optional*, defaults to 0):
            Decoupled weight decay to apply.
        correct_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to correct bias in Adam (for instance, in Bert TF repository they use `False`).
        no_deprecation_warning (`bool`, *optional*, defaults to `False`):
            A flag used to disable the deprecation warning (set to `True` to disable the warning).
    """

    def __init__(
        self,
        params: Iterable[torch.nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
    ):
        """

        :param params: The parameters of the model to optimize. A parameter is a Tensor that has requires_grad=True.
            Params is a dict of string to torch.Tensor, where the string key is the name of the parameter in the model.
        :param lr: The learning rate to use. This defines the step size at each iteration while moving toward a minimum
            of a loss function. After the warmup phase, the model's learning rate will take on this value.
        :param betas: Adams beta parameters (b1, b2). These control the exponential moving average of the gradient
            (b1) and the exponential moving average of the squared gradient (b2). Default: (0.9, 0.999)
        :param eps: Adams epsilon. This is a small number used to prevent any division by zero in the implementation.
            Default: 1e-6
        :param weight_decay: Weight decay to apply. This is a regularization term that decays the model's weights
            to prevent overfitting. Default: 0.0
        :param correct_bias: Whether or not to correct bias in Adam (for instance, in Bert TF repository they use False).
            Default: True
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    @staticmethod
    def _rms(tensor: torch.Tensor) -> torch.Tensor:
        """
        Computes the root mean square of the tensor.

        :param tensor: The tensor for which we are computing the root mean square. This is used to scale the learning
            rate.
        :return: The root mean square of the tensor.
        """
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    def step(self, closure: callable = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]
                beta1, beta2 = group["betas"]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                # /Adapt Step from Adafactor
                step_size = step_size * max(1e-3, self._rms(p.data))
                # /Adapt Step from Adafactor

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                if group["weight_decay"] > 0.0:
                    p.data.add_(p.data, alpha=(-group["lr"] * group["weight_decay"]))

        return loss

def get_lr_scheduler(
        optimizer: torch.optim.Optimizer,
        args: omegaconf.DictConfig,
        logger: logging_utils.Logger,
) -> torch.optim.lr_scheduler.LRScheduler:
    """
    Get the learning rate scheduler.

    :param optimizer: The optimizer for which we are generating a scheduler.
    :param args: The omegaconf configuration used to generate the scheduler.
    :param logger: The logger. See `logging_utils.py` for more details.
    :return: The learning rate scheduler.
    """
    logger.log_message(f'Using lr scheduler: {args.optim.lr_scheduler}')
    if args.optim.lr_scheduler == 'cosine':

        # Scheduler Part 1/2: Linear warmup
        scheduler1 = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.5,
            end_factor=1,
            total_iters=args.optim.warmup_steps,
            last_epoch=-1,
        )

        # Scheduler Part 2/2: Cosine annealing (after warmup)
        scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.optim.total_steps - args.optim.warmup_steps,
            eta_min=args.optim.final_cosine,
        )

        # Full scheduler: Warmup + Cosine Annealing
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[scheduler1, scheduler2],
            milestones=[args.optim.warmup_steps]
        )

    elif args.optim.lr_scheduler == 'legacy':

        msg = "You are using T5 legacy LR Schedule, it's independent from the optim.base_lr"
        logger.log_message(msg)

        num_steps_optimizer1 = math.ceil(args.optim.total_steps * 0.9)
        iters_left_for_optimizer2 = args.optim.total_steps - num_steps_optimizer1

        scheduler1 = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: min(
                1e-2, 1.0 / math.sqrt(step)
            ) / args.optim.base_lr if step else 1e-2 / args.optim.base_lr
        )

        scheduler2 = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=(
                    min(1e-2, 1.0 / math.sqrt(num_steps_optimizer1)) / args.optim.base_lr
            ),
            end_factor=0,
            total_iters=iters_left_for_optimizer2,
            last_epoch=-1,
        )

        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[scheduler1, scheduler2],
            milestones=[num_steps_optimizer1]
        )
    elif args.optim.lr_scheduler == 'constant':
        lr_scheduler = transformers.get_scheduler(
            name=args.optim.lr_scheduler,
            optimizer=optimizer,
        )
    else:
        raise NotImplementedError

    return lr_scheduler

def get_optimizer(
        model: transformers.T5ForConditionalGeneration,
        args: omegaconf.DictConfig,
        logger: logging_utils.Logger,
) -> torch.optim.Optimizer:
    """
    Get the optimizer. This is used to optimize the model's parameters during training.


    :param model: The model for which we are generating an optimizer.
    :param args: The omegaconf configuration used to generate the optimizer.
    :param logger: The logger. See `logging_utils.py` for more details.
    :return: The optimizer.
    """

    # Optimizer grouped parameters consist of two groups: (1) parameters with weight decay and (2) parameters without
    # weight decay. The parameters without weight decay are the bias and LayerNorm parameters. The parameters with
    # weight decay are all other parameters.
    optimizer_grouped_parameters = [
        {
            "params": [
                parameter for name, parameter in model.named_parameters() if
                not any(no_decay_type in name for no_decay_type in base_constants.SchedulerConstants.NO_DECAY)
            ],
            "weight_decay": args.optim.weight_decay,
        },
        {
            "params": [
                parameter for name, parameter in model.named_parameters() if
                any(no_decay_type in name for no_decay_type in base_constants.SchedulerConstants.NO_DECAY)
            ],
            "weight_decay": 0.0,
        },
    ]

    if args.optim.name == base_constants.OptimizerConstants.ADAMW:
        logger.log_message('Using AdamW optimizer')
        optimizer = transformers.AdamW(
            optimizer_grouped_parameters,
            lr=args.optim.base_lr,
        )
    elif args.optim.name == base_constants.OptimizerConstants.ADAMWSCALE:
        logger.log_message('Using AdamWScale optimizer')
        optimizer = AdamWScale(
            optimizer_grouped_parameters,
            lr=args.optim.base_lr,
        )
    elif args.optim.name == base_constants.OptimizerConstants.ADAFACTOR:
        logger.log_message('Using Adafactor optimizer')
        optimizer = transformers.Adafactor(
            optimizer_grouped_parameters,
            lr=args.optim.base_lr,
            relative_step=False,
        )
    else:
        raise NotImplementedError

    return optimizer