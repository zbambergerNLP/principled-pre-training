import typing

import torch
import time
import evaluate as hf_evaluate
from datasets.iterable_dataset import IterableDataset
import omegaconf
import accelerate
import transformers

from utils import logging_utils
from constants import base_constants


def maybe_save_checkpoint(
        accelerator: accelerate.Accelerator,
        args: omegaconf.DictConfig,
):
    """
    Conditionally saves the checkpoint based on the current training step.

    :param accelerator: The accelerator object. Notably, this contains the save_state method, which saves the state of
        the model, optimizer, and lr_scheduler.
    :param args: The hydra config which contains the model, data, and training arguments. See the
        configs/default_config.yaml file for the default configuration.
    """
    if (
        args.current_train_step > args.optim.total_steps
        or args.current_train_step % args.checkpoint.every_steps == 0
    ):
        output_dir = f'checkpoint-{args.mode}-{args.current_train_step}'
        accelerator.save_state(output_dir=output_dir)


def maybe_eval_predict(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        logger: logging_utils.Logger,
        args: omegaconf.DictConfig,
        tokenizer: transformers.PreTrainedTokenizer,
):
    """
    Conditionally evaluates and predicts based on the current training step.

    :param model: A encoder-decoder pytorch model.
    :param dataloader: The dataloader to use for evaluation and prediction.
    :param logger: The logger object. See utils/logging_utils.py for more details.
    :param args: The hydra config which contains the model, data, and training arguments. See the
        configs/default_config.yaml file for the default configuration.
    :param tokenizer: The tokenizer to use for decoding the predictions.
    """

    # Perform evaluation if we have reached the end of training or if 'args.eval.every_steps' steps have passed since
    # the last evaluation.
    if (
        args.current_train_step > args.optim.total_steps
        or args.current_train_step % args.evaluate.every_steps == 0
    ):
        model.eval()

        # Perform evaluation on the validation set. Do not perform gradient calculations.
        with torch.no_grad():
            evaluate(model=model, dataloader=dataloader, logger=logger, args=args, tokenizer=tokenizer)

        args.last_log = time.time()
        model.train()


def maybe_logging(
        averager: logging_utils.Averager,
        args: omegaconf.DictConfig,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        logger: logging_utils.Logger,
):
    """
    Conditionally logs based on the current training step.

    :param averager: A logging_utils.Averager object. This is used to average the stats before logging.
    :param args: The hydra config which contains the model, data, and training arguments. See the
        configs/default_config.yaml file for the default configuration.
    :param model: A encoder-decoder pytorch model.
    :param optimizer: The optimizer object.
    :param logger: The logger object. See utils/logging_utils.py for more details.
    """
    if args.current_train_step % args.logging.every_steps == 0:
        stats = extra_stats(args, model, optimizer)

        averager.update(stats)
        averaged_stats = averager.average()

        logger.log_stats(
            stats=averaged_stats,
            step=args.current_train_step,
            args=args,
            prefix='train/'
        )

        args.last_log = time.time()


def maybe_grad_clip_and_grad_calc(
        accelerator: accelerate.Accelerator,
        model: torch.nn.Module,
        args: omegaconf.DictConfig,
) -> typing.Dict[str, float]:
    """
    Conditionally clips the gradients and calculates the gradient norm.
    :param accelerator: The accelerator object. Notably, this contains the clip_grad_norm_ method, which clips the
        gradients.
    :param model: A encoder-decoder pytorch model.
    :param args: The hydra config which contains the model, data, and training arguments. See the
        configs/default_config.yaml file for the default configuration.
    :return: A dictionary of stats. This contains the gradient norm if args.logging.grad_l2 is True. Otherwise, it
        returns an empty dictionary. This is because calculating the gradient norm is expensive.
    """
    if args.optim.grad_clip > 0:
        grad_l2 = accelerator.clip_grad_norm_(
            parameters=model.parameters(),
            max_norm=args.optim.grad_clip,
            norm_type=2,
        )
    else:
        grad_l2 = None

    if args.logging.grad_l2:
        if grad_l2 is None:
            grad_l2 = (
                sum(p.grad.detach().data.norm(2).item() ** 2 for p in model.parameters()) ** 0.5
            )

        return {'grad_l2': grad_l2}
    else:
        return {}


def extra_stats(
        args: omegaconf.DictConfig,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
) -> typing.Dict[str, float]:
    """
    Calculates extra stats to log. This includes the weights_l2 if args.logging.weights_l2 is True. Otherwise, it
    returns an empty dictionary.

    :param args: The hydra config which contains the model, data, and training arguments. See the
        configs/default_config.yaml file for the default configuration.
    :param model: A encoder-decoder pytorch model.
    :param optimizer: The optimizer object.
    :return: A dictionary of stats. This contains the weights_l2 if args.logging.weights_l2 is True. Otherwise, it
        returns a dictionary with only the learning rate and seconds_per_step.
    """
    stats = {}

    if args.logging.weights_l2:
        weights_l2 = sum(p.detach().norm(2).item() ** 2 for p in model.parameters()) ** 0.5
        stats['weights_l2'] = weights_l2

    stats['lr'] = optimizer.param_groups[0]['lr']
    stats['seconds_per_step'] = (time.time() - args.last_log) / args.logging.every_steps

    return stats


def forward(
        model: torch.nn.Module,
        batch: typing.Dict[str, torch.Tensor],
        calc_acc=False,
) -> typing.Tuple[torch.Tensor, typing.Dict[str, float]]:
    """
    Forward pass for the model.

    :param model: A encoder-decoder pytorch model.
    :param batch: A batch of data. This is a dictionary of tensors. The keys are 'input_ids', 'attention_mask', and
        'labels'. The values are the corresponding tensors.
    :param calc_acc: Whether to calculate the accuracy or not.
    :return: A tuple of the loss and a dictionary of stats. The stats contains the loss and optionally the accuracy.
    """
    outputs = model(**batch)
    loss = outputs.loss

    stats = {}
    stats['loss'] = loss.detach().float().item()

    if calc_acc:
        correct = (outputs.logits.argmax(-1) == batch["labels"]).sum().item()
        accuracy = correct / batch["labels"].numel()
        stats['accuracy'] = accuracy

    return loss, stats


def evaluate(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        logger: logging_utils.Logger,
        args: omegaconf.DictConfig,
        tokenizer: transformers.PreTrainedTokenizer,
):
    """
    Evaluates the model on the given dataloader.

    :param model: An encoder-decoder pytorch model.
    :param dataloader: The dataloader to use for evaluation.
    :param logger: The logger object. See utils/logging_utils.py for more details.
    :param args: The hydra config which contains the model, data, and training arguments. See the
        configs/default_config.yaml file for the default configuration.
    :param tokenizer: The tokenizer to use for decoding the predictions.
    """
    args.last_log = time.time()
    averager = logging_utils.Averager()

    # TODO: Use tqdm to show progress bar instead of using 'enumerate'.
    for batch_id, batch in enumerate(dataloader, start=1):
        if batch_id == args.evaluate.corrected_steps * args.optim.grad_acc:
            break

        _, stats = forward(model, batch, calc_acc=True)
        averager.update(stats)

    averager.update({'time': time.time() - args.last_log})
    averaged_stats = averager.average()

    logger.log_stats(
        stats=averaged_stats,
        step=args.current_train_step,
        args=args,
        prefix='eval/'
    )


def predict(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        logger: logging_utils.Logger,
        args: omegaconf.DictConfig,
        tokenizer: transformers.PreTrainedTokenizer,
):
    """
    Predicts on the given dataloader and saves the predictions to a file.
    :param model: An encoder-decoder pytorch model.
    :param dataloader: The dataloader to use for prediction.
    :param logger: The logger object. See utils/logging_utils.py for more details.
    :param args: The hydra config which contains the model, data, and training arguments. See the
        configs/default_config.yaml file for the default configuration.
    :param tokenizer: The tokenizer to use for decoding the predictions.
    """
    args.last_log = time.time()
    # TODO: Compute rouge score indipendently. Don't rely on the evaluate library.
    metric = hf_evaluate.load('rouge')
    samples_seen = 0

    def decode(preds: torch.Tensor):
        """
        Decodes the predictions.
        :param preds: The predictions. This is a tensor of shape (batch_size, max_target_len). The values are the
            token ids. The padding tokens are represented by -100. The other tokens are represented by their token ids.
        :return: The decoded predictions. This is a list of strings. The strings are the decoded predictions. The
            strings are stripped of leading and trailing whitespaces. The strings are also stripped of special tokens.
        """
        preds[preds == -100] = tokenizer.pad_token_id
        preds = tokenizer.batch_decode(
            preds, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        preds = [pred.strip() for pred in preds]
        return preds

    for step, batch in enumerate(dataloader):
        predictions = model.generate(
            input_ids=batch[base_constants.TokenizedTrainingExampleConstants.INPUT_IDS],
            attention_mask=batch[base_constants.TokenizedTrainingExampleConstants.ATTENTION_MASK],
            max_length=args.data.max_target_len,
            generation_config=model.generation_config,
        )
        predictions = decode(predictions)
        references = decode(batch[base_constants.TokenizedTrainingExampleConstants.LABELS])

        # If we are in a multiprocess environment, the last batch has duplicates
        if step == len(dataloader) - 1:
            predictions = predictions[: len(dataloader.dataset) - samples_seen]
            references = references[: len(dataloader.dataset) - samples_seen]
        else:
            samples_seen += len(references)

        metric.add_batch(
            predictions=predictions,
            references=references,
        )

    eval_metric = metric.compute(use_stemmer=True, use_aggregator=False)
    rougeL = sum(eval_metric["rougeL"]) * 100 / len(eval_metric["rougeL"])

    logger.log_stats(
        stats={
            "rougeL": rougeL,
            "time": time.time() - args.last_log,
        },
        step=args.current_train_step,
        args=args,
        prefix="test/",
    )


def train(
        model: torch.nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        test_dataloader: torch.utils.data.DataLoader,
        accelerator: accelerate.Accelerator,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        optimizer: torch.optim.Optimizer,
        logger: logging_utils.Logger,
        args: omegaconf.DictConfig,
        tokenizer: transformers.PreTrainedTokenizer,
):
    """
    Trains the model.

    :param model: An encoder-decoder pytorch model.
    :param train_dataloader: The dataloader to use for training.
    :param test_dataloader: The dataloader to use for evaluation and prediction.
    :param accelerator: The accelerator object. Notably, this contains the backward method, which is used to calculate
        the gradients. The accelerator is responsible for distributing the data across (potentially) multiple GPUs.
    :param lr_scheduler: The learning rate scheduler object. This is used to schedule the learning rate. See
        utils/model_utils.py for more details.
    :param optimizer: The optimizer object. This is used to calculate the gradients. See utils/model_utils.py for more
        details.
    :param logger: The logger object. See utils/logging_utils.py for more details.
    :param args: The hydra config which contains the model, data, and training arguments.
    :param tokenizer: The tokenizer to use for decoding the predictions.
    """
    logger.log_message('Starting training...')
    model.train()

    train_averager = logging_utils.Averager()

    while args.current_train_step <= args.optim.total_steps:

        if isinstance(train_dataloader.dataset, IterableDataset):
            train_dataloader.dataset.set_epoch(args.current_epoch)

        # In case there is a remainder from previous epoch, we need to reset the optimizer
        optimizer.zero_grad(set_to_none=True)

        # TODO: Handle the case where the dataloader is an IterableDataset. Concretely, dataloaders which wrap
        #  streaming datasets don't support 'enumerate'. This is because the dataloader doesn't know the length of the
        #  dataset (a core principle of streaming datasets).
        for batch_id, batch in enumerate(train_dataloader, start=1):

            if args.current_train_step > args.optim.total_steps:
                break

            loss, stats = forward(model=model, batch=batch)
            accelerator.backward(loss / args.optim.grad_acc)
            train_averager.update(stats=stats)

            # If we have accumulated enough gradients, we can perform an optimizer step, a learning rate scheduler step,
            # and log the stats.
            if batch_id % args.optim.grad_acc == 0:
                stats = maybe_grad_clip_and_grad_calc(
                    accelerator=accelerator,
                    model=model,
                    args=args)
                train_averager.update(stats)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                maybe_logging(
                    averager=train_averager,
                    args=args,
                    model=model,
                    optimizer=optimizer,
                    logger=logger,
                )
                maybe_eval_predict(
                    model=model,
                    dataloader=test_dataloader,
                    logger=logger,
                    args=args,
                    tokenizer=tokenizer,
                )
                maybe_save_checkpoint(accelerator=accelerator, args=args)

                args.current_train_step += 1

        args.current_epoch += 1

    maybe_eval_predict(
        model=model,
        dataloader=test_dataloader,
        logger=logger,
        args=args,
        tokenizer=tokenizer,
    )
    maybe_save_checkpoint(accelerator=accelerator, args=args)