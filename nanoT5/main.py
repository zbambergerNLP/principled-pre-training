import typing

from accelerate import Accelerator
import omegaconf
import hydra
import torch
import time

from .utils import (
    setup_basics,
    train,
    predict,
    evaluate,
    get_lr_scheduler,
    get_optimizer,
    get_tokenizer,
    get_model,
    get_dataloaders,
    get_config,
    constants,  # Move to a new directory that is dedicated to constants
)

"""
Example usage (from the nanoT5 directory):

srun --partition nlp --account nlp --gres=gpu:4 -c 80 --pty bash
conda activate ml_training
accelerate launch -m nanoT5.main
"""


@hydra.main(config_path="configs", config_name="default", version_base='1.3')
def main(dict_config: omegaconf.DictConfig):
    """
    Pre-train an encoder-decoder model, or fine-tune it on a downstream task.

    :param dict_config: The hydra config which contains the model, data, and training arguments. See the
        configs/default.yaml file for the default configuration.
    """

    # Determine which kind of logging the user wants via the dict_config.
    report_to: typing.List[str] = []
    if dict_config.logging.wandb:
        report_to.append(constants.MonitoringPlatform.WANDB.value)

    # Initialize the accelerator and logger.
    accelerator = Accelerator(
        cpu=dict_config.device == constants.Device.CPU.value,
        mixed_precision=dict_config.precision,
        log_with=report_to,
    )
    logger = setup_basics(accelerator=accelerator, args=dict_config)
    logger.log_message("Initialized accelerator and logger successfully.")

    config = get_config(args=dict_config, logger=logger)
    logger.log_message(f"Using config: {config}")

    model = accelerator.prepare(get_model(args=dict_config, config=config, logger=logger))
    # model = get_model(args=dict_config, config=config, logger=logger)
    logger.log_message(f"Using model: {model}")

    tokenizer = get_tokenizer(args=dict_config, logger=logger)
    logger.log_message(f"Using tokenizer: {tokenizer}")

    # TODO: resolve the following warning:
    #  FSDP Warning: When using FSDP, it is efficient and recommended to call prepare for the model before creating the
    #  optimizer
    optimizer = get_optimizer(model=model, args=dict_config, logger=logger)
    logger.log_message(f"Using optimizer: {optimizer}")

    lr_scheduler = get_lr_scheduler(optimizer=optimizer, args=dict_config, logger=logger)
    logger.log_message(f"Using lr_scheduler: {lr_scheduler}")

    train_dataloader, test_dataloader = get_dataloaders(
        tokenizer=tokenizer, 
        config=config, 
        args=dict_config, 
        logger=logger,
    )

    logger.log_args(args=dict_config)

    (
        optimizer,
        lr_scheduler,
        train_dataloader,
        test_dataloader,
    ) = accelerator.prepare(
        # model,
        optimizer,
        lr_scheduler,
        train_dataloader,
        test_dataloader
    )

    if dict_config.model.compile:
        model = torch.compile(model)

    with omegaconf.open_dict(dict_config):
        dict_config.current_train_step = 1
        dict_config.current_epoch = 1
        dict_config.last_log = time.time()

    if dict_config.eval_only:
        model.eval()
        with torch.no_grad():
            evaluate(
                model=model,
                dataloader=test_dataloader,
                logger=logger,
                args=dict_config,
                tokenizer=tokenizer,
            )
    elif dict_config.predict_only:
        model.eval()
        with torch.no_grad():
            predict(
                model=model,
                dataloader=test_dataloader,
                logger=logger,
                args=dict_config,
                tokenizer=tokenizer,
            )
    else:
        train(
            model=model,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            accelerator=accelerator,
            lr_scheduler=lr_scheduler,
            optimizer=optimizer,
            logger=logger,
            args=dict_config,
            tokenizer=tokenizer,
        )

    logger.finish()


if __name__ == "__main__":
    main()
