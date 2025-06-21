import torch

from omegaconf import ListConfig, OmegaConf, DictConfig
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.etc.constants import TRANSFORMER_BASED_MODELS
from .lr_scheduler import CosineLRScheduler


def setOptimizerAndScheduler(config: DictConfig | ListConfig, model) -> dict:
    optimizer = setOptimizer(config=config, model=model)
    scheduler = setScheduler(config=config, optimizer=optimizer)

    config_dict = {"optimizer": optimizer}

    if scheduler:
        config_dict["lr_scheduler"] = scheduler

    return config_dict


def setOptimizer(config: DictConfig | ListConfig, model) -> None | torch.optim.Optimizer:
    optimizer = None
    optimizer_name = config.args.optimizer.name

    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config.args.optimizer.params.lr)

    elif optimizer_name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=config.args.optimizer.params.lr)

    elif optimizer_name == "AdamW":
        # FT-transformerの場合、特定のパラメータをProtectして最適化する必要がある
        # 原文：
        # In the paper, some of FT-Transformer's parameters
        # were protected from the weight decay regularization.
        # There is a special method for doing that:
        if config.args.models.name == "FTMultiMaskTransformer":
            parameters = model.parameters()

        elif config.args.models.name in TRANSFORMER_BASED_MODELS:
            parameters = model.make_parameter_groups()

        else:
            parameters = model.parameters()

        optimizer = torch.optim.AdamW(
            parameters,
            lr=config.args.optimizer.params.lr,
            weight_decay=config.args.optimizer.params.weight_decay,
        )

    return optimizer


def setScheduler(config: DictConfig | ListConfig, optimizer) -> None | dict:
    scheduler_name = config.args.scheduler.name
    use_multiprocessing = False

    if "advance_setting" in config.args:
        if "use_multiprocessing" in config.args.advance_setting:
            use_multiprocessing = config.args.advance_setting.use_multiprocessing

    lr_scheduler = None

    if scheduler_name == "CosineLRScheduler":
        warmup_lr_init = config.args.scheduler.params.warmup_lr_init

        if isinstance(warmup_lr_init, float):
            pass

        elif warmup_lr_init == "same":
            warmup_lr_init = config.args.optimizer.params.lr

        elif "digits_smaller" in warmup_lr_init:
            warmup_lr_init = warmup_lr_init.split("_")
            warmup_lr_init = config.args.optimizer.params.lr * (0.1 ** int(warmup_lr_init[-1]))

        scheduler = CosineLRScheduler(
            optimizer=optimizer,
            t_initial=config.args.scheduler.params.t_initial,
            warmup_t=config.args.scheduler.params.warmup_t,
            warmup_lr_init=warmup_lr_init,
            lr_min=config.args.scheduler.params.lr_min,
        )

        lr_scheduler = {
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": 1,
            "name": "lr_fixed",
        }

    elif scheduler_name == "ReduceLROnPlateau":
        scheduler = ReduceLROnPlateau(
            optimizer=optimizer,
            mode="min",
            factor=config.args.scheduler.params.factor,
            patience=config.args.scheduler.params.patience,
            verbose=False if use_multiprocessing else True,
        )

        lr_scheduler = {
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": 1,
            "name": "lr_fixed",
            "monitor": config.args.scheduler.params.monitor,
        }

    return lr_scheduler
