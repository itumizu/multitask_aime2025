from omegaconf import DictConfig, ListConfig

from src.etc.constants import USE_PYTORCH_TABULAR_MODELS

if USE_PYTORCH_TABULAR_MODELS:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import Callback, ModelCheckpoint, EarlyStopping, LearningRateMonitor

else:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import Callback, ModelCheckpoint, EarlyStopping, LearningRateMonitor


def selectCallbacks(config: DictConfig | ListConfig, use_multiprocessing: bool = False):
    callbacks = []
    is_merged_layers_by_tasks = False

    if "is_merged_layers_by_tasks" in config.args.models.params:
        is_merged_layers_by_tasks = config.args.models.params.is_merged_layers_by_tasks

    if config.args.callbacks.model_checkpoint.use_callback:
        checkpoint_callback = ModelCheckpoint(
            monitor=config.args.callbacks.model_checkpoint.monitor,
            mode=config.args.callbacks.model_checkpoint.mode,
            save_top_k=1,
            save_last=True,
            verbose=False if use_multiprocessing else True,
        )

        callbacks.append(checkpoint_callback)

        if len(config.args.tasks.targets) > 1 and not is_merged_layers_by_tasks:
            for target_name in config.args.tasks.targets:
                monitor_name = config.args.callbacks.model_checkpoint.monitor + f"_{target_name}"

                callbacks.append(
                    ModelCheckpoint(
                        monitor=f"{monitor_name}",
                        mode=config.args.callbacks.model_checkpoint.mode,
                        save_top_k=1,
                        save_last=True,
                        verbose=False if use_multiprocessing else True,
                        # filename='{epoch}-' + monitor_name + '-{' + monitor_name + ':.2f}'
                    )
                )

    if config.args.callbacks.early_stopping.use_callback:
        early_stopping_callback = EarlyStopping(
            monitor=config.args.callbacks.early_stopping.monitor,
            mode=config.args.callbacks.early_stopping.mode,
            patience=config.args.callbacks.early_stopping.patience,
            min_delta=0.00,
            verbose=False if use_multiprocessing else True,
        )

        callbacks.append(early_stopping_callback)

    if config.args.callbacks.learning_rate_monitor.use_callback:
        lr_monitor_callback = LearningRateMonitor(logging_interval="epoch")

        callbacks.append(lr_monitor_callback)

    return callbacks


class ExperimentLoggingCallback(Callback):
    def __init__(self, config: DictConfig | ListConfig, what="epochs", verbose=True):
        self.what = what
        self.config = config
        self.verbose = verbose
        self.state = {"epochs": 0, "batches": 0}

        self.metrics_per_epoch_dict = {
            "train": [],
            "val": [],
            "test": [],
        }

    def logOptimizer(self, phase: str, trainer: pl.Trainer):
        scheduler_config = trainer.lr_scheduler_configs
        print(scheduler_config)

    def logExperiment(self, phase: str, trainer: pl.Trainer):
        metrics = trainer.callback_metrics
        self.metrics_per_epoch_dict[phase].append({key: value.detach().cpu() for key, value in metrics.items()})

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        super().on_train_epoch_end(trainer, pl_module)
        self.logExperiment(phase="train", trainer=trainer)
        return

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        super().on_validation_epoch_end(trainer, pl_module)
        self.logExperiment(phase="val", trainer=trainer)
        return

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.logExperiment(phase="test", trainer=trainer)
        return super().on_validation_epoch_end(trainer, pl_module)

    def load_state_dict(self, state_dict):
        self.state.update(state_dict)

    def state_dict(self):
        return self.state.copy()

    def getLoggingDict(self):
        output_metric_value_dict = {
            "train": [],
            "val": [],
            "test": [],
        }

        for phase, metrics_list in self.metrics_per_epoch_dict.items():
            for metric in metrics_list:
                output_metric_value_dict[phase].append({key: value.cpu() for key, value in metric.items()})

        return output_metric_value_dict
