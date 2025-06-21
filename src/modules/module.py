import copy
from typing import Any, Tuple

import numpy as np

import torch
from torch.utils.data import DataLoader

import lightning.pytorch as pl
from lightning.pytorch.utilities.types import STEP_OUTPUT

from omegaconf import DictConfig, ListConfig
from src.loaders.dataset import HealthDataset
from src.utils.neural_network.select_loss import selectLoss
from src.utils.neural_network.set_optimizer import setOptimizerAndScheduler
from src.utils.calc_metrics import calculateMetrics

from src.etc.constants import (
    BINARY_CLASSIFICATION_LOSS_FUNCTION_LIST,
    MULTICLASS_CLASSIFICATION_LOSS_FUNCTION_LIST,
    REGRESSION_LOSS_FUNCTION_LIST,
    TRANSFORMER_BASED_MODELS,
)

class HealthExperimentModule(pl.LightningModule):
    def __init__(
        self,
        config: DictConfig | ListConfig,
        model,
        fold_test_num: int,
        fold_val_num: int,
        accelerator="cpu",
        devices: list | str = [],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.config = copy.deepcopy(config)
        self.model = model
        self.is_multitask = True
        self.loss_function_dict = {}

        self.automatic_optimization = True

        if "automatic_optimization" in self.config.args.models:
            self.automatic_optimization = self.config.args.models.automatic_optimization

        for target_name in self.config.args.tasks.targets:
            loss_function_name = self.config.args.models.loss_functions[target_name]
            self.loss_function_dict[target_name] = selectLoss(loss_function_name=loss_function_name)

        self.fold_val_num = fold_val_num
        self.fold_test_num = fold_test_num

        self.outputs_dict = {"train": [], "val": [], "test": []}
        self.results_dict = {"train": {}, "val": {}, "test": {}}
        self.logging_dict = {"train": [], "val": [], "test": []}

        self.use_step_multi_task = False
        self.started_experiment = False
        self.train_loss_buffer = np.zeros([len(self.config.args.tasks.targets), self.config.args.optimizer.max_epochs])

        self.send_epochs = True

    def forward(self, x: dict):
        x = self.model.forward(**x)
        return x

    def calculateLoss(self, idx, target_name: str, batch: dict, x_list, phase: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        y_true = batch[f"y_{target_name}"]
        y_pred: torch.Tensor = x_list[:, idx : idx + 1]

        loss_function_name = self.config.args.models.loss_functions[target_name]
        loss = self.loss_function_dict[target_name](y_pred, y_true)

        if loss_function_name in BINARY_CLASSIFICATION_LOSS_FUNCTION_LIST:
            y_pred = torch.sigmoid(y_pred)

        elif loss_function_name in MULTICLASS_CLASSIFICATION_LOSS_FUNCTION_LIST:
            y_pred = torch.softmax(y_pred, dim=-1)

        elif loss_function_name in REGRESSION_LOSS_FUNCTION_LIST:
            pass

        else:
            raise

        if not "loss_function_weights" in self.config.args.models:
            loss_weight = torch.tensor(1.0)

        else:
            loss_weight = torch.tensor(self.config.args.models.loss_function_weights[target_name])

        is_sum_this_task = True

        if is_sum_this_task:
            loss = loss * loss_weight

        loss_output = loss

        return (
            loss,
            loss_output,
            y_pred,
            y_true,
        )

    def experiment_step(self, batch, batch_idx, phase: str) -> STEP_OUTPUT:
        outputs: dict = {}
        self.started_experiment = not self.trainer.sanity_checking
        outputs[f"participant_id_list"] = batch["index_name"]

        x_data = {key: value for key, value in batch.items() if "x" in key and not "index" in key}
        x_list = self.forward(x=x_data)

        if self.config.args.models.name in TRANSFORMER_BASED_MODELS:
            attention_dict = x_list[1]
            x_list = x_list[0]

            if phase == "test":
                outputs["attention"] = attention_dict

        else:
            raise

        loss_list = torch.zeros(len(self.config.args.tasks.targets), device=self.device)
        loss_output_list = torch.zeros(len(self.config.args.tasks.targets), device=self.device)

        ### summarize loss
        loss_sum = torch.zeros(1, device=self.device)
        loss_output_sum = torch.zeros(1, device=self.device)

        ### calculate loss for each task
        for idx, target_name in enumerate(self.config.args.tasks.targets):
            loss, loss_output, y_pred, y_true = self.calculateLoss(idx=idx, target_name=target_name, batch=batch, x_list=x_list, phase=phase)

            loss_list[idx] = loss
            loss_output_list[idx] = loss_output

            outputs[f"loss_{target_name}"] = loss
            outputs[f"y_pred_{target_name}"] = y_pred
            outputs[f"y_true_{target_name}"] = y_true

        loss_sum = torch.sum(loss_list)  # This weights is calculated with weights
        loss_output_sum = torch.sum(loss_output_list)  # This weights is not calculated with weights

        if phase == "val":
            outputs["loss"] = loss_output_sum

        else:
            outputs["loss"] = loss_sum

        outputs[f"{phase}_loss"] = loss_output_sum
        self.outputs_dict[phase].append(outputs)

        return outputs

    def backward(self, loss: torch.Tensor, *args: Any, **kwargs: Any) -> None:
        return super().backward(loss, *args, **kwargs)

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        return self.experiment_step(batch, batch_idx, phase="train")

    def on_train_epoch_end(self):
        metrics_dict = self.calcMetrics(phase="train")

        self.logging_dict["train"].append(
            {key: value.detach().cpu() if isinstance(value, torch.Tensor) else value for key, value in metrics_dict.items()}
        )

        return metrics_dict

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        return self.experiment_step(batch, batch_idx, phase="val")

    def on_validation_epoch_end(self):
        metrics_dict = self.calcMetrics(phase="val")

        if self.started_experiment:
            self.logging_dict["val"].append(
                {key: value.detach().cpu() if isinstance(value, torch.Tensor) else value for key, value in metrics_dict.items()}
            )

        return metrics_dict

    def test_step(self, batch, batch_idx) -> STEP_OUTPUT:
        return self.experiment_step(batch, batch_idx, phase="test")

    def on_test_epoch_end(self):
        metrics_dict = self.calcMetrics(phase="test")

        self.logging_dict["test"].append(
            {key: value.detach().cpu() if isinstance(value, torch.Tensor) else value for key, value in metrics_dict.items()}
        )

        return metrics_dict

    def calcMetrics(self, phase) -> dict:
        metrics_dict, df_true_pred = calculateMetrics(
            config=self.config, outputs_dict=self.outputs_dict, phase=phase, fold_test_num=self.fold_test_num, fold_val_num=self.fold_val_num
        )

        if not self.started_experiment:
            self.outputs_dict[phase].clear()
            return metrics_dict

        if phase == "train":
            self.train_loss_buffer[:, self.current_epoch] = np.array(
                [metrics_dict[f"{phase}_loss_{target_name}"].item() for target_name in self.config.args.tasks.targets]
            )

        self.results_dict[phase] = {"metrics_dict": metrics_dict, "df_true_pred": df_true_pred}

        for name, value in metrics_dict.items():
            self.log(name=name, value=value, prog_bar=True if "val_loss" == name else False)

        if phase == "test":
            output_attention_dict = {}
            attention_dict = {}

            for output in self.outputs_dict[phase]:
                if not "attention" in output:
                    break

                for branch_name, attention_block_dict in output["attention"].items():
                    output_attention_dict.setdefault(branch_name, {})

                    for block_num, attention in attention_block_dict.items():
                        output_attention_dict[branch_name].setdefault(block_num, [])
                        output_attention_dict[branch_name][block_num].append(attention.to("cpu"))

            for branch_name, attention_block_dict in output_attention_dict.items():
                for block_num, attention_list in attention_block_dict.items():
                    attention_dict.setdefault(branch_name, {})
                    attention_dict[branch_name][block_num] = torch.cat(attention_list)

            self.results_dict[phase]["attention"] = attention_dict

        self.outputs_dict[phase].clear()
        return metrics_dict

    def configure_optimizers(self) -> dict:
        # self.model
        config_dict = setOptimizerAndScheduler(self.config, self.trainer.model.model)

        return config_dict

    def lr_scheduler_step(self, scheduler, metric):
        if metric is None:
            scheduler.step(epoch=self.current_epoch, metrics=None)  # type: ignore

        else:
            scheduler.step(
                metrics=metric,  # type: ignore
                # epoch=self.current_epoch,
            )

    def train_dataloader(self):
        self.train_dataset = HealthDataset(config=self.config, fold_test_num=self.fold_test_num, fold_val_num=self.fold_val_num, phase="train")

        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.args.dataset.batch_size,
            num_workers=self.config.args.dataset.num_workers,
            drop_last=True,
            pin_memory=True,
            shuffle=True,
            persistent_workers=True,
        )

        return train_dataloader

    def val_dataloader(self):
        self.val_dataset = HealthDataset(config=self.config, fold_test_num=self.fold_test_num, fold_val_num=self.fold_val_num, phase="val")

        val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.config.args.dataset.batch_size,
            num_workers=self.config.args.dataset.num_workers,
            drop_last=False,
            pin_memory=True,
            shuffle=False,
            persistent_workers=True,
        )

        return val_dataloader

    def test_dataloader(self):
        self.test_dataset = HealthDataset(config=self.config, fold_test_num=self.fold_test_num, fold_val_num=self.fold_val_num, phase="test")

        test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.config.args.dataset.batch_size,
            num_workers=self.config.args.dataset.num_workers,
            drop_last=False,
            pin_memory=True,
            shuffle=False,
        )

        return test_dataloader
