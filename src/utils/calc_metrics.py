import re

from typing import Any, Tuple
import torch
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    roc_auc_score,
    matthews_corrcoef,
    precision_score,
    confusion_matrix,
    classification_report,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from omegaconf import DictConfig, ListConfig
from ..etc.constants import (
    BINARY_CLASSIFICATION_LOSS_FUNCTION_LIST,
    MULTICLASS_CLASSIFICATION_LOSS_FUNCTION_LIST,
    REGRESSION_LOSS_FUNCTION_LIST,
    NEURAL_NETWORK_MODELS,
)


def calculateEachTarget(
    config: DictConfig | ListConfig,
    target_name: str,
    loss: Any,
    y_pred: np.ndarray,
    y_true: np.ndarray,
    y_pred_label: np.ndarray,
    y_pred_true_array: np.ndarray,
    participant_id_list,
    phase: str,
    replace_target_name="",
):
    task_type = ""
    if config.args.models.name in NEURAL_NETWORK_MODELS:
        if config.args.models.loss_functions[target_name] in BINARY_CLASSIFICATION_LOSS_FUNCTION_LIST + MULTICLASS_CLASSIFICATION_LOSS_FUNCTION_LIST:
            task_type = "classification"

        elif config.args.models.loss_functions[target_name] in REGRESSION_LOSS_FUNCTION_LIST:
            task_type = "regression"

        else:
            raise

    else:
        if "Regressor" in config.args.models.name:
            task_type = "regression"

        elif "Classifier" in config.args.models.name:

            task_type = "classification"
        else:
            raise

    if replace_target_name:
        target_name = replace_target_name

    metrics_dict_method = {}

    if task_type == "classification":
        metrics_dict_method = calculateMetricsAsClassification(
            target_name=target_name, loss=loss, y_pred=y_pred, y_pred_label=y_pred_label, y_true=y_true, phase=phase
        )

    elif task_type == "regression":
        metrics_dict_method = calculateMetricsAsRegression(
            target_name=target_name, loss=loss, y_pred=y_pred, y_pred_label=y_pred_label, y_true=y_true, phase=phase
        )

    df_true_pred_each_target = pd.DataFrame(
        y_pred_true_array,
        columns=[f"y_pred_{target_name}", f"y_true_{target_name}"],
        index=participant_id_list,
    )

    return metrics_dict_method, df_true_pred_each_target


def aggregateResultsEachTarget(target_name: str, config: ListConfig | DictConfig, output_dict: dict):
    if config.args.models.name in NEURAL_NETWORK_MODELS:
        y_pred = torch.cat([output[f"y_pred_{target_name}"] for output in output_dict])
        y_true = torch.cat([output[f"y_true_{target_name}"] for output in output_dict])
        loss = torch.stack([output[f"loss_{target_name}"] for output in output_dict])

        participant_id_list = []

        for output in output_dict:
            participant_id_list.extend(output[f"participant_id_list"])

        y_pred = y_pred.detach().cpu().float().numpy().copy()
        y_true = y_true.detach().cpu().float().numpy().copy()

        # participant_id_list = participant_id_list.detach().cpu().numpy().copy()
        y_pred_true_array = np.hstack((y_pred, y_true))

    else:
        y_pred = output_dict[f"y_pred_{target_name}"]
        y_true = output_dict[f"y_true_{target_name}"]

        loss = output_dict[f"loss_{target_name}"]
        loss = np.array(loss)

        y_pred_true_array = np.column_stack((y_pred, y_true))

        participant_id_list = output_dict["participant_id_list"]

    y_pred_label = np.round(y_pred)

    return loss, y_pred, y_true, y_pred_label, y_pred_true_array, participant_id_list


def calculateMetrics(
    config: DictConfig | ListConfig, outputs_dict: dict, phase: str, fold_test_num=None, fold_val_num=None
) -> Tuple[dict, pd.DataFrame]:
    metrics_dict = {}
    results_dict = {}

    df_true_pred = pd.DataFrame()
    participant_id_list = []

    output_dict = outputs_dict[phase]
    merged_dict = {}

    for target_name in config.args.tasks.targets:

        # 集計
        loss, y_pred, y_true, y_pred_label, y_pred_true_array, participant_id_list = aggregateResultsEachTarget(
            target_name=target_name, config=config, output_dict=output_dict
        )

        if config.args.dataset.use_standardization_label:
            from pathlib import Path
            import pickle
            from omegaconf import OmegaConf

            scaler_path = Path(config.args.dataset.data_dir, f"fold_test_{fold_test_num}", f"fold_val_{fold_val_num}", "scaler_label.pkl")
            config_dataset = OmegaConf.load(Path(config.args.dataset.data_dir, "config.yaml"))
            label_columns = []

            temp = config_dataset.args.target_columns

            for target in config_dataset.args.target_columns:
                if not target.name in label_columns:
                    if target.task_type != "classification":
                        label_columns.append(target.name)

            labels_true = pd.read_csv(
                Path(config.args.dataset.data_dir, f"fold_test_{fold_test_num}", f"fold_val_{fold_val_num}", phase, f"label_{phase}.csv"), index_col=0
            )

            with open(scaler_path, "rb") as f:
                scaler_load = pickle.load(f)
                mean_scaler, scale_scaler = scaler_load.mean_, scaler_load.scale_

                mean_scaler = pd.DataFrame([mean_scaler], columns=label_columns)
                scale_scaler = pd.DataFrame([scale_scaler], columns=label_columns)

            y_pred = y_pred * scale_scaler.loc[0, target_name] + mean_scaler.loc[0, target_name]
            # y_true = y_true * scale_scaler.loc[0, target_name] + mean_scaler.loc[0, target_name]
            y_true = labels_true.loc[participant_id_list, target_name].values

            # print(y_pred)
            # print("y_true by scaler:", y_true)
            # print("y_true:", labels_true.loc[participant_id_list, target_name])

        # 評価指標の計算
        metrics_dict_method, df_true_pred_each_target = calculateEachTarget(
            config=config,
            target_name=target_name,
            loss=loss,
            y_pred=y_pred,
            y_true=y_true,
            y_pred_label=y_pred_label,
            y_pred_true_array=y_pred_true_array,
            participant_id_list=participant_id_list,
            phase=phase,
        )

        metrics_dict.update(metrics_dict_method)
        df_true_pred = pd.concat([df_true_pred, df_true_pred_each_target], axis=1)

        results_dict[target_name] = {
            "loss": loss,
            "y_pred": y_pred,
            "y_true": y_true,
            "y_pred_label": y_pred_label,
            "y_pred_true_array": y_pred_true_array,
            "participant_id_list": participant_id_list,
        }

        if "SAME" in target_name:
            name = re.sub("_SAME.*", "", target_name)
            merged_dict.setdefault(name, [])
            merged_dict[name].append(y_pred)

    if len(merged_dict.keys()) > 0:
        y_pred_all_list = [results_dict[name]["y_pred"]]

        for name, y_pred_list in merged_dict.items():
            y_pred_all_list.extend(y_pred_list)

        y_pred_merged = np.stack(y_pred_all_list)
        y_true = results_dict[name]["y_true"]

        # mean
        y_pred_merged = np.mean(y_pred_merged, axis=0)
        y_pred_merged_label = np.round(y_pred_merged)
        y_pred_merged_true_array = np.column_stack((y_pred_merged, y_true))

        metrics_dict_method, df_true_pred_each_target = calculateEachTarget(
            config=config,
            target_name=name,
            loss=loss,
            y_pred=y_pred_merged,
            y_true=y_true,
            y_pred_label=y_pred_merged_label,
            y_pred_true_array=y_pred_merged_true_array,
            participant_id_list=participant_id_list,
            phase=phase,
            replace_target_name=name + "_MERGED",
        )

        metrics_dict.update(metrics_dict_method)
        df_true_pred = pd.concat([df_true_pred, df_true_pred_each_target], axis=1)

        # median
        y_pred_merged = np.stack(y_pred_all_list)
        y_pred_merged = np.median(y_pred_merged, axis=0)
        y_pred_merged_label = np.round(y_pred_merged)
        y_pred_merged_true_array = np.column_stack((y_pred_merged, y_true))

        metrics_dict_method, df_true_pred_each_target = calculateEachTarget(
            config=config,
            target_name=name,
            loss=loss,
            y_pred=y_pred_merged,
            y_true=y_true,
            y_pred_label=y_pred_merged_label,
            y_pred_true_array=y_pred_merged_true_array,
            participant_id_list=participant_id_list,
            phase=phase,
            replace_target_name=name + "_MERGED_MEDIAN",
        )

        metrics_dict.update(metrics_dict_method)
        df_true_pred = pd.concat([df_true_pred, df_true_pred_each_target], axis=1)

    if config.args.models.name in NEURAL_NETWORK_MODELS:
        loss_multi = torch.stack([output["loss"] for output in output_dict])
        metrics_dict[f"{phase}_loss"] = loss_multi.mean()

    return metrics_dict, df_true_pred


def calculateMetricsAsClassification(target_name, loss, y_pred, y_pred_label, y_true, phase: str) -> dict:
    acc = accuracy_score(y_pred=y_pred_label, y_true=y_true)
    recall = recall_score(y_pred=y_pred_label, y_true=y_true)
    mcc = matthews_corrcoef(y_pred=y_pred_label, y_true=y_true)
    precision = precision_score(y_pred=y_pred_label, y_true=y_true)
    matrix = confusion_matrix(y_pred=y_pred_label, y_true=y_true)

    try:
        tn, fp, fn, tp = matrix.flatten()

    except:
        tn, fp, fn, tp = -1, -1, -1, -1

    if phase == "val":
        print(classification_report(y_pred=y_pred_label, y_true=y_true))

    try:
        auroc = roc_auc_score(y_score=y_pred, y_true=y_true)

    except:
        auroc = -1

    metrics_dict_method = {
        f"{phase}_acc_{target_name}": acc,
        f"{phase}_recall_{target_name}": recall,
        f"{phase}_auroc_{target_name}": auroc,
        f"{phase}_mcc_{target_name}": mcc,
        f"{phase}_precision_{target_name}": precision,
        f"{phase}_loss_{target_name}": loss.mean(),
        f"{phase}_tn_{target_name}": tn,
        f"{phase}_fp_{target_name}": fp,
        f"{phase}_fn_{target_name}": fn,
        f"{phase}_tp_{target_name}": tp,
    }

    return metrics_dict_method


def calculateMetricsAsRegression(target_name, loss, y_pred, y_pred_label, y_true, phase: str) -> dict:
    mse_score = mean_squared_error(y_true=y_true, y_pred=y_pred)
    mae_score = mean_absolute_error(y_true=y_true, y_pred=y_pred)
    r2_score_value = r2_score(y_true=y_true, y_pred=y_pred)
    rmse_score = np.sqrt(mse_score)

    metrics_dict_method = {
        f"{phase}_rmse_{target_name}": rmse_score.item(),
        f"{phase}_mae_{target_name}": mae_score.item(),
        f"{phase}_r2_score_{target_name}": r2_score_value,
        f"{phase}_mse_{target_name}": mse_score.item(),
        f"{phase}_loss_{target_name}": loss.mean(),
    }

    return metrics_dict_method
