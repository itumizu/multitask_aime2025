import os
import json
import traceback
from pathlib import Path
from typing import Any, List

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import ray
import mlflow
from omegaconf import ListConfig, OmegaConf, DictConfig

from src.loaders.dataset import HealthDataset
from src.etc.constants import (
    METRICS_FOR_CLASSIFICATION_LIST,
    METRICS_FOR_REGRESSION_LIST,
    NEURAL_NETWORK_MODELS,
)
from src.utils.set_params import setParamsFromConfig


def plotBarhGraph(
    data: np.ndarray | pd.Series | List | Any,
    labels: np.ndarray | List,
    x_label: str,
    y_label: str,
    title: str,
    output_path: Path,
    file_name: str,
    file_format: str = "pdf",
):
    fig = plt.figure(figsize=(15, 15), dpi=150)
    plt.barh(y=labels, width=data)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    output_path = Path(output_path, f"{file_name}.{file_format}")
    plt.tight_layout()
    plt.savefig(output_path, format=file_format)
    plt.clf()
    plt.close(fig)

    return output_path


def plotScatterPlotGraph(
    x: np.ndarray | pd.Series | List | Any,
    y: np.ndarray | pd.Series | List | Any,
    x_label: str,
    y_label: str,
    title: str,
    output_path: Path,
    file_name: str,
    file_format: str = "pdf",
):
    fig = plt.figure(figsize=(8, 8), dpi=150)
    # plt.scatter(x=x, y=y, alpha=0.5)
    sns.jointplot(x=x, y=y)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    output_path = Path(output_path, f"{file_name}.{file_format}")
    plt.tight_layout()
    plt.savefig(output_path, format=file_format)
    plt.clf()
    plt.close(fig)

    return output_path


@ray.remote(num_cpus=12)
def setResultsOfValidationFold(
    config: DictConfig | ListConfig,
    results_dict: dict,
    default_root_dir: Path,
    fold_test_num: int,
    fold_val_num: int,
    database_id: str,
    is_best: bool,
    is_plot_graph: bool,
    is_skip_save: bool,
):
    metrics_dict = {}
    logging_dict = {}
    attention_dict = {}
    shap_value_val_fold_dict = {}
    cf_results_val_fold_dict = {}
    ckpt_path_list = []

    default_root_dir_test_fold = Path(
        default_root_dir,
        f"fold_test_{fold_test_num}",
        f"fold_val_{fold_val_num}",
    )
    os.makedirs(default_root_dir_test_fold, exist_ok=True)

    if config.args.models.params.get("is_merged_layers_by_tasks", False):
        config.args.tasks.targets = [config.args.tasks.targets[0]]

    for result_name, results_by_fold in results_dict.items():
        if result_name == "fold_val_num":
            continue

        elif result_name == "logging":
            logging_dict |= results_by_fold["metrics"]

            if "checkpoints" in results_by_fold:
                for ckpt_name, ckpt_path in results_by_fold["checkpoints"].items():
                    ckpt_path = Path(ckpt_path)
                    ckpt_path_list.append(
                        {
                            "local_path": str(ckpt_path),
                            "artifact_path": f"models/{ckpt_name}/{ckpt_path.name}",
                        }
                    )

        elif result_name == "cf_results":
            cf_results_val_fold_dict = results_by_fold
            for monitor_name, cf_dict in results_by_fold.items():
                output_cf_dir_path = Path(default_root_dir_test_fold, monitor_name)
                os.makedirs(output_cf_dir_path, exist_ok=True)
                with open(Path(output_cf_dir_path, "cf_results.json"), "w") as f:
                    json.dump(cf_dict, f)
                # mlflow.log_artifact(str(Path(output_cf_dir_path, "cf_results.json")), f"cf_results")

        # DataFrame of true and predicted values
        elif result_name == "shap":
            shap_value_val_fold_dict = results_by_fold
            for monitor_name, shap_value_dict in results_by_fold.items():
                output_df_dir_path = Path(default_root_dir_test_fold, monitor_name)
                os.makedirs(output_df_dir_path, exist_ok=True)

                for task_name, task_shap_value_dict in shap_value_dict.items():
                    for value_name, df_value in task_shap_value_dict.items():
                        output_df_path = Path(output_df_dir_path, f"{task_name}_{value_name}.csv")
                        df_value.to_csv(output_df_path)
                        # mlflow.log_artifact(str(output_df_path), f"shap/{monitor_name}")

                        if value_name != "shap_values":
                            continue

                        # Bar graph of mean absolute SHAP values
                        df_abs_mean_sorted: pd.Series = df_value.abs().mean().sort_values(ascending=True)
                        df_abs_mean_sorted.to_csv(
                            Path(output_df_dir_path, f"{task_name}_{value_name}_abs_mean.csv")
                        )
                        # mlflow.log_artifact(str(Path(output_df_dir_path, f"{task_name}_{value_name}_abs_mean.csv")), f"shap/{monitor_name}")

                        if is_plot_graph:
                            dataset = HealthDataset(
                                config,
                                fold_test_num=fold_test_num,
                                fold_val_num=fold_val_num,
                                phase="test",
                            )
                            file_path = plotBarhGraph(
                                data=df_abs_mean_sorted.values,
                                labels=[
                                    dataset.schema.loc[col, "name"]
                                    for col in df_abs_mean_sorted.index.tolist()
                                ],
                                x_label="Mean absolute SHAP value",
                                y_label="Attributes",
                                title=f"Mean absolute SHAP value: {task_name} task",
                                output_path=Path(output_df_dir_path),
                                file_name=f"shap_mean_abs_{task_name}_{monitor_name}",
                            )
                            # mlflow.log_artifact(str(file_path), f"shap/{monitor_name}")

                        # Scatter plot of input attribute and SHAP values
                        # for column in dataset.use_columns:
                        #     column_kana = str(dataset.schema.loc[column, "name_kana"])
                        #     file_path = plotScatterPlotGraph(
                        #         x=dataset.x.loc[:, column],
                        #         y=df_value.loc[:, column],
                        #         x_label=column_kana,
                        #         y_label="SHAP value",
                        #         title=f"Distribution of {column_kana} and SHAP value",
                        #         output_path=Path(output_df_dir_path),
                        #         file_name=f"shap_scatter_{monitor_name}_{column}",
                        #     )
                        #     mlflow.log_artifact(str(file_path), f"shap/{monitor_name}/{task_name}/scatter_plot")

        # DataFrame of true and predicted values
        elif result_name == "df_true_pred":
            for metric_name, df_true_pred in results_by_fold.items():
                output_df_dir_path = Path(default_root_dir_test_fold, "df_true_pred")
                os.makedirs(output_df_dir_path, exist_ok=True)

                output_df_path = Path(output_df_dir_path, f"{metric_name}.csv")
                df_true_pred.to_csv(output_df_path)

                if is_plot_graph:
                    for task_name in config.args.tasks.targets:
                        file_path = plotScatterPlotGraph(
                            x=df_true_pred.loc[:, f"y_pred_{task_name}"],
                            y=df_true_pred.loc[:, f"y_true_{task_name}"],
                            x_label="Predicted value",
                            y_label="True value",
                            title=f"Scatter plot of {task_name} task",
                            output_path=Path(output_df_dir_path),
                            file_name=f"y_pred_true_{task_name}_scatter_{metric_name}",
                        )

        elif result_name == "attention":
            if is_skip_save:
                continue

            dataset = HealthDataset(
                config, fold_test_num=fold_test_num, fold_val_num=fold_val_num, phase="test"
            )
            output_df_dir_path = Path(default_root_dir_test_fold, "attention")
            os.makedirs(output_df_dir_path, exist_ok=True)

            attention_dict = results_by_fold["attention"]
            for component_name, attention_block_dict in attention_dict.items():
                for block_num, attention_list in attention_block_dict.items():
                    for att_idx, attention in enumerate(attention_list.tolist()):
                        os.makedirs(
                            Path(output_df_dir_path, component_name, str(block_num)),
                            exist_ok=True,
                        )
                        file_path = Path(
                            output_df_dir_path,
                            component_name,
                            str(block_num),
                            f"attention_{att_idx}.csv",
                        )
                        pd.DataFrame(attention).to_csv(file_path)
                        # mlflow.log_artifact(str(file_path), f"attention/{component_name}/{block_num}")

                        # Should we save after converting to numpy array?
                        # plt.figure(figsize=(10, 10), dpi=150)
                        # sns.heatmap(
                        #     attention,
                        #     annot=False,
                        #     xticklabels=["label"] + dataset.use_columns,
                        #     yticklabels=["label"] + dataset.use_columns,
                        #     cmap="viridis",
                        # )
                        # file_path = Path(output_df_dir_path, f"attention_{component_name}_{block_num}_{idx}.pdf")
                        # plt.savefig(file_path)
                        # mlflow.log_artifact(str(file_path), f"attention/{monitor_name}")

                    attention_tensor = torch.mean(attention_list, dim=0)
                    attention_array = attention_tensor.numpy()

                    for head_num, attention in enumerate(attention_array):
                        file_path_csv = Path(
                            default_root_dir_test_fold,
                            f"attention_{component_name}_{block_num}_{head_num}.csv",
                        )
                        file_path_pdf = Path(
                            default_root_dir_test_fold,
                            f"attention_{component_name}_{block_num}_{head_num}.pdf",
                        )

                        pd.DataFrame(attention).to_csv(file_path_csv)
                        mlflow.log_artifact(
                            str(file_path_csv), f"attention/{component_name}/{block_num}"
                        )

                        if is_plot_graph:
                            y_tick_labels = ["label"] + dataset.use_columns

                            # Heatmap visualization
                            fig = plt.figure(figsize=(10, 10), dpi=150)
                            sns.heatmap(
                                attention,
                                annot=False,
                                xticklabels=["label"] + dataset.use_columns,
                                yticklabels=y_tick_labels[0 : attention.shape[0]],
                                cmap="viridis",
                            )
                            plt.savefig(file_path_pdf)
                            plt.clf()
                            plt.close(fig)
                            mlflow.log_artifact(
                                str(file_path_pdf),
                                f"attention_graphs/{component_name}/{block_num}",
                            )

                    file_path_csv = Path(
                        default_root_dir_test_fold,
                        f"attention_{component_name}_{block_num}_summarize.csv",
                    )
                    file_path_pdf = Path(
                        default_root_dir_test_fold,
                        f"attention_{component_name}_{block_num}_summarize.pdf",
                    )

                    attention_summarize = torch.mean(attention_tensor, dim=0)
                    pd.DataFrame(attention_summarize.numpy()).to_csv(file_path_csv)
                    mlflow.log_artifact(
                        str(file_path_csv), f"attention/{component_name}/{block_num}"
                    )

                    if is_plot_graph:
                        y_tick_labels = ["label"] + dataset.use_columns

                        # Heatmap visualization
                        fig = plt.figure(figsize=(10, 10), dpi=150)
                        sns.heatmap(
                            attention_summarize,
                            annot=False,
                            xticklabels=["label"] + dataset.use_columns,
                            yticklabels=y_tick_labels[0 : attention_summarize.shape[0]],
                            cmap="viridis",
                        )
                        plt.savefig(file_path_pdf)
                        plt.clf()
                        plt.close(fig)
                        mlflow.log_artifact(
                            str(file_path_pdf),
                            f"attention_graphs/{component_name}/{block_num}",
                        )

        elif isinstance(results_by_fold, dict):
            metrics_dict |= results_by_fold

        else:
            continue

    if is_best:
        try:
            pass
        except Exception:
            error_message = traceback.format_exc()
            print(error_message)

    output_dict = {
        "metrics": metrics_dict,
        "logging": logging_dict,
        "ckpt_path": ckpt_path_list,
    }

    if is_best:
        output_dict |= {
            "shap": shap_value_val_fold_dict,
            "attention": attention_dict,
            "cf_results": cf_results_val_fold_dict,
        }

    if is_best:
        # Sample data: validation loss per model
        # Graph settings
        loss_dict = {"train": {}, "val": {}, "test": {}}

        if is_plot_graph:
            for phase, loss_phase_dict in loss_dict.items():
                plt.figure(figsize=(10, 6))

                # Title and axis labels
                plt.title(f"{phase} Loss Over Epochs")
                plt.xlabel("Epoch")
                plt.ylabel(f"{phase} Loss")

                for loss_name, loss_value_list in loss_phase_dict.items():
                    plt.plot(
                        [epoch for epoch in range(len(loss_value_list))],
                        loss_value_list,
                        label=loss_name,
                        marker="o",
                    )

                # Add legend
                plt.legend()

                # Add grid
                plt.grid(True)
                output_loss_path = Path(
                    default_root_dir_test_fold,
                    f"{phase}_loss_test_{fold_test_num}_val_{fold_val_num}.pdf",
                )
                plt.savefig(output_loss_path, format="pdf", dpi=150)

                mlflow.log_artifact(
                    local_path=str(output_loss_path),
                    artifact_path="loss",
                )

                plt.close()

                output_dict.setdefault("graph_path", {})
                output_dict["graph_path"][f"{phase}_loss"] = output_loss_path

    return output_dict


def setResultsOfEachFold(
    config: DictConfig | ListConfig,
    fold_results_dict: dict,
    mean_value_dict: dict,
    default_root_dir: Path,
    fold_test_num: int,
    is_best: bool = False,
):
    database_id = ""
    process_list = []

    # Speeding up the experiment
    is_plot_graph = config.args.advance_setting.get("is_plot_graph", False)
    is_skip_save = config.args.advance_setting.get("is_skip_save", True)

    for fold_val_num, results_dict in fold_results_dict.items():
        process = setResultsOfValidationFold.remote(
            config,
            results_dict,
            default_root_dir,
            fold_test_num,
            fold_val_num,
            database_id,
            is_best,
            is_plot_graph,
            is_skip_save,
        )
        process_list.append(process)

    results_list = ray.get(process_list)
    ray.shutdown()

    for fold_val_num, results_dict in enumerate(results_list):
        args = {"nested": True}
        args["run_name"] = f"fold_val_{fold_val_num}"

        default_root_dir_val_fold = Path(
            default_root_dir,
            f"fold_test_{fold_test_num}",
            f"fold_val_{fold_val_num}",
        )

        with mlflow.start_run(**args):
            mlflow.set_tag(key="fold_val", value=fold_val_num)
            mlflow.log_dict(OmegaConf.to_container(config), "params.yaml")

            setParamsFromConfig(config=config)
            mlflow.log_artifacts(str(default_root_dir_val_fold), "results")

            for phase_name, results_by_phase in results_dict["logging"].items():
                for epoch, results_by_epoch in enumerate(results_by_phase):
                    mlflow.log_metrics(metrics=results_by_epoch, step=epoch)

            mlflow.log_metrics(results_dict["metrics"])

            for ckpt_dict in results_dict["ckpt_path"]:
                mlflow.log_artifact(**ckpt_dict)

    if is_best:
        output_mean_value_dict = {}
        for metric_name, mean_value in mean_value_dict.items():
            for metric in METRICS_FOR_CLASSIFICATION_LIST + METRICS_FOR_REGRESSION_LIST:
                for task_name in config.args.tasks.targets:
                    if f"test_{metric}_{task_name}" == metric_name:
                        output_mean_value_dict[metric_name] = mean_value
                        break

        if output_mean_value_dict:
            tmp_path = Path(
                default_root_dir, f"fold_test_{fold_test_num}", "mean_metrics.json"
            )
            with open(tmp_path, "w") as f:
                json.dump(output_mean_value_dict, f, indent=2)
            mlflow.log_artifact(str(tmp_path), artifact_path="results")
