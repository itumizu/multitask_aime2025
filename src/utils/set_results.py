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

from src.etc.constants import METRICS_FOR_CLASSIFICATION_LIST, METRICS_FOR_REGRESSION_LIST, NEURAL_NETWORK_MODELS
from src.utils.set_params import setParamsFromConfig
from src.utils.send_notion import (
    addResultDatabaseToNotion,
    addFoldResultToDatabase,
    addConfigToResultPage,
    addTextToResultPage,
    updateExperimentStatus,
    addDatabaseToPageInNotion,
    addImagesToPage,
)

from src.utils.summarize import uploadGraph


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

    if "is_merged_layers_by_tasks" in config.args.models.params:
        if config.args.models.params.is_merged_layers_by_tasks:
            config.args.tasks.targets = [config.args.tasks.targets[0]]

    for result_name, results_by_fold in results_dict.items():
        if result_name in ["fold_val_num"]:
            continue

        elif result_name in ["logging"]:
            logging_dict |= results_by_fold["metrics"]

            if "checkpoints" in results_by_fold:
                for ckpt_name, ckpt_path in results_by_fold["checkpoints"].items():
                    ckpt_path = Path(ckpt_path)
                    ckpt_path_list.append({"local_path": str(ckpt_path), "artifact_path": f"models/{ckpt_name}/{ckpt_path.name}"})

        elif result_name in ["cf_results"]:
            cf_results_val_fold_dict = results_by_fold

            for monitor_name, cf_dict in results_by_fold.items():
                output_cf_dir_path = Path(default_root_dir_test_fold, monitor_name)
                os.makedirs(output_cf_dir_path, exist_ok=True)

                with open(Path(output_cf_dir_path, "cf_results.json"), "w") as f:
                    json.dump(cf_dict, f)

                # mlflow.log_artifact(str(Path(output_cf_dir_path, "cf_results.json")), f"cf_results")

        # 予測値と正解値をまとめたDataFrame
        elif result_name in ["shap"]:
            shap_value_val_fold_dict = results_by_fold

            for monitor_name, shap_value_dict in results_by_fold.items():
                output_df_dir_path = Path(default_root_dir_test_fold, monitor_name)
                os.makedirs(output_df_dir_path, exist_ok=True)

                for task_name, task_shap_value_dict in shap_value_dict.items():
                    df_value: pd.DataFrame

                    for value_name, df_value in task_shap_value_dict.items():
                        output_df_path = Path(output_df_dir_path, f"{task_name}_{value_name}.csv")
                        df_value.to_csv(output_df_path)

                        # mlflow.log_artifact(str(output_df_path), f"shap/{monitor_name}")

                        if value_name != "shap_values":
                            continue

                        # 平均絶対SHAP値の棒グラフ
                        df_abs_value = df_value.abs()
                        df_abs_mean = df_abs_value.mean()
                        df_abs_mean_sorted: pd.Series = df_abs_mean.sort_values(ascending=True)
                        df_abs_mean_sorted.to_csv(Path(output_df_dir_path, f"{task_name}_{value_name}_abs_mean.csv"))
                        # mlflow.log_artifact(str(Path(output_df_dir_path, f"{task_name}_{value_name}_abs_mean.csv")), f"shap/{monitor_name}")

                        if is_plot_graph:
                            dataset = HealthDataset(config, fold_test_num=fold_test_num, fold_val_num=fold_val_num, phase="test")

                            file_path = plotBarhGraph(
                                data=df_abs_mean_sorted.values,
                                labels=[dataset.schema.loc[column, "name"] for column in df_abs_mean_sorted.index.tolist()],
                                x_label="Mean absolute SHAP value",
                                y_label="Attributes",
                                title=f"Mean absolute SHAP value: {task_name} task",
                                output_path=Path(output_df_dir_path),
                                file_name=f"shap_mean_abs_{task_name}_{monitor_name}",
                            )

                        # mlflow.log_artifact(str(file_path), f"shap/{monitor_name}")

                        # 入力属性とSHAP値の散布図
                        # for column in dataset.use_columns:
                        #     column_kana = str(dataset.schema.loc[column, "name_kana"])

                        #     file_path = plotScatterPlotGraph(
                        #         x=dataset.x.loc[:, column],
                        #         y=df_value.loc[:, column],
                        #         x_label=column_kana,
                        #         y_label="SHAP値",
                        #         title=f"{column_kana}とSHAP値の分布",
                        #         output_path=Path(output_df_dir_path),
                        #         file_name=f"shap_scatter_{monitor_name}_{column}",
                        #     )

                        #     mlflow.log_artifact(str(file_path), f"shap/{monitor_name}/{task_name}/scatter_plot")

        # 予測値と正解値をまとめたDataFrame
        elif result_name in ["df_true_pred"]:
            for metric_name, df_true_pred in results_by_fold.items():
                output_df_dir_path = Path(default_root_dir_test_fold, "df_true_pred")
                os.makedirs(output_df_dir_path, exist_ok=True)

                output_df_path = Path(default_root_dir_test_fold, f"{metric_name}.csv")
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

        # attention層
        elif result_name in ["attention"]:
            if is_skip_save:
                continue

            dataset = HealthDataset(config, fold_test_num=fold_test_num, fold_val_num=fold_val_num, phase="test")
            output_df_dir_path = Path(default_root_dir_test_fold, "attention")
            os.makedirs(output_df_dir_path, exist_ok=True)

            attention_dict = results_by_fold["attention"]

            for component_name, attention_block_dict in results_by_fold["attention"].items():
                attention_list: torch.Tensor

                for block_num, attention_list in attention_block_dict.items():
                    for attention_idx, attention in enumerate(attention_list.tolist()):
                        os.makedirs(Path(output_df_dir_path, component_name, str(block_num)), exist_ok=True)
                        file_path = Path(output_df_dir_path, component_name, str(block_num), f"attention_{attention_idx}.csv")

                        df_attention = pd.DataFrame(attention)
                        df_attention.to_csv(file_path)

                        file_label_path = Path(output_df_dir_path, component_name, str(block_num), f"attention_{attention_idx}_label.csv")
                        attention_array = np.array(attention)
                        attention_array_label = attention_array[:, 0, :]
                        df_attention_label = pd.DataFrame(attention_array_label)
                        df_attention_label.to_csv(file_label_path)

                        # mlflow.log_artifact(str(file_path), f"attention/{component_name}/{block_num}")

                        # numpy array にしてから保存したほうがよい？
                        # print(attention)

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

                    # attention を加算
                    attention_tensor = torch.mean(attention_list, dim=0)
                    attention_array = attention_tensor.numpy()

                    for head_num, attention in enumerate(attention_array):
                        file_path_csv = Path(default_root_dir_test_fold, f"attention_{component_name}_{block_num}_{head_num}.csv")
                        file_path_pdf = Path(default_root_dir_test_fold, f"attention_{component_name}_{block_num}_{head_num}.pdf")

                        df_attention = pd.DataFrame(attention)
                        df_attention.to_csv(file_path_csv)
                        mlflow.log_artifact(str(file_path_csv), f"attention/{component_name}/{block_num}")

                        if is_plot_graph:
                            y_tick_labels = ["label"] + dataset.use_columns

                            # ヒートマップ可視化
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
                            mlflow.log_artifact(str(file_path_pdf), f"attention_graphs/{component_name}/{block_num}")

                        # 総合
                    file_path_csv = Path(default_root_dir_test_fold, f"attention_{component_name}_{block_num}_summarize.csv")
                    file_path_pdf = Path(default_root_dir_test_fold, f"attention_{component_name}_{block_num}_summarize.pdf")

                    attention_summarize = torch.mean(attention_tensor, dim=0)
                    df_attention = pd.DataFrame(attention_summarize.numpy())
                    df_attention.to_csv(file_path_csv)
                    mlflow.log_artifact(str(file_path_csv), f"attention/{component_name}/{block_num}")

                    if is_plot_graph:
                        y_tick_labels = ["label"] + dataset.use_columns

                        # ヒートマップ可視化
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

                        mlflow.log_artifact(str(file_path_pdf), f"attention_graphs/{component_name}/{block_num}")

        elif isinstance(results_by_fold, dict):
            metrics_dict |= results_by_fold

        else:
            continue

    if is_best:
        try:
            # 学習曲線等のアップロードは要実装 (logging)
            response = addFoldResultToDatabase(
                database_id=database_id,
                fold_val_num=fold_val_num,
                results_dict=results_dict["test_results"] | results_dict["val_results"],
            )

        except:
            error_message = traceback.format_exc()
            print(error_message)

            response = updateExperimentStatus(
                page_id=config.args.notion_page_id,
                status="実行中(エラーあり)",
                message=error_message,
            )

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

    # グラフをアップロードする
    if is_best:
        # サンプルデータ: 各モデルのvalidation loss
        # グラフの設定
        loss_dict = {"train": {}, "val": {}, "test": {}}

        if is_plot_graph:
            for epoch, (phase, results_by_epoch) in enumerate(logging_dict.items()):
                for value_dict in results_by_epoch:
                    for phase_name in loss_dict.keys():
                        for metric_name, value in value_dict.items():
                            if phase_name in metric_name and "loss" in metric_name:
                                loss_dict[phase_name].setdefault(metric_name, [])

                                if isinstance(value, torch.Tensor):
                                    loss_dict[phase_name][metric_name].append(value.item())

                                else:
                                    loss_dict[phase_name][metric_name].append(value)
        if is_plot_graph:
            for phase, loss_phase_dict in loss_dict.items():
                plt.figure(figsize=(10, 6))

                # タイトルと軸ラベルの追加
                plt.title(f"{phase} Loss Over Epochs")
                plt.xlabel("Epochs")
                plt.ylabel(f"{phase} Loss")

                for loss_name, loss_value_list in loss_phase_dict.items():
                    plt.plot([epoch for epoch in range(len(loss_value_list))], loss_value_list, label=loss_name, marker="o")

                # 凡例の追加
                plt.legend()

                # グリッドの追加
                plt.grid(True)
                output_loss_path = Path(default_root_dir_test_fold, f"{phase}_loss_test_{fold_test_num}_val_{fold_val_num}.pdf")
                plt.savefig(output_loss_path, format="pdf", dpi=150)

                mlflow.log_artifact(
                    local_path=str(output_loss_path),
                    artifact_path=f"loss",
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

    # For test data
    if is_best:
        try:
            response = addConfigToResultPage(config=config, page_id=config.args.notion_page_id)

            if config.args.models.name in NEURAL_NETWORK_MODELS:
                response = addTextToResultPage(title="モデル", text=fold_results_dict[0]["model_structure"], page_id=config.args.notion_page_id)

            response = addResultDatabaseToNotion(
                page_id=config.args.notion_page_id,
                result_dict=fold_results_dict[0]["test_results"] | fold_results_dict[0]["val_results"],
                title="実験結果(val fold ごと)",
            )

            database_id = response["id"]

        except:
            error_message = traceback.format_exc()
            print(error_message)

            response = updateExperimentStatus(
                page_id=config.args.notion_page_id,
                status="実行中(エラーあり)",
                message=error_message,
            )

    results_list = []

    # 実験の高速化
    is_plot_graph = False  # more fast: False
    is_skip_save = True  # more fast: True

    if "is_skip_save" in config.args.advance_setting:
        is_skip_save = config.args.advance_setting.is_skip_save

    if "is_plot_graph" in config.args.advance_setting:
        is_plot_graph = config.args.advance_setting.is_plot_graph

    # 各Foldをそれぞれ画像化等を実施する
    for idx, (fold_val_num, results_dict) in enumerate(fold_results_dict.items()):
        # For Debug
        # results = setResultsOfValidationFold(
        #     config, results_dict, default_root_dir, fold_test_num, fold_val_num, database_id, is_best, is_plot_graph, is_skip_save
        # )
        # results_list.append(results)

        process = setResultsOfValidationFold.remote(
            config, results_dict, default_root_dir, fold_test_num, fold_val_num, database_id, is_best, is_plot_graph, is_skip_save
        )

        process_list.append(process)

    results_list = ray.get(process_list)
    ray.shutdown()

    artifacts_path_dict = {}

    for fold_val_num, results_dict in enumerate(results_list):
        args = {}
        args["nested"] = True

        if "run_id" in results_dict:
            args["run_id"] = results_dict["run_id"]

        else:
            args["run_name"] = f"fold_val_{fold_val_num}"

        default_root_dir_val_fold = Path(default_root_dir, f"fold_test_{fold_test_num}", f"fold_val_{fold_val_num}")

        with mlflow.start_run(**args) as run_fold:
            config_dict: Any = OmegaConf.to_container(config)

            mlflow.set_tag(key="fold_val", value=fold_val_num)
            mlflow.log_dict(config_dict, "params.yaml")

            setParamsFromConfig(config=config)
            mlflow.log_artifacts(str(default_root_dir_val_fold), "results")

            for phase_name, results_by_phase in results_dict["logging"].items():
                for epoch, results_by_epoch in enumerate(results_by_phase):
                    mlflow.log_metrics(metrics=results_by_epoch, step=epoch)

            mlflow.log_metrics(results_dict["metrics"])

            for ckpt_dict in results_dict["ckpt_path"]:
                mlflow.log_artifact(**ckpt_dict)

        artifacts_path_dict.setdefault(f"val_{fold_val_num}", {})

        if is_plot_graph:
            artifacts_path_dict[f"val_{fold_val_num}"] |= results_dict["graph_path"]

    # テストデータのとき
    if is_best:
        try:
            output_mean_value_dict = {}
            for metric_name, mean_value in mean_value_dict.items():
                for metric in METRICS_FOR_CLASSIFICATION_LIST + METRICS_FOR_REGRESSION_LIST:
                    for task_name in config.args.tasks.targets:
                        if f"test_{metric}_{task_name}" == metric_name:
                            output_mean_value_dict[metric_name] = mean_value
                            break

            response = addResultDatabaseToNotion(page_id=config.args.notion_page_id, result_dict=output_mean_value_dict, title="実験結果(平均)")
            response = addFoldResultToDatabase(database_id=response["id"], fold_val_num="平均", results_dict=output_mean_value_dict)

        except:
            error_message = traceback.format_exc()
            print(error_message)

            response = updateExperimentStatus(
                page_id=config.args.notion_page_id,
                status="実行中(エラーあり)",
                message=error_message,
            )

    if is_best:
        default_root_dir_test_fold = Path(default_root_dir, f"fold_test_{fold_test_num}")

        # タスクごとにSHAP値の可視化
        for task_name in config.args.tasks.targets:
            df_shap_dict = {}
            df_data_dict = {}
            attention_dict = {}

            if not is_plot_graph:
                continue

            # 結合
            for fold_val_num, results_dict in enumerate(results_list):
                # shap値が入ったdictを取り出す
                shap_value_val_fold_dict = results_dict["shap"]
                attention_val_fold_dict = results_dict["attention"]

                for monitor_name, shap_value_dict in shap_value_val_fold_dict.items():
                    df_shap_dict.setdefault(monitor_name, [])
                    df_data_dict.setdefault(monitor_name, [])

                    if not len(shap_value_dict.keys()):
                        break

                    for value_name, df_shap_value in shap_value_dict[task_name].items():
                        if value_name == "shap_values":
                            dataset = HealthDataset(config, fold_test_num=fold_test_num, fold_val_num=fold_val_num, phase="test")
                            df_data_dict[monitor_name].append(dataset.x)
                            df_shap_dict[monitor_name].append(df_shap_value)

                # Attentionの集計
                for component_name, attention_block_dict in attention_val_fold_dict.items():
                    attention_list: torch.Tensor
                    attention_dict.setdefault(component_name, {})

                    for block_num, attention_list in attention_block_dict.items():
                        attention_dict[component_name].setdefault(block_num, [])
                        attention = torch.sum(attention_list, dim=0)
                        attention_dict[component_name][block_num].append(attention)

            # SHAP値の可視化
            # print(df_shap_dict)

            for monitor_name in df_shap_dict.keys():
                if not monitor_name in df_data_dict:
                    continue

                if not df_shap_dict[monitor_name]:
                    continue

                df_shap_value = pd.concat(df_shap_dict[monitor_name], axis=0, ignore_index=True)

                df_abs = df_shap_value.abs()
                df_abs_mean = df_abs.mean()

                df_abs_mean_sorted: pd.Series = df_abs_mean.sort_values(ascending=True)
                df_abs_mean_sorted.to_csv(Path(default_root_dir_test_fold, f"{task_name}_{value_name}_abs_mean.csv"))

                mlflow.log_artifact(str(Path(default_root_dir_test_fold, f"{task_name}_{value_name}_abs_mean.csv")), f"shap/{monitor_name}")

                index_list = []
                for column in df_abs_mean_sorted.index.tolist():
                    index_list.append(dataset.schema.loc[column, "name_kana"])

                if is_plot_graph:
                    file_path = plotBarhGraph(
                        data=df_abs_mean_sorted.values,
                        labels=index_list,
                        x_label="平均絶対SHAP値",
                        y_label="属性",
                        title=f"5 Validation Fold での平均絶対SHAP値(タスク: {task_name})",
                        output_path=Path(default_root_dir_test_fold),
                        file_name=f"shap_mean_abs_all_folds_{task_name}_{monitor_name}",
                    )

                    mlflow.log_artifact(str(file_path), f"shap/{monitor_name}")

                # # 入力属性とSHAP値の散布図
                # for column in dataset.use_columns:
                #     column_kana = str(dataset.schema.loc[column, "name_kana"])
                #     print(column)

                #     file_path = plotScatterPlotGraph(
                #         x=data_x.loc[:, column],
                #         y=df_shap_value.loc[:, column],
                #         x_label=column_kana,
                #         y_label="SHAP値",
                #         title=f"{column_kana}とSHAP値の分布 (タスク: {task_name}, 5 val foldでまとめ)",
                #         output_path=Path(default_root_dir_test_fold),
                #         file_name=f"shap_scatter_all_folds_{task_name}_{monitor_name}_{column}",
                #     )

                #     mlflow.log_artifact(str(file_path), f"shap/{monitor_name}/scatter_plot/{task_name}")

            # attentionの可視化
            for component_name, attention_block_dict in attention_dict.items():
                attention_list: torch.Tensor
                dataset = HealthDataset(config, fold_test_num=fold_test_num, fold_val_num=fold_val_num, phase="test")

                for block_num, attention_list in attention_block_dict.items():
                    # attention を結合
                    attention = torch.stack(attention_dict[component_name][block_num])

                    # attention を加算
                    attention_array = torch.mean(attention, dim=0)

                    for head_num, attention_block in enumerate(attention_array.numpy()):
                        file_path_csv = Path(default_root_dir_test_fold, f"attention_{component_name}_{block_num}_{head_num}.csv")
                        file_path_pdf = Path(default_root_dir_test_fold, f"attention_{component_name}_{block_num}_{head_num}.pdf")

                        df_attention = pd.DataFrame(attention_block)
                        df_attention.to_csv(file_path_csv)
                        mlflow.log_artifact(str(file_path_csv), f"attention/{component_name}/{block_num}")

                        if is_plot_graph:
                            y_tick_labels = ["label"] + dataset.use_columns

                            # ヒートマップ可視化
                            fig = plt.figure(figsize=(10, 10), dpi=150)

                            sns.heatmap(
                                attention_block,
                                annot=False,
                                xticklabels=["label"] + dataset.use_columns,
                                yticklabels=y_tick_labels[0 : attention_block.shape[0]],
                                cmap="viridis",
                            )

                            plt.savefig(file_path_pdf)
                            plt.clf()
                            plt.close(fig)

                            mlflow.log_artifact(str(file_path_pdf), f"attention_graphs/{component_name}/{block_num}")

                    file_path_csv = Path(default_root_dir_test_fold, f"attention_{component_name}_{block_num}_summarize.csv")
                    file_path_pdf = Path(default_root_dir_test_fold, f"attention_{component_name}_{block_num}_summarize.pdf")

                    attention_summarize = torch.mean(attention_array, dim=0)
                    attention_summarize = attention_summarize.numpy()

                    df_attention = pd.DataFrame(attention_summarize)
                    df_attention.to_csv(file_path_csv)
                    mlflow.log_artifact(str(file_path_csv), f"attention/{component_name}/{block_num}")

                    if is_plot_graph:
                        y_tick_labels = ["label"] + dataset.use_columns

                        # ヒートマップ可視化
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
                        mlflow.log_artifact(str(file_path_pdf), f"attention_graphs/{component_name}/{block_num}")

    if is_best:
        default_root_dir = Path(default_root_dir)
        url_merge_dict = {}
        folder_id_dict = {}

        if is_plot_graph:
            for artifact_dict_name, artifact_dict in artifacts_path_dict.items():
                # for artifact_name, artifacts_path in artifact_dict.items():
                url_dict, folder_id = uploadGraph(
                    Path(default_root_dir.parent.stem, default_root_dir.stem, artifact_dict_name), list(artifact_dict.values())
                )

                url_merge_dict |= url_dict
                folder_id_dict[artifact_dict_name] = f"https://drive.google.com/drive/folders/{folder_id}"

            properties = {f"val fold": {"title": {}}, "フォルダURL": {"url": {}}}

            # 画像ファイルをアップロードする設定
            response = addDatabaseToPageInNotion(page_id=config.args.notion_page_id, title="画像ファイル", properties=properties)
            addImagesToPage(url_dict=folder_id_dict, database_id=response["id"], fold_name="val")
