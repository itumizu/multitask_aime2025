import os
from datetime import datetime
from typing import Any
from pathlib import Path

import ray
import torch
import shap
import numpy as np
import pandas as pd

import mlflow
from mlflow import MlflowClient
from omegaconf import ListConfig, OmegaConf, DictConfig
from src.loaders.dataset import HealthDataset
from src.etc.constants import (
    GRADIENT_BOOSTING_TREE_CLASSIFICATION_MODELS,
    NEURAL_NETWORK_MODELS,
    GRADIENT_BOOSTING_TREE_MODELS,
    TRANSFORMER_BASED_MODELS,
    CATBOOST_MODELS,
    LIGHTGBM_MODELS,
)


def calculateSHAPValuesForGradientBoostingTrees(
    config: DictConfig | ListConfig,
    model,
    fold_test_num: int,
    fold_val_num: int,
    device,
):
    current_date = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    print(f"start: {current_date}")

    model_output = "raw"

    if config.args.models.name in GRADIENT_BOOSTING_TREE_CLASSIFICATION_MODELS:
        model_output = "model.predict_proba"

    feature_perturbation = "tree_path_dependent"
    explainer = shap.TreeExplainer(
        model,
        data=None,
        model_output=model_output,
        feature_perturbation=feature_perturbation,
    )

    args_explainer = {}

    test_dataset = HealthDataset(
        config=config,
        fold_test_num=fold_test_num,
        fold_val_num=fold_val_num,
        phase="test",
    )
    test_x = test_dataset.x.loc[:, test_dataset.use_columns]

    if config.args.models.name in CATBOOST_MODELS:
        for column in test_dataset.categorical_columns:
            test_x[column] = test_x[column].astype(int)

    if "shap" in config.args.advance_setting:
        args_explainer = OmegaConf.to_container(config.args.advance_setting.shap)

    if config.args.models.name in CATBOOST_MODELS:
        shap_values = explainer.shap_values(test_x)
    else:
        shap_values = explainer.shap_values(test_x.values)

    shap_values_dict = {"expected_value": pd.DataFrame([explainer.expected_value])}

    if len(config.args.tasks.targets) > 2:
        for idx, task_name in enumerate(config.args.tasks.targets):
            shap_values_dict[task_name] = {
                "shap_values": pd.DataFrame(
                    data=shap_values[:, idx],
                    index=test_x.index,
                    columns=test_dataset.use_columns,
                ),
                "expected_value": pd.DataFrame([explainer.expected_value]),
            }

    else:
        for idx, task_name in enumerate(config.args.tasks.targets):
            shap_values_dict[task_name] = {
                "shap_values": pd.DataFrame(
                    data=shap_values,
                    index=test_x.index,
                    columns=test_dataset.use_columns,
                ),
                "expected_value": pd.DataFrame([explainer.expected_value]),
            }

    current_date = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    print(f"current_date: {current_date}")

    return shap_values_dict


def readCalucalatedResults(config: DictConfig | ListConfig, fold_test_num: int, fold_val_num: int):
    """Read pre-calculated SHAP results that have been logged to MLflow."""
    shap_values_dict = {}
    ulid = config.args.advance_setting.ulid_shap_calculated_results
    mlflow.set_tracking_uri(f'http://{os.environ["CONTAINER_NAME"] + "_mlflow_server:5000"}')

    # Retrieve MLflow runs
    client = MlflowClient()
    tracking_uri = mlflow.get_tracking_uri()
    print(f"tracking_uri: {tracking_uri}")

    experiment = client.get_experiment_by_name(config.args.experiment_name)
    experiment_id = experiment.experiment_id

    query = 'tags.ulid = "' + ulid + '"'

    experiment = client.get_experiment_by_name(config.args.experiment_name)

    if not experiment:
        raise ValueError(f"{config.args.experiment} was not found")

    experiment_id = experiment.experiment_id
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=query,
        order_by=["Created DESC"],
    )

    for run in runs:
        if "fold_test" in run.data.tags:
            if run.data.tags["fold_test"] == fold_test_num:
                break

    query_child_runs = (
        f'tags.mlflow.parentRunId = "{run.info.run_id}" and '
        f'attributes.status = "FINISHED" and tags.fold_val = "{fold_val_num}"'
    )
    run_child_list = client.search_runs(
        experiment_ids=[experiment_id], filter_string=query_child_runs
    )

    if not len(run_child_list) == 1:
        raise ValueError(
            f"{run.info.run_id} is not found about validation fold results."
        )

    run = run_child_list[0]
    artifacts_dir = Path(str(run.info.artifact_uri))

    for idx, task_name in enumerate(config.args.tasks.targets):
        file_dir = Path(artifacts_dir, "results", "shap")

        shap_values_dict[task_name] = {
            "shap_values": pd.read_csv(
                Path(file_dir, f"{task_name}_shap_values.csv"), index_col=0
            ),
            "base_value": pd.read_csv(
                Path(file_dir, f"{task_name}_base_value.csv"), index_col=0
            ),
        }

    return shap_values_dict


def calculateSHAPValues(
    config: DictConfig | ListConfig,
    model,
    fold_test_num: int,
    fold_val_num: int,
    device,
):
    if "is_calculating_shap" in config.args.advance_setting:
        if not config.args.advance_setting.is_calculating_shap:
            return {}
    else:
        return {}

    if "ulid_shap_calculated_results" in config.args.advance_setting:
        return readCalucalatedResults(
            config=config, fold_test_num=fold_test_num, fold_val_num=fold_val_num
        )

    if config.args.models.name in (
        GRADIENT_BOOSTING_TREE_MODELS + CATBOOST_MODELS + LIGHTGBM_MODELS
    ):
        return calculateSHAPValuesForGradientBoostingTrees(
            config=config,
            model=model,
            fold_test_num=fold_test_num,
            fold_val_num=fold_val_num,
            device=device,
        )

    else:
        raise ValueError(
            f"SHAP値の計算方法が指定されていないモデルです: {config.args.models.name}"
        )


@ray.remote(num_gpus=1)
def calcShapValuesByExplainer(config, model, background_x, data, feature_names):
    with torch.inference_mode():
        masker = shap.maskers.Independent(data=background_x, max_samples=100)

        explainer = shap.Explainer(
            model=model,
            masker=masker,
            seed=config.args.seed,
            feature_names=feature_names,
            output_names=config.args.tasks.targets,
        )

        shap_values = explainer(data)

    return shap_values


def calcShapValuesByRay(
    config, model, masker, data, background_data, feature_names, n_splits=10
):
    """Split the data and compute SHAP values in parallel with Ray."""
    shap_results_dict = {n: {} for n in range(n_splits)}

    process_list = []
    splits = np.array_split(data, n_splits)
    # import copy

    for idx, data_splited in enumerate(splits):
        # masker = shap.maskers.Independent(data=background_data, max_samples=100)

        # explainer = shap.Explainer(
        #     model=model,
        #     masker=copy.deepcopy(masker),
        #     seed=config.args.seed,
        #     feature_names=feature_names,
        #     output_names=config.args.tasks.targets,
        # )

        process = calcShapValuesByExplainer.remote(
            config, model, background_data, data_splited, feature_names
        )

        process_list.append(process)

    results_list = ray.get(process_list)

    for num_splited, results in enumerate(results_list):
        shap_results_dict[num_splited] = results

    ray.shutdown()

    return shap_results_dict


def calculateSHAPValuesForNeuralNetworks(
    config: DictConfig | ListConfig, model, fold_test_num, fold_val_num, device
):
    current_date = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    print(f"start: {current_date}")

    explainer_name = ""
    args_explainer = {}

    if "shap" in config.args.advance_setting:
        args_explainer: dict | Any = OmegaConf.to_container(config.args.advance_setting.shap)

        if "explainer_name" in args_explainer:
            explainer_name = args_explainer["explainer_name"]

    background_dataset = HealthDataset(
        config=config,
        fold_test_num=fold_test_num,
        fold_val_num=fold_val_num,
        phase="train",
    )
    test_dataset = HealthDataset(
        config=config,
        fold_test_num=fold_test_num,
        fold_val_num=fold_val_num,
        phase="test",
    )

    if config.args.dataset.is_splited_into_cont_and_cat and explainer_name in ["deep_explainer"]:
        # Test data
        test_continuous_x = test_dataset.x.loc[:, test_dataset.continuous_columns].values
        test_continuous_x = torch.tensor(test_continuous_x, dtype=torch.float32).to(device=device)

        test_categorical_x = test_dataset.x.loc[:, test_dataset.categorical_columns].values
        test_categorical_x = torch.tensor(test_categorical_x, dtype=torch.int32).to(device=device)

        test_x = [test_continuous_x, test_categorical_x]

        # Background samples
        background_x = background_dataset.x.loc[:, background_dataset.use_columns]
        background_x = background_x.sample(n=1000, random_state=config.args.seed)

        # Background dataset
        background_continuous_x = background_x.loc[:, background_dataset.continuous_columns]
        background_continuous_x = torch.tensor(background_continuous_x.values, dtype=torch.float32).to(device=device)
        background_categorical_x = background_x.loc[:, background_dataset.categorical_columns]
        background_categorical_x = torch.tensor(background_categorical_x.values, dtype=torch.int32).to(device=device)

        background_x = [background_continuous_x, background_categorical_x]

    else:
        pass

    index_list = test_dataset.x.index.tolist()

    model.eval()
    model = model.to(device)
    model.is_calc_shap_values = True

    shap_values_dict = {}
    base_values = None
    shap_values = None

    with torch.inference_mode():
        if explainer_name == "deep_explainer":
            del args_explainer["explainer_name"]

            model.return_attention = False
            explainer = shap.DeepExplainer(
                model=model,
                data=background_x,
                **args_explainer,
            )
            shap_values = explainer.shap_values(test_x)

        elif explainer_name == "gradient_explainer":
            explainer = shap.GradientExplainer(model=model, data=background_x, **args_explainer)
            shap_values = explainer.shap_values(test_x)

        else:
            test_x = test_dataset.x.loc[:, test_dataset.use_columns]
            background_x = background_dataset.x.loc[:, background_dataset.use_columns]
            background_x = background_x.sample(n=100, random_state=config.args.seed).values

            # Calculate SHAP values
            n_cont_features = config.args.models.params.n_cont_features

            def predict(x):
                chunk_size = 512

                y_pred_list = []

                for i in range(0, len(x), chunk_size):
                    if config.args.dataset.is_splited_into_cont_and_cat:
                        x_cont, x_cat = x[i : i + chunk_size, 0:n_cont_features], x[i : i + chunk_size, n_cont_features:]
                        x_cont = torch.tensor(x_cont, dtype=torch.float32, device=model.device)
                        x_cat = torch.tensor(x_cat, dtype=torch.long, device=model.device)

                        with torch.inference_mode():
                            if config.args.models.name in TRANSFORMER_BASED_MODELS:
                                y_pred, _ = model.forward(x_cont=x_cont, x_cat=x_cat)
                            else:
                                y_pred = model.forward(x_cont=x_cont, x_cat=x_cat)

                    else:
                        x_input = torch.tensor(x[i : i + chunk_size, :], dtype=torch.float32, device=model.device)

                        with torch.inference_mode():
                            _, y_pred = model.forward(x_input)

                    y_pred_list.append(y_pred)

                y_pred_list = torch.cat(y_pred_list, dim=0)
                return y_pred_list.to("cpu").numpy()

            # Distributed computation (commented out)
            # if config.args.advance_setting.use_multiprocessing:
            #     shap_values_dict = calcShapValuesByRay(
            #         config=config,
            #         model=predict,
            #         masker=masker,
            #         data=test_x.values,
            #         background_data=background_x,
            #         feature_names=test_x.columns,
            #     )

            # else:
            masker = shap.maskers.Independent(data=background_x, max_samples=100)

            explainer = shap.Explainer(
                model=predict,
                masker=masker,
                seed=config.args.seed,
                feature_names=test_x.columns,
                output_names=config.args.tasks.targets,
            )

            data = test_x.values

            if "debug_calc_shap" in config.args.advance_setting:
                if config.args.advance_setting.debug_calc_shap:
                    data = test_x.values[0:10]
                    index_list = index_list[0:10]

            shap_values_data = explainer(data)

            # Output
            shap_values = shap_values_data.values
            base_values = shap_values_data.base_values

    if len(config.args.tasks.targets) == 1:
        shap_values = np.expand_dims(shap_values, -1)

    for idx, task_name in enumerate(config.args.tasks.targets):
        shap_values_dict[task_name] = {
            "shap_values": pd.DataFrame(
                data=shap_values[:, :, idx],
                index=index_list,
                columns=test_dataset.use_columns,
            ),
            "base_value": pd.DataFrame(base_values[:, idx]),
        }

    model.is_calc_shap_values = False
    current_date = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    print(f"current_date: {current_date}")

    return shap_values_dict
