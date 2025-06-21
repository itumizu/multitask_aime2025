import traceback
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from omegaconf import ListConfig, OmegaConf, DictConfig

import xgboost

from src.loaders.dataset import HealthDataset
from src.utils.select_model import selectModel
from src.utils.calc_metrics import calculateMetrics
from src.utils.calc_shap_values import calculateSHAPValues

def trainGradientBoostingTree(
    config: DictConfig | ListConfig,
    gpus: list,
    fold_test_num: int,
    fold_val_num: int,
    default_root_dir_fold: Path,
    return_with_model=False,
    return_validation_results_only=False,
) -> dict:
    # Load datasets
    dataset_train = HealthDataset(config=config, fold_test_num=fold_test_num, fold_val_num=fold_val_num, phase="train")
    dataset_val = HealthDataset(config=config, fold_test_num=fold_test_num, fold_val_num=fold_val_num, phase="val")
    dataset_test = HealthDataset(config=config, fold_test_num=fold_test_num, fold_val_num=fold_val_num, phase="test")

    X_train, y_train = dataset_train.getDataset()
    X_val, y_val = dataset_val.getDataset()
    X_test, y_test = dataset_test.getDataset()

    # Model
    model: xgboost.XGBModel = selectModel(config=config, gpus=gpus, fold_test_num=0, fold_val_num=0)

    use_multiprocessing = config.args.advance_setting.get("use_multiprocessing", False)

    # Train the model
    model.fit(
        X=X_train,
        y=y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=not use_multiprocessing,
    )

    evals_result = model.evals_result()
    outputs_dict = {"train": {}, "val": {}, "test": {}}

    # Evaluate on training data
    y_pred_train = model.predict(X=X_train)

    # Evaluate on validation data
    y_pred_val = model.predict(X=X_val)

    # Participant IDs
    outputs_dict["train"]["participant_id_list"] = X_train.index.tolist()
    outputs_dict["val"]["participant_id_list"] = X_val.index.tolist()

    if len(y_pred_val.shape) == 1:
        y_pred_train = np.expand_dims(y_pred_train, -1)
        y_pred_val = np.expand_dims(y_pred_val, -1)

    results_dict = {"logging": {"metrics": {}}}

    for idx, target_name in enumerate(config.args.tasks.targets):
        train_loss = evals_result["validation_0"][config.args.models.params.eval_metric]
        val_loss = evals_result["validation_1"][config.args.models.params.eval_metric]

        # Training
        outputs_dict["train"][f"y_pred_{target_name}"] = y_pred_train[:, idx]
        outputs_dict["train"][f"y_true_{target_name}"] = y_train.iloc[:, idx]
        outputs_dict["train"][f"loss_{target_name}"] = train_loss
        results_dict["logging"]["metrics"]["train"] = [
            {f"train_loss_{target_name}": value} for value in train_loss
        ]

        # Validation
        outputs_dict["val"][f"y_pred_{target_name}"] = y_pred_val[:, idx]
        outputs_dict["val"][f"y_true_{target_name}"] = y_val.iloc[:, idx]
        outputs_dict["val"][f"loss_{target_name}"] = val_loss
        results_dict["logging"]["metrics"]["val"] = [
            {f"val_loss_{target_name}": value} for value in val_loss
        ]

    metrics_dict_train, _ = calculateMetrics(config=config, outputs_dict=outputs_dict, phase="train")
    metrics_dict_val, _ = calculateMetrics(config=config, outputs_dict=outputs_dict, phase="val")

    results_dict["train_results"] = metrics_dict_train
    results_dict["val_results"] = metrics_dict_val

    if not return_validation_results_only:
        for idx, target_name in enumerate(config.args.tasks.targets):
            y_pred_test = model.predict(X=X_test)

            if len(y_pred_test.shape) == 1:
                y_pred_test = np.expand_dims(y_pred_test, -1)

            outputs_dict["test"][f"y_pred_{target_name}"] = y_pred_test[:, idx]
            outputs_dict["test"][f"y_true_{target_name}"] = y_test.iloc[:, idx]
            outputs_dict["test"][f"loss_{target_name}"] = [-1]
            outputs_dict["test"]["participant_id_list"] = X_test.index.tolist()

        shap_values_dict = {}
        metrics_dict_test, df_true_pred = calculateMetrics(config=config, outputs_dict=outputs_dict, phase="test")

        try:
            shap_values_dict = calculateSHAPValues(
                config=config,
                model=model,
                fold_test_num=fold_test_num,
                fold_val_num=fold_val_num,
                device="cpu",
            )
        except:
            error_message = traceback.format_exc()
            print(error_message)
            shap_values_dict = {}

        results_dict["test_results"] = metrics_dict_test
        results_dict["df_true_pred"] = {"df_true_pred": df_true_pred}
        results_dict["shap"] = {"shap": shap_values_dict}

    if return_with_model:
        results_dict["model"] = model

    return results_dict
