import traceback
from pathlib import Path

import numpy as np
from omegaconf import ListConfig, DictConfig

import catboost

from src.loaders.dataset import HealthDataset
from src.utils.select_model import selectModel
from src.utils.calc_metrics import calculateMetrics
from src.utils.calc_shap_values import calculateSHAPValues
from src.utils.send_notification import sendNotification


def trainCatboost(
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

    # Identify categorical feature indices
    cat_features = [idx for idx, column in enumerate(dataset_train.use_columns) if column in dataset_train.categorical_columns]

    for column in dataset_train.categorical_columns:
        X_train[column] = X_train[column].astype(int)
        X_val[column] = X_val[column].astype(int)
        X_test[column] = X_test[column].astype(int)

    # Initialize model
    model: catboost.CatBoost = selectModel(config=config, gpus=gpus, fold_test_num=0, fold_val_num=0)

    use_multiprocessing = config.args.advance_setting.get("use_multiprocessing", False)

    print(cat_features)

    # Train the model
    model.fit(
        X=X_train,
        y=y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False if use_multiprocessing else True,
        early_stopping_rounds=config.args.callbacks.early_stopping.patience if config.args.callbacks.early_stopping.use_callback else 0,
        cat_features=cat_features,
    )

    evals_result = model.evals_result_
    print(evals_result)

    outputs_dict = {"train": {}, "val": {}, "test": {}}

    # Evaluate on training set
    y_pred_train = model.predict(data=X_train)

    # Evaluate on validation set
    y_pred_val = model.predict(data=X_val)

    # Store participant IDs
    outputs_dict["train"]["participant_id_list"] = X_train.index.tolist()
    outputs_dict["val"]["participant_id_list"] = X_val.index.tolist()

    # Ensure 2D shape
    if len(y_pred_val.shape) == 1:
        y_pred_train = np.expand_dims(y_pred_train, -1)
        y_pred_val = np.expand_dims(y_pred_val, -1)

    results_dict = {"logging": {"metrics": {}}}

    for idx, target_name in enumerate(config.args.tasks.targets):
        train_loss = evals_result[f"validation_0"][config.args.models.params.eval_metric]
        val_loss = evals_result[f"validation_1"][config.args.models.params.eval_metric]

        # Training results
        outputs_dict["train"][f"y_pred_{target_name}"] = y_pred_train[:, idx]
        outputs_dict["train"][f"y_true_{target_name}"] = y_train.iloc[:, idx]
        outputs_dict["train"][f"loss_{target_name}"] = train_loss
        results_dict["logging"]["metrics"]["train"] = [
            {f"train_loss_{target_name}": value} for value in train_loss
        ]

        # Validation results
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
            # Evaluate on test set
            y_pred_test = model.predict(data=X_test)

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
                device="cpu"
            )
        except:
            error_message = traceback.format_exc()
            print(error_message)

            sendNotification(
                "Error (SHAP)",
                message=f"Details: {str(error_message)}",
                add_current_time=True,
            )

        results_dict["test_results"] = metrics_dict_test
        results_dict["df_true_pred"] = {"df_true_pred": df_true_pred}
        results_dict["shap"] = {"shap": shap_values_dict}

    if return_with_model:
        results_dict["model"] = model

    return results_dict
