import os
from typing import Tuple
from pathlib import Path

import ray
import numpy as np
import pandas as pd
from omegaconf import DictConfig, ListConfig

from src.experiments.run_val_fold import experimentValidationFold, experimentValidationFoldByRay


def calculateMeanValue(fold_results_dict: dict) -> dict:
    # クロスバリデーションの結果を集計する
    mean_results_dict = {}

    for fold_val_num, results_dict in sorted(fold_results_dict.items(), reverse=True, key=lambda x: x[0]):
        for result_phase_name, results_by_phase in results_dict.items():
            if result_phase_name in ["fold_val_num", "logging", "shap", "df_true_pred", ""]:
                continue

            if not isinstance(results_by_phase, dict):
                continue

            for metrics_name, metrics_value in results_by_phase.items():
                if isinstance(metrics_value, pd.DataFrame) or isinstance(metrics_value, dict):
                    continue

                mean_results_dict.setdefault(metrics_name, [])
                mean_results_dict[metrics_name].append(metrics_value)

    mean_value_dict = {key: np.mean(value_list) for key, value_list in mean_results_dict.items()}

    temp_dict = {}

    for key, value_list in mean_results_dict.items():
        key_list = key.split("_")

        phase = key_list[0]
        metrics_name = key_list[1]

        temp_dict.setdefault(phase, {})
        temp_dict[phase].setdefault(metrics_name, [])
        temp_dict[phase][metrics_name].extend(value_list)

    for phase, value_dict in temp_dict.items():
        for metrics_name, value_list in value_dict.items():
            mean_value_dict[f"{phase}_{metrics_name}"] = np.mean(value_list)

    return mean_value_dict


def experimentCrossValidationByMultiProcessing(
    config: DictConfig | ListConfig,
    gpus: list,
    fold_test_num: int,
    default_root_dir: Path,
    fold_start_num: int,
    fold_end_num: int,
    return_validation_results_only=False,
    return_with_model=False,
):
    fold_results_dict = {n: {} for n in range(fold_start_num, fold_end_num + 1)}
    process_list = []

    for idx, fold_val_num in enumerate(range(fold_start_num, fold_end_num + 1)):
        use_gpu_list = []

        if len(gpus) > 0:
            use_gpu_list = [0]

        process = experimentValidationFoldByRay.options(num_gpus=len(use_gpu_list)).remote(
            config,
            use_gpu_list,
            fold_test_num,
            fold_val_num,
            default_root_dir,
            return_validation_results_only,
            return_with_model,
        )

        process_list.append(process)

    results_list = ray.get(process_list)

    for fold_num, results in enumerate(results_list):
        fold_results_dict[fold_num] = results

    ray.shutdown()

    return fold_results_dict


def experimentCrossValidation(
    config: DictConfig | ListConfig,
    gpus: list,
    fold_test_num: int,
    default_root_dir: Path,
    return_validation_results_only=False,
    return_with_model=False,
    is_best=False,
    parent_run_id: int | None = None,
) -> Tuple[dict, dict]:

    fold_start_num = 0
    fold_end_num = 4

    if "fold_val_num" in config.args.dataset:
        fold_start_num = config.args.dataset.fold_val_num
        fold_end_num = config.args.dataset.fold_val_num

    use_multiprocessing = False

    if "advance_setting" in config.args:
        if "use_multiprocessing" in config.args.advance_setting:
            use_multiprocessing = config.args.advance_setting.use_multiprocessing

    if use_multiprocessing:
        fold_results_dict = experimentCrossValidationByMultiProcessing(
            config=config,
            gpus=gpus,
            fold_test_num=fold_test_num,
            default_root_dir=default_root_dir,
            fold_start_num=fold_start_num,
            fold_end_num=fold_end_num,
            return_validation_results_only=return_validation_results_only,
            return_with_model=return_with_model,
        )

    else:
        if len(gpus) > 1:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(list(map(str, gpus)))

        fold_results_dict = {n: {} for n in range(fold_start_num, fold_end_num + 1)}

        for idx, fold_val_num in enumerate(range(fold_start_num, fold_end_num + 1)):
            results_dict = experimentValidationFold(
                config,
                gpus=gpus,
                fold_test_num=fold_test_num,
                fold_val_num=fold_val_num,
                default_root_dir=default_root_dir,
                return_validation_results_only=return_validation_results_only,
                return_with_model=return_with_model,
            )

            fold_results_dict[fold_val_num] = results_dict

    mean_value_dict = calculateMeanValue(fold_results_dict=fold_results_dict)

    return mean_value_dict, fold_results_dict
