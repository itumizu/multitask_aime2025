import os
from pathlib import Path

import ray
import lightning.pytorch as pl
from omegaconf import DictConfig, ListConfig

from src.modules.train import (
    trainNeuralNetworkModel,
    trainGradientBoostingTree,
    trainLightGBM,
    trainCatboost,
)

from src.etc.constants import (
    CATBOOST_MODELS,
    GRADIENT_BOOSTING_TREE_MODELS,
    NEURAL_NETWORK_MODELS,
    LIGHTGBM_MODELS,
)


def experimentValidationFold(
    config: DictConfig | ListConfig,
    gpus: list,
    fold_test_num: int,
    fold_val_num: int,
    default_root_dir: Path,
    return_validation_results_only=False,
    return_with_model=False,
):
    pl.seed_everything(config.args.seed, workers=True)

    default_root_dir_fold = Path(default_root_dir, f"fold_test_{fold_test_num}", f"fold_{fold_val_num}")
    os.makedirs(default_root_dir_fold, exist_ok=True)

    if config.args.models.name in GRADIENT_BOOSTING_TREE_MODELS:
        results_dict = trainGradientBoostingTree(
            config=config,
            gpus=gpus,
            fold_test_num=fold_test_num,
            fold_val_num=fold_val_num,
            default_root_dir_fold=default_root_dir_fold,
            return_with_model=return_with_model,
            return_validation_results_only=return_validation_results_only,
        )

    elif config.args.models.name in CATBOOST_MODELS:
        results_dict = trainCatboost(
            config=config,
            gpus=gpus,
            fold_test_num=fold_test_num,
            fold_val_num=fold_val_num,
            default_root_dir_fold=default_root_dir_fold,
            return_with_model=return_with_model,
            return_validation_results_only=return_validation_results_only,
        )


    elif config.args.models.name in LIGHTGBM_MODELS:
        results_dict = trainLightGBM(
            config=config,
            gpus=gpus,
            fold_test_num=fold_test_num,
            fold_val_num=fold_val_num,
            default_root_dir_fold=default_root_dir_fold,
            return_with_model=return_with_model,
            return_validation_results_only=return_validation_results_only,
        )


    elif config.args.models.name in NEURAL_NETWORK_MODELS:
        results_dict = trainNeuralNetworkModel(
            config=config,
            gpus=gpus,
            fold_test_num=fold_test_num,
            fold_val_num=fold_val_num,
            default_root_dir_fold=default_root_dir_fold,
            return_with_model=return_with_model,
            return_validation_results_only=return_validation_results_only,
        )
    else:
        raise

    results_dict["fold_val_num"] = fold_val_num

    return results_dict


@ray.remote(num_gpus=1)
def experimentValidationFoldByRay(
    config: DictConfig | ListConfig,
    gpus: list,
    fold_test_num: int,
    fold_val_num: int,
    default_root_dir: Path,
    return_validation_results_only: bool,
    return_with_model: bool,
):
    results_dict = experimentValidationFold(
        config=config,
        gpus=gpus,
        fold_test_num=fold_test_num,
        fold_val_num=fold_val_num,
        default_root_dir=default_root_dir,
        return_validation_results_only=return_validation_results_only,
        return_with_model=return_with_model,
    )

    return results_dict
