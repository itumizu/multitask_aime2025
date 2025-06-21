import os
from pathlib import Path

import mlflow
from omegaconf import DictConfig, ListConfig, OmegaConf
from src.experiments.run_crossvalidation import experimentCrossValidation
from src.utils.set_results import setResultsOfEachFold
from src.utils.set_params import setParamsFromConfig


def runExperiment(
    config: DictConfig | ListConfig,
    gpus: list,
    fold_test_num: int,
    default_root_dir: Path,
    dir_name: str,
    tags_dict: dict,
    log_dict: dict,
    run_name: str,
    experiment_id: str,
    return_validation_results_only: bool = False,
    return_with_model: bool = False,
    is_best: bool = False,
):
    default_root_dir = Path(default_root_dir, dir_name)
    os.makedirs(default_root_dir, exist_ok=True)

    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
        mean_value_dict, fold_results_dict = experimentCrossValidation(
            config=config,
            gpus=gpus,
            fold_test_num=fold_test_num,
            default_root_dir=default_root_dir,
            parent_run_id=run.info.run_id,
            return_validation_results_only=return_validation_results_only,
            return_with_model=return_with_model,
            is_best=is_best,
        )

        OmegaConf.save(config, Path(default_root_dir, log_dict["artifact_file"]))

        mlflow.set_tags(tags=tags_dict)
        mlflow.log_dict(**log_dict)
        mlflow.log_metrics(metrics=mean_value_dict)
        mlflow.log_artifact(str(Path(default_root_dir, log_dict["artifact_file"])), "config")

        if is_best:
            setParamsFromConfig(config=config)

            setResultsOfEachFold(
                config=config,
                fold_results_dict=fold_results_dict,
                mean_value_dict=mean_value_dict,
                default_root_dir=default_root_dir,
                fold_test_num=fold_test_num,
                is_best=is_best,
            )

        else:
            setParamsFromConfig(config=config)

    return mean_value_dict, default_root_dir
