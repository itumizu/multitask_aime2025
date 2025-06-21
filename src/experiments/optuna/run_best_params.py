from typing import Any
from pathlib import Path
from omegaconf import OmegaConf, DictConfig, ListConfig

from src.experiments.optuna.set_best_params import setBestParamsToConfig
from src.experiments.run_experiment import runExperiment

def experimentByBestConfig(
    config: DictConfig | ListConfig,
    gpus: list,
    best_params,
    default_root_dir: Path,
    fold_test_num: int,
    experiment_id: str,
    run_name: str,
):
    config_best = setBestParamsToConfig(config=config, best_params=best_params)
    run_name_best = f"trial_best_{run_name}"

    config_best_dict: Any = OmegaConf.to_container(config_best)

    tags_dict = {"fold_test": fold_test_num, "trial": "best", "phase": "test", "ulid": config.args.ulid}
    log_dict = {"dictionary": config_best_dict, "artifact_file": "params_best.yaml"}
    dir_name = "best"

    mean_value_dict = runExperiment(
        config=config_best,
        gpus=gpus,
        fold_test_num=fold_test_num,
        default_root_dir=default_root_dir,
        dir_name=dir_name,
        tags_dict=tags_dict,
        log_dict=log_dict,
        experiment_id=experiment_id,
        run_name=run_name_best,
        return_validation_results_only=False,
        is_best=True,
    )

    return mean_value_dict
