import os
import traceback
from pathlib import Path
from typing import Any

import numpy as np

import ray
import optuna
from optuna.storages import RetryFailedTrialCallback

from mlflow.entities import Experiment
from omegaconf import ListConfig, DictConfig

from src.experiments.run_experiment import runExperiment
from src.experiments.optuna.select_params import selectParam
from src.experiments.optuna.callbacks import StopAfterCompletedTrialsCallback
from src.experiments.optuna.run_best_params import experimentByBestConfig

from src.utils.set_default_root_dir import setDefaltRootDir
from src.utils.send_notification import sendNotification
from src.utils.send_notion import updateExperimentStatus
from src.utils.remove_tempfiles import removeTempFiles


class Objective:
    def __init__(
        self,
        config: DictConfig | ListConfig,
        gpus: list,
        experiment: Experiment,
        run_name: str,
        fold_test_num: int,
        default_root_dir: Path,
    ):
        self.config = config
        self.fold_test_num = fold_test_num
        self.run_name = run_name
        self.experiment = experiment
        self.gpus = gpus
        self.optimization_criterion = config.args.optuna.optimization_criterion
        self.optimization_direction = config.args.optuna.optimization_direction
        self.default_root_dir = default_root_dir

        self.trash_output_dir = None

    def __call__(self, trial: optuna.trial.Trial):
        params = self.config.copy()
        hyperparameter_space_config = self.config.args.optuna.hyperparameter_space

        params = selectParam(
            trial=trial,
            params=params,
            hyperparameter_space_config=hyperparameter_space_config,
        )

        print("params: ", params)
        config_trial = params
        config_trial.args.trial_number = trial.number
        config_trial.args.run_name = self.run_name

        run_name_trial = f"trial_{trial.number}_{self.run_name}"
        dir_name = f"trial_{trial.number}"

        log_dict = {"dictionary": config_trial, "artifact_file": "params_best.yaml"}
        tags_dict = {"fold_test": self.fold_test_num, "trial": trial.number, "phase": "train"}

        mean_value_dict, output_dir = runExperiment(
            config=config_trial,
            gpus=self.gpus,
            fold_test_num=self.fold_test_num,
            default_root_dir=self.default_root_dir,
            dir_name=dir_name,
            tags_dict=tags_dict,
            log_dict=log_dict,
            experiment_id=self.experiment.experiment_id,
            run_name=run_name_trial,
            return_validation_results_only=True,
        )

        if self.trash_output_dir:
            try:
                removeTempFiles(target_path=output_dir)

            except:
                error_message = traceback.format_exc()
                print(error_message)

                response = updateExperimentStatus(
                    page_id=config_trial.args.notion_page_id,
                    status="実行中(エラーあり)",
                    message=error_message,
                )

            self.trash_output_dir = output_dir

        if isinstance(self.optimization_criterion, DictConfig):
            criterion_list = []

            for target_name, _ in self.optimization_direction.items():
                criterion_list.append(mean_value_dict[self.optimization_criterion[target_name]])

            return tuple(criterion_list)

        else:
            return mean_value_dict[self.optimization_criterion]


def experimentCrossValidationWithOptuna(
    config_path: Path,
    config: DictConfig | ListConfig,
    gpus: list,
    run_name: str,
    experiment: Experiment,
    fold_test_num: int,
):
    config.args.experiment_id = experiment.experiment_id
    default_root_dir = setDefaltRootDir(config=config)

    # 初期化
    objective = Objective(
        config=config,
        gpus=gpus,
        experiment=experiment,
        run_name=run_name,
        fold_test_num=fold_test_num,
        default_root_dir=default_root_dir,
    )

    direction = None
    directions = None

    # Multi-objective optimization
    if isinstance(config.args.optuna.optimization_direction, DictConfig):
        directions = [target_direction for _, target_direction in config.args.optuna.optimization_direction.items()]

    else:
        direction = config.args.optuna.optimization_direction

    storage = optuna.storages.RDBStorage(
        url=os.environ["POSTGRES_DB_URI"] + "/optuna",
        heartbeat_interval=60 * 3,
        failed_trial_callback=RetryFailedTrialCallback(max_retry=5),
    )

    sampler = optuna.samplers.TPESampler(seed=config.args.seed, multivariate=True, group=True, constant_liar=True)
    study = optuna.create_study(
        study_name=f"{config.args.experiment_name}_{config_path.parent.stem}_{config_path.stem}_fold_test_{fold_test_num}",
        direction=direction,
        directions=directions,
        sampler=sampler,
        load_if_exists=True,
        storage=storage,
    )

    n_trials = config.args.optuna.n_trials
    mean_value_dict = {}

    if "advance_setting" in config.args:
        if "n_jobs" in config.args.advance_setting:
            n_jobs = config.args.advance_setting.n_jobs

    # これまでに完了しているトライアルがあれば、それを継続して実行する
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"n_trials: {n_trials} completed_trials: {len(completed_trials)}")

    print(f"gpus: {gpus}")

    if len(gpus) > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(gpus[n % len(gpus)]) for n in range(0, 5)])
        ray.init(num_gpus=5, num_cpus=64)

    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "".join(list(map(str, gpus)))

    if len(completed_trials) < config.args.optuna.n_trials:
        n_trials = n_trials - len(completed_trials)

        callbacks = [StopAfterCompletedTrialsCallback(config=config)]
        study.optimize(objective, callbacks=callbacks)

    # 最良のパラメータをconfigにセット
    if directions:
        # すべてminimize か maximize の場合は、自動的に選定する
        if directions.count("minimize") == len(directions) or directions.count("maximize") == len(directions):
            metrics = []

            for idx, trial in enumerate(study.best_trials):
                values = float(np.mean(trial.values))

                if not len(metrics):
                    metrics = [idx, values]

                else:
                    if "minimize" in directions:
                        if values < metrics[1]:
                            metrics = [idx, values]

                    elif "maximize" in directions:
                        if values > metrics[1]:
                            metrics = [idx, values]

            best_params = study.best_trials[metrics[0]].params

        else:
            raise

    else:
        best_params = study.best_params

    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    if len(completed_trials) == config.args.optuna.n_trials:
        mean_value_dict = experimentByBestConfig(
            config=config,
            gpus=gpus,
            best_params=best_params,
            default_root_dir=default_root_dir,
            fold_test_num=fold_test_num,
            experiment_id=experiment.experiment_id,
            run_name=run_name,
        )

    elif len(completed_trials) > config.args.optuna.n_trials:
        sendNotification(
            f"エラー: {run_name} (fold test {fold_test_num})",
            message=f"この条件は、{len(completed_trials)} trials 実施されています。",
            add_current_time=True,
        )

    else:
        pass

    return mean_value_dict
