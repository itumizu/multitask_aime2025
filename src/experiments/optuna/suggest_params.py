from typing import Any

import optuna
from omegaconf import DictConfig, ListConfig


def suggestParams(trial: optuna.trial.Trial, hyper_param_name: str, params_dict: DictConfig):
    search_type = params_dict.search_type
    step: Any = None

    if search_type == "int":
        step = 1

    if "step" in params_dict:
        step = params_dict.step

    if search_type == "float":
        return trial.suggest_float(hyper_param_name, params_dict.min, params_dict.max, step=step)

    elif search_type == "int":
        return trial.suggest_int(hyper_param_name, params_dict.min, params_dict.max, step=step)

    elif search_type == "log":
        return trial.suggest_float(hyper_param_name, params_dict.min, params_dict.max, step=step, log=True)

    elif search_type == "select":
        return trial.suggest_categorical(hyper_param_name, params_dict.choices)

    else:
        raise ValueError("Unsupported search_type in the configuration: {}".format(search_type))
