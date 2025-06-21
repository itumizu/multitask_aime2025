from typing import Any

import mlflow
from omegaconf import OmegaConf, DictConfig, ListConfig


def setParamsFromConfig(config: DictConfig | ListConfig) -> None:
    for category_name, params in config.args.items():
        if category_name in ["optuna"]:
            continue

        if isinstance(params, (str, int)):
            mlflow.log_param(key=category_name, value=params)
            continue

        for param_name, param_values in params.items():
            if param_name == "params" and isinstance(param_values, DictConfig):
                params_dict: Any = OmegaConf.to_container(param_values)
                mlflow.log_params(params_dict)

            elif category_name == "callbacks":
                mlflow.log_params({f"{param_name}_{key}": value for key, value in param_values.items()})

            elif param_name == "name":
                mlflow.log_param(key=f"{category_name}_{param_name}", value=param_values)

            else:
                mlflow.log_param(key=param_name, value=param_values)
