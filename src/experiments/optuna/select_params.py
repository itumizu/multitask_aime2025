import optuna
from omegaconf import DictConfig, ListConfig
from src.experiments.optuna.suggest_params import suggestParams


def splitParamName(hyper_param_name: str):
    """Split a dotted hyper-parameter name into its components."""
    hyper_param_name_list = hyper_param_name.split(".")

    if len(hyper_param_name_list) == 2:
        target_root_param_name = hyper_param_name_list[0]
        target_first_param_name = hyper_param_name_list[1]
        target_second_param_name = ""

    elif len(hyper_param_name_list) == 3:
        target_root_param_name = hyper_param_name_list[0]
        target_first_param_name = hyper_param_name_list[1]
        target_second_param_name = hyper_param_name_list[2]

    else:
        raise ValueError("Unsupported hyper-parameter name depth")

    return target_root_param_name, target_first_param_name, target_second_param_name


def getParamValueByParamName(params: dict | DictConfig | ListConfig, hyper_param_name: str):
    """Retrieve the value of a parameter given its dotted name."""
    root_param_name, first_param_name, second_param_name = splitParamName(hyper_param_name)
    if "args" in params:  # Configs wrapped under an ``args`` key
        if second_param_name:
            return params["args"][root_param_name][first_param_name][second_param_name]
        return params["args"][root_param_name][first_param_name]
    else:  # Plain DictConfig / ListConfig
        if second_param_name:
            return params[root_param_name][first_param_name][second_param_name]
        return params[root_param_name][first_param_name]


def setInputParamToTargetParam(params, input_param_name, target_param_name):
    """Copy a value from *input_param_name* to *target_param_name* inside *params*."""
    root_param_name, first_param_name, second_param_name = splitParamName(target_param_name)

    if second_param_name:
        params["args"][root_param_name][first_param_name][second_param_name] = getParamValueByParamName(
            params, input_param_name
        )
    else:
        params["args"][root_param_name][first_param_name] = getParamValueByParamName(params, input_param_name)

    return params


def setParamToTargetParam(params, target_param_name, input_param_value):
    """Set *input_param_value* into *target_param_name* inside *params*."""
    root_param_name, first_param_name, second_param_name = splitParamName(target_param_name)

    if second_param_name:
        params["args"][root_param_name][first_param_name][second_param_name] = input_param_value
    else:
        params["args"][root_param_name][first_param_name] = input_param_value

    return params


def selectParam(
    trial: optuna.trial.Trial,
    params: DictConfig | ListConfig,
    hyperparameter_space_config: DictConfig,
):
    """
    Suggest and set hyper-parameters according to *hyperparameter_space_config*.

    Returns the updated *params*.
    """
    params_instead_dict = {}  # Mapping of ``use_instead_of`` → list of target names
    params_use_range_dict = {}  # Mapping of params that need dynamic range adjustment

    # ---- First pass: perform suggestions or record special cases ----
    for root_param_name, root_space_dict in hyperparameter_space_config.items():
        for first_param_name, first_param_dict in root_space_dict.items():
            if "search_type" in first_param_dict:  # Leaf node (depth 2)
                hyper_param_name = f"{root_param_name}.{first_param_name}"

                if "use_instead_of" in first_param_dict:
                    # This param copies the value of another param
                    param_name_instead = first_param_dict.use_instead_of
                    if len(param_name_instead.split(".")) == 1:
                        param_name_instead = f"{root_param_name}.{param_name_instead}"

                    params_instead_dict.setdefault(param_name_instead, [])
                    params_instead_dict[param_name_instead].append(hyper_param_name)

                elif "range" in first_param_dict:
                    # Will be suggested later after its dynamic range is resolved
                    params_use_range_dict[hyper_param_name] = {}

                else:
                    # Standard suggestion
                    params = setParamToTargetParam(
                        params=params,
                        target_param_name=hyper_param_name,
                        input_param_value=suggestParams(  # type: ignore
                            trial=trial,
                            hyper_param_name=hyper_param_name,
                            params_dict=first_param_dict,
                        ),
                    )

            else:  # One level deeper (depth 3)
                for second_param_name, second_param_dict in first_param_dict.items():
                    hyper_param_name = f"{root_param_name}.{first_param_name}.{second_param_name}"

                    if "search_type" in second_param_dict:
                        if "use_instead_of" in second_param_dict:
                            param_name_instead = second_param_dict.use_instead_of
                            if len(param_name_instead.split(".")) == 1:
                                param_name_instead = f"{root_param_name}.{first_param_name}.{param_name_instead}"

                            params_instead_dict.setdefault(param_name_instead, [])
                            params_instead_dict[param_name_instead].append(hyper_param_name)

                        elif "range" in second_param_dict:
                            params_use_range_dict[hyper_param_name] = {}

                        else:
                            params = setParamToTargetParam(
                                params=params,
                                target_param_name=hyper_param_name,
                                input_param_value=suggestParams(  # type: ignore
                                    trial=trial,
                                    hyper_param_name=hyper_param_name,
                                    params_dict=second_param_dict,
                                ),
                            )

                    else:
                        raise ValueError("Invalid hyper-parameter configuration")

    # ---- Second pass: handle ``use_instead_of`` relationships ----
    for root_param_name, root_space_dict in hyperparameter_space_config.items():
        for first_param_name, first_param_dict in root_space_dict.items():
            param_name = f"{root_param_name}.{first_param_name}"

            if param_name in params_instead_dict:
                for target_param_name in params_instead_dict[param_name]:
                    params = setInputParamToTargetParam(
                        params=params,
                        input_param_name=param_name,
                        target_param_name=target_param_name,
                    )

            # Dive one level deeper
            for second_param_name, second_param_dict in first_param_dict.items():
                param_name = f"{root_param_name}.{first_param_name}.{second_param_name}"

                if param_name in params_instead_dict:
                    for target_param_name in params_instead_dict[param_name]:
                        params = setInputParamToTargetParam(
                            params=params,
                            input_param_name=param_name,
                            target_param_name=target_param_name,
                        )

    # ---- Third pass: suggest parameters whose range depends on other params ----
    for hyper_param_name, _ in params_use_range_dict.items():
        param_dict = getParamValueByParamName(hyperparameter_space_config, hyper_param_name)
        range_info_dict = param_dict["range"]

        if param_dict["search_type"] == "select":
            raise ValueError("The 'range' option is not supported for 'select' search type")

        min_diff = range_info_dict.get("min_diff", 0)
        max_diff = range_info_dict.get("max_diff", 0)

        if "min" in range_info_dict:
            min_value = getParamValueByParamName(params=params, hyper_param_name=range_info_dict["min"])
            param_dict["min"] = min_value + min_diff

        if "max" in range_info_dict:
            max_value = getParamValueByParamName(params=params, hyper_param_name=range_info_dict["max"])
            param_dict["max"] = max_value + max_diff

        params = setParamToTargetParam(
            params=params,
            target_param_name=hyper_param_name,
            input_param_value=suggestParams(  # type: ignore
                trial=trial,
                hyper_param_name=hyper_param_name,
                params_dict=param_dict,
            ),
        )

    return params
