from omegaconf import DictConfig, ListConfig

def setBestParamsToConfig(config: DictConfig | ListConfig, best_params: dict) -> DictConfig | ListConfig:
    config_best = config.copy()
    hyperparameter_space_config = config_best.args.optuna.hyperparameter_space

    param_path_dict = {}
    params_instead_dict = {}

    for param, value in best_params.items():
        param_list = param.split(".")

        if len(param_list) == 2:
            config_best["args"][param_list[0]][param_list[1]] = value  # type: ignore

        elif len(param_list) == 3:
            config_best["args"][param_list[0]][param_list[1]][param_list[2]] = value  # type: ignore

        param_path_dict[param] = param_list

    for root_param_name, root_space_dict in hyperparameter_space_config.items():
        for first_param_name, first_param_dict in root_space_dict.items():
            if "search_type" in first_param_dict:
                hyper_param_name = f"{root_param_name}.{first_param_name}"

                if "use_instead_of" in first_param_dict:
                    param_name_instead = first_param_dict.use_instead_of

                    if not len(param_name_instead.split(".")) > 1:
                        param_name_instead = f"{root_param_name}.{param_name_instead}"

                    params_instead_dict.setdefault(param_name_instead, [])
                    params_instead_dict[param_name_instead].append([root_param_name, first_param_name])

                else:
                    config_best["args"][root_param_name][first_param_name] = best_params[hyper_param_name]  # type: ignore

            else:
                for second_param_name, second_param_dict in first_param_dict.items():
                    hyper_param_name = f"{root_param_name}.{first_param_name}.{second_param_name}"

                    if "search_type" in second_param_dict:
                        if "use_instead_of" in second_param_dict:
                            param_name_instead = second_param_dict.use_instead_of

                            if not len(param_name_instead.split(".")) > 1:
                                param_name_instead = f"{root_param_name}.{first_param_name}.{param_name_instead}"

                            params_instead_dict.setdefault(param_name_instead, [])
                            params_instead_dict[param_name_instead].append([root_param_name, first_param_name, second_param_name])

                        else:
                            config_best["args"][root_param_name][first_param_name][second_param_name] = best_params[hyper_param_name]  # type: ignore

                    else:
                        raise Exception("Invalid parameter configuration")
    for root_param_name, root_space_dict in hyperparameter_space_config.items():
        for first_param_name, first_param_dict in root_space_dict.items():
            param_name = f"{root_param_name}.{first_param_name}"

            if param_name in params_instead_dict:
                for arg_path_list in params_instead_dict[param_name]:
                    if len(arg_path_list) == 2:
                        target_root_param_name, target_first_param_name = arg_path_list
                        config_best["args"][target_root_param_name][target_first_param_name] = config_best["args"][root_param_name][first_param_name]  # type: ignore

                    elif len(arg_path_list) == 3:
                        (
                            target_root_param_name,
                            target_first_param_name,
                            target_second_param_name,
                        ) = arg_path_list
                        config_best["args"][target_root_param_name][target_first_param_name][target_second_param_name] = config_best["args"][  # type: ignore
                            root_param_name
                        ][
                            first_param_name
                        ]

                    elif len(arg_path_list) == 1:
                        raise

            for second_param_name, second_param_dict in first_param_dict.items():
                param_name = f"{root_param_name}.{first_param_name}.{second_param_name}"

                if param_name in params_instead_dict:
                    for arg_path_list in params_instead_dict[param_name]:
                        if len(arg_path_list) == 2:
                            (
                                target_root_param_name,
                                target_first_param_name,
                            ) = arg_path_list
                            config_best["args"][target_root_param_name][target_first_param_name] = config_best["args"][root_param_name][  # type: ignore
                                first_param_name
                            ][
                                second_param_name
                            ]

                        elif len(arg_path_list) == 3:
                            (
                                target_root_param_name,
                                target_first_param_name,
                                target_second_param_name,
                            ) = arg_path_list
                            config_best["args"][target_root_param_name][target_first_param_name][target_second_param_name] = config_best["args"][  # type: ignore
                                root_param_name
                            ][
                                first_param_name
                            ][
                                second_param_name
                            ]

                        elif len(arg_path_list) == 1:
                            raise

    return config_best
