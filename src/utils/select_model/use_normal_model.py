import re
from pathlib import Path
from omegaconf import OmegaConf, ListConfig, DictConfig

from src.models.ft_transformer import (
    FTTransformer,
    FTMultiTransformer,
)
from src.models.ft_transformer.ft_transformer_multi_branches import FTTransformerMultiBranches
from src.loaders.dataset import HealthDataset
from src.etc.constants import TRANSFORMER_BASED_MODELS


def selectModelUseNormal(config: DictConfig | ListConfig, model_name: str, fold_test_num: int, fold_val_num: int, gpus: list):
    if config.args.dataset.is_splited_into_cont_and_cat:
        config_dataset = OmegaConf.load(Path(config.args.dataset.data_dir, "config.yaml"))
        config.args.models.params.n_cont_features = config_dataset.args.n_cont_features
        config.args.models.params.cat_cardinalities = [
            value for key, value in config_dataset.args.cat_cardinalities.items()
        ]

    if model_name in TRANSFORMER_BASED_MODELS:
        train_dataset = HealthDataset(config=config, fold_test_num=fold_test_num, fold_val_num=fold_val_num, phase="train")

        if len(train_dataset.continuous_columns) != config.args.models.params.n_cont_features:
            config.args.models.params.n_cont_features = len(train_dataset.continuous_columns)

        if len(train_dataset.categorical_columns) != len(config.args.models.params.cat_cardinalities):
            config_dataset = OmegaConf.load(Path(config.args.dataset.data_dir, "config.yaml"))
            config.args.models.params.cat_cardinalities = [
                value for key, value in config_dataset.args.cat_cardinalities.items()
                if key in train_dataset.categorical_columns
            ]

        for params_name, params in config.args.models.items():
            if not isinstance(params, (int, float)) and "ffn_d_hidden_multiplier" in params:
                if isinstance(params.ffn_d_hidden_multiplier, str):
                    split_list = re.sub(r"\s+", "", params.ffn_d_hidden_multiplier).split("/")
                    if len(split_list) == 2:
                        config.args.models[params_name]["ffn_d_hidden_multiplier"] = float(split_list[0]) / float(split_list[1])

        if model_name == "FTTransformer":
            model = FTTransformer(**config.args.models.params)

        elif model_name == "FTTransformerMultiBranches":
            config_tasks = OmegaConf.create({})
            config_middle = OmegaConf.create({})

            for task in config.args.tasks.targets:
                if f"params_{task}" in config.args.models:
                    config_tasks[f"params_{task}"] = config.args.models[f"params_{task}"]

                if f"params_middle_{task}" in config.args.models:
                    config_middle[f"params_middle_{task}"] = config.args.models[f"params_middle_{task}"]

            params_shared = config.args.models.params
            params_middle = config_middle if len(config_middle) > 0 else None

            model = FTTransformerMultiBranches(params_shared=params_shared, params_middle=params_middle, params_task=config_tasks)

        elif model_name == "FTMultiTransformer":
            config_tasks = OmegaConf.create({})
            for task in config.args.tasks.targets[1:]:
                if f"params_{task}" in config.args.models:
                    config_tasks[f"params_{task}"] = config.args.models[f"params_{task}"]

            model = FTMultiTransformer(tasks_params=config_tasks, **config.args.models.params)

        else:
            raise ValueError(f"Unsupported model name: {model_name}")

    else:
        raise ValueError(f"Model name {model_name} is not a supported transformer model.")

    return model
