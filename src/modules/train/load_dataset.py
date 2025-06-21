from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from omegaconf import OmegaConf, DictConfig


def loadData(config: DictConfig, fold_test_num: int, fold_val_num: int, phase: str) -> dict:
    data_type = ""
    data_dir = config.args.dataset.data_dir

    config_dataset = OmegaConf.load(Path(config.args.dataset.data_dir, "config.yaml"))

    if config.args.dataset.use_standardization:
        data_type = "_std"

    data_path = Path(
        data_dir,
        f"fold_test_{fold_test_num}",
        f"fold_val_{fold_val_num}",
        f"{phase}",
        f"data{data_type}_{phase}.csv",
    )

    label_path = Path(
        data_dir,
        f"fold_test_{fold_test_num}",
        f"fold_val_{fold_val_num}",
        f"{phase}",
        f"label_{phase}.csv",
    )

    X = pd.read_csv(data_path, index_col=0)
    y = pd.read_csv(label_path, index_col=0)

    identifier_columns = config_dataset.args.identifier_columns
    categorical_columns = config_dataset.args.categorical_columns

    use_columns = [column for column in X.columns if not column in identifier_columns + categorical_columns]
    # use_columns.append("SUDOS_SAKE")

    X = X.loc[:, use_columns]
    y = y.loc[:, config.args.tasks.targets]

    return {"data": X, "label": y}


def loadDataset(config: DictConfig, fold_test_num: int, fold_val_num: int):
    data_label_dict = {}

    data_label_dict_train = loadData(config, fold_test_num, fold_val_num, "train")
    data_label_dict_val = loadData(config, fold_test_num, fold_val_num, "val")
    data_label_dict_test = loadData(config, fold_test_num, fold_val_num, "test")

    data_label_dict["train"] = data_label_dict_train
    data_label_dict["val"] = data_label_dict_val
    data_label_dict["test"] = data_label_dict_test

    return data_label_dict


def loadDatasetConfig(config: DictConfig, columns: list):
    config_dataset = OmegaConf.load(Path(config.args.dataset.data_dir, "config.yaml"))
    cat_cardinalities_dict = config_dataset.args.cat_cardinalities

    cat_idxs = []
    cat_dims = []

    for idx, column in enumerate(columns):
        if column in config_dataset.args.categorical_columns:
            cat_idxs.append(idx)
            cat_dims.append(cat_cardinalities_dict[column])

        else:
            continue

    return config_dataset, cat_idxs, cat_dims
