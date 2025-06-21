import re
from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from omegaconf import ListConfig, OmegaConf, DictConfig


class HealthDataset(Dataset):
    def __init__(self, config: DictConfig | ListConfig, fold_test_num: int, fold_val_num: int, phase: str):
        super().__init__()
        self.config = config
        self.data_dir = config.args.dataset.data_dir
        self.phase = phase

        self.fold_test_num = fold_test_num
        self.fold_val_num = fold_val_num
        self.config_dataset = OmegaConf.load(Path(self.data_dir, "config.yaml"))

        if "schema_file_path" in self.config_dataset.args and Path(self.config_dataset.args.schema_file_path).exists():
            self.schema = pd.read_csv(Path(self.config_dataset.args.schema_file_path), index_col=0, header=0)
        else:
            self.schema = None

        self.prepare_data()

    def prepare_data(self):
        data_type = ""
        label_type = ""

        self.un_used_columns = []
        self.is_splited_into_cont_and_cat = False
        self.use_one_hot = False
        self.use_standardization_label = False
        self.is_splited_use_columns = False
        self.use_input_year_column = False

        self.file_extension = ".csv"

        if "file_extension" in self.config_dataset.args:
            self.file_extension = self.config_dataset.args.file_extension

        if "is_splited_into_cont_and_cat" in self.config.args.dataset:
            self.is_splited_into_cont_and_cat = self.config.args.dataset.is_splited_into_cont_and_cat

        if "use_one_hot" in self.config.args.dataset:
            self.use_one_hot = self.config.args.dataset.use_one_hot

        if "un_used_columns" in self.config.args.dataset:
            self.un_used_columns = self.config.args.dataset.un_used_columns

        if "use_standardization_label" in self.config.args.dataset:
            self.use_standardization_label = self.config.args.dataset.use_standardization_label

        if "is_splited_use_columns" in self.config.args.dataset:
            self.is_splited_use_columns = self.config.args.dataset.is_splited_use_columns

        if "use_input_year_column" in self.config.args.dataset:
            self.use_input_year_column = self.config.args.dataset.use_input_year_column

        if self.use_input_year_column:
            self.un_used_columns.extend(self.config.args.tasks.targets)

        self.x_not_std = pd.read_csv(
            Path(self.data_dir, f"fold_test_{self.fold_test_num}", f"fold_val_{self.fold_val_num}", f"{self.phase}", f"data_{self.phase}{self.file_extension}"),
            index_col=0,
        )

        if self.use_one_hot:
            data_type += "_one_hot"

        if self.config.args.dataset.use_standardization:
            data_type += "_std"

        if self.use_standardization_label:
            label_type += "_std"

        self.identifier_columns = self.config_dataset.args.get("identifier_columns", [])
        use_combined_h5_file = self.config_dataset.args.get("use_combined_h5_file", False)

        if not use_combined_h5_file:
            data_path = Path(self.data_dir, f"fold_test_{self.fold_test_num}", f"fold_val_{self.fold_val_num}", f"{self.phase}", f"data{data_type}_{self.phase}{self.file_extension}")
            label_path = Path(self.data_dir, f"fold_test_{self.fold_test_num}", f"fold_val_{self.fold_val_num}", f"{self.phase}", f"label{label_type}_{self.phase}{self.file_extension}")

            if self.file_extension == ".csv":
                self.x = pd.read_csv(data_path, index_col=0)
                self.y = pd.read_csv(label_path, index_col=0)
            elif self.file_extension == ".npy":
                self.x = pd.DataFrame(np.load(data_path))
                self.y = pd.DataFrame(np.load(label_path))
            else:
                raise ValueError("Unsupported file format")
        else:
            path_data_labels = Path(self.data_dir, f"fold_test_{self.fold_test_num}", f"fold_val_{self.fold_val_num}", "data_labels.h5")
            with h5py.File(path_data_labels, "r") as f:
                self.x_cont = pd.DataFrame(f[f"X_num_{self.phase}"][:])
                self.x_cat = pd.DataFrame(f[f"X_cat_{self.phase}"][:])
                self.y = pd.DataFrame(f[f"y_{self.phase}"][:])

                self.columns_x_cont = list(map(lambda x: x.decode(), f["columns_num"][:]))
                self.columns_x_cat = list(map(lambda x: x.decode(), f["columns_cat"][:]))

            self.y.columns = self.config_dataset.args.label_names

            if self.x_cont.shape[0] > 0:
                self.x_cont.columns = self.columns_x_cont
            if self.x_cat.shape[0] > 0:
                self.x_cat.columns = self.columns_x_cat

            if self.x_cont.shape[0] > 0 and self.x_cat.shape[0] > 0:
                self.x = pd.concat([self.x_cont, self.x_cat], axis=1)
            elif self.x_cont.shape[0] > 0:
                self.x = self.x_cont
            elif self.x_cat.shape[0] > 0:
                self.x = self.x_cat

        self.use_columns = [col for col in self.x.columns if col not in self.identifier_columns and col not in self.un_used_columns]
        self.categorical_columns = self.config_dataset.args.categorical_columns
        self.continuous_columns = [col for col in self.use_columns if col not in self.categorical_columns]

        if self.phase == "train" and "train_size_ratio" in self.config.args.dataset:
            train_size_ratio = self.config.args.dataset.train_size_ratio

            if train_size_ratio <= 0.0 or train_size_ratio > 1.0:
                raise ValueError(f"Train data ratio is invalid: {train_size_ratio}")

            self.x = self.x.sample(frac=train_size_ratio, random_state=self.config.args.seed)
            self.y = self.y.loc[self.x.index, :]

            if self.x.index.tolist() != self.y.index.tolist():
                raise ValueError("Index mismatch between x and y")

        self.columns_dict = {}

        if self.is_splited_use_columns:
            self.columns_dict = self.getSeparatedColumns()

    def getDataset(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        targets = []
        for target_name in self.config.args.tasks.targets:
            target_name = re.sub("_SAME.*", "", target_name)
            if target_name not in targets:
                targets.append(target_name)
        return self.x.loc[:, self.use_columns], self.y.loc[:, targets]

    def getSeparatedColumns(self) -> dict:
        columns_dict = {"main": {"x_cont": [], "x_cat": []}}
        use_continuous_columns = []
        use_categorical_columns = []

        for task_name in self.config.args.tasks.targets:
            if f"params_{task_name}" not in self.config.args.models:
                raise ValueError("Missing per-task parameter configuration.")

            task_params = self.config.args.models[f"params_{task_name}"]
            columns_dict[task_name] = {"x_cont": [], "x_cat": []}

            if "use_columns" in task_params:
                for col in task_params.use_columns:
                    if col in self.continuous_columns:
                        columns_dict[task_name]["x_cont"].append(col)
                        use_continuous_columns.append(col)
                    elif col in self.categorical_columns:
                        columns_dict[task_name]["x_cat"].append(col)
                        use_categorical_columns.append(col)
                    else:
                        raise ValueError("Specified column not found.")
            else:
                columns_dict[task_name]["x_cont"].extend(self.continuous_columns)
                columns_dict[task_name]["x_cat"].extend(self.categorical_columns)

        if "use_columns" in self.config.args.models.params:
            if self.config.args.models.params.use_columns == "others":
                columns_dict["main"]["x_cont"] = [col for col in self.continuous_columns if col not in use_continuous_columns]
                columns_dict["main"]["x_cat"] = [col for col in self.categorical_columns if col not in use_categorical_columns]
            else:
                columns_dict["main"]["x_cont"].extend(self.continuous_columns)
                columns_dict["main"]["x_cat"].extend(self.categorical_columns)
        else:
            raise ValueError("Please specify which attributes to use.")

        return columns_dict

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return_dict = {}
        index_name = self.y.index[idx]
        return_dict["index_name"] = str(index_name)

        if not self.is_splited_use_columns:
            if not self.is_splited_into_cont_and_cat:
                x = self.x.loc[index_name, self.use_columns].values.astype(np.float32)
                x = torch.tensor(x, dtype=torch.float32)
                return_dict["x"] = x
            else:
                return_dict["x_cont"] = torch.tensor(self.x.loc[index_name, self.continuous_columns].values.astype(np.float32), dtype=torch.float32)
                if len(self.categorical_columns):
                    return_dict["x_cat"] = torch.tensor(self.x.loc[index_name, self.categorical_columns].values.astype(np.float32), dtype=torch.float32)
                else:
                    return_dict["x_cat"] = torch.tensor([])
        else:
            for task_name, task_columns_dict in self.columns_dict.items():
                return_dict[f"x_cont_{task_name}"] = torch.tensor(self.x.loc[index_name, task_columns_dict["x_cont"]].values.astype(np.float32), dtype=torch.float32)
                x_cat_values = self.x.loc[index_name, task_columns_dict["x_cat"]].values
                if x_cat_values.shape[1] > 0:
                    return_dict[f"x_cat_{task_name}"] = torch.tensor(x_cat_values.astype(np.float32), dtype=torch.float32)
                else:
                    return_dict[f"x_cat_{task_name}"] = None

        for target_name in self.config.args.tasks.targets:
            if self.use_input_year_column:
                y = self.x_not_std.loc[index_name, re.sub("_SAME.*", "", target_name)]
            else:
                y = self.y.loc[index_name, re.sub("_SAME.*", "", target_name)]
            y_true = torch.tensor([y], dtype=torch.float32)
            return_dict[f"y_{target_name}"] = y_true

        return return_dict
