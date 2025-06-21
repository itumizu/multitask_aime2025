from pathlib import Path

from omegaconf import DictConfig, ListConfig, OmegaConf

from src.models.catboost import prepareCatBoost
from src.models.light_gbm import prepareLightGBM
from src.models.random_forest import prepareRandomForest
from src.models.tabnet_multi_branch import TabNetMultiBranch
from src.utils.select_model.load_pretrained_weights import loadPretrainedWeights
from src.utils.select_model.use_normal_model import selectModelUseNormal
from src.loaders.dataset import HealthDataset
from src.models.xgboost import prepareXGBoost
from src.etc.constants import CATBOOST_MODELS, GRADIENT_BOOSTING_TREE_MODELS, TABNET_MODELS, RANDOM_FOREST_MODELS, LIGHTGBM_MODELS
from .use_multi_task_strategy import selectModelUseMultiTaskStrategy
from src.models.tabnet import TabNet


def selectModel(config: DictConfig | ListConfig, gpus: list, fold_test_num=0, fold_val_num=0):
    model = None
    model_name = config.args.models.name
    config.args.models.params.gpus = gpus

    if model_name in ["ResNet", "ResNetBranch"] and not "d_in" in config.args.models.params:
        dataset = HealthDataset(config=config, fold_test_num=fold_test_num, fold_val_num=fold_val_num, phase="test")
        config.args.models.params.d_in = len(dataset.use_columns)

    if model_name in ["TabNet", "TabNetMultiBranch"] and not "input_dim" in config.args.models.params:
        dataset = HealthDataset(config=config, fold_test_num=fold_test_num, fold_val_num=fold_val_num, phase="test")
        config.args.models.params.input_dim = len(dataset.use_columns)

    if "multi_task_strategy" in config.args.models and config.args.models.multi_task_strategy:
        model = selectModelUseMultiTaskStrategy(
            config=config, model_name=model_name, fold_test_num=fold_test_num, fold_val_num=fold_val_num, gpus=gpus
        )

    else:
        if model_name in GRADIENT_BOOSTING_TREE_MODELS:
            model = prepareXGBoost(config=config, gpus=gpus)

        elif model_name in CATBOOST_MODELS:
            model = prepareCatBoost(config=config)

        elif model_name in LIGHTGBM_MODELS:
            model = prepareLightGBM(config=config)

        elif model_name in RANDOM_FOREST_MODELS:
            model = prepareRandomForest(config=config, gpus=gpus)

        elif model_name == "TabNet":
            config_dataset = OmegaConf.load(Path(config.args.dataset.data_dir, "config.yaml"))
            config.args.models.params.n_cont_features = config_dataset.args.n_cont_features
            train_dataset = HealthDataset(config=config, fold_test_num=fold_test_num, fold_val_num=fold_val_num, phase="train")

            cat_dims = [value for key, value in config_dataset.args.cat_cardinalities.items()]
            cat_idxs = [idx for idx, column in enumerate(train_dataset.use_columns) if column in config_dataset.args.categorical_columns]

            config.args.models.params.cat_dims = cat_dims
            config.args.models.params.cat_idxs = cat_idxs

            model = TabNet(**config.args.models.params)

        elif model_name == "TabNetMultiBranch":
            config_dataset = OmegaConf.load(Path(config.args.dataset.data_dir, "config.yaml"))
            config.args.models.params.n_cont_features = config_dataset.args.n_cont_features
            train_dataset = HealthDataset(config=config, fold_test_num=fold_test_num, fold_val_num=fold_val_num, phase="train")

            cat_dims = [value for key, value in config_dataset.args.cat_cardinalities.items()]
            cat_idxs = [idx for idx, column in enumerate(train_dataset.use_columns) if column in config_dataset.args.categorical_columns]

            config.args.models.params.cat_dims = cat_dims
            config.args.models.params.cat_idxs = cat_idxs

            config_tasks = OmegaConf.create({})

            for task in config.args.tasks.targets:
                # 条件に基づいてキーを抽出
                if f"params_{task}" in config.args.models:
                    config_tasks[f"params_{task}"] = config.args.models[f"params_{task}"]

            params_shared = config.args.models.params
            model = TabNetMultiBranch(params_shared=params_shared, params_task=config_tasks, **config.args.models.params)

        else:
            model = selectModelUseNormal(config=config, model_name=model_name, fold_test_num=fold_test_num, fold_val_num=fold_val_num, gpus=gpus)

        # 事前学習済みの重みがある場合は、部分的でもいいので読み込むようにする
        if "pretraind_weights_path" in config.args.models:
            model = loadPretrainedWeights(model=model, config=config, fold_test_num=fold_test_num, fold_val_num=fold_val_num)

    print(model)

    return model
