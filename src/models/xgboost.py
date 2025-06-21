import xgboost as xgb
from xgboost import XGBRegressor, XGBClassifier
from omegaconf import DictConfig, ListConfig
from typing import Optional

import xgboost


def prepareXGBoost(config: DictConfig | ListConfig, gpus: list) -> xgboost.XGBModel:
    model_name = config.args.models.name
    device = "cpu"
    tree_method = "hist"

    if len(gpus) == 1:
        device = f"cuda:{gpus[0]}"

    elif len(gpus) > 1:
        raise

    if model_name == "XGBClassifier":
        model = XGBClassifier(
            objective=config.args.models.params.objective,
            random_state=config.args.seed,
            n_estimators=config.args.models.params.n_estimators,
            max_depth=config.args.models.params.max_depth,
            learning_rate=config.args.models.params.learning_rate,
            min_child_weight=config.args.models.params.min_child_weight,
            subsample=config.args.models.params.subsample,
            colsample_bylevel=config.args.models.params.col_sample_by_level,
            colsample_bytree=config.args.models.params.col_sample_by_tree,
            gamma=config.args.models.params.gamma,
            reg_lambda=config.args.models.params.reg_lambda,
            alpha=config.args.models.params.alpha,
            tree_method=tree_method,
            device=device,
            multi_strategy="multi_output_tree" if len(config.args.tasks.targets) > 1 else "one_output_per_tree",
            nthread=1,
        )

    elif model_name == "XGBRegressor":
        model = XGBRegressor(
            objective=config.args.models.params.objective,
            random_state=config.args.seed,
            n_estimators=config.args.models.params.n_estimators,
            max_depth=config.args.models.params.max_depth,
            learning_rate=config.args.models.params.learning_rate,
            min_child_weight=config.args.models.params.min_child_weight,
            subsample=config.args.models.params.subsample,
            colsample_bylevel=config.args.models.params.col_sample_by_level,
            colsample_bytree=config.args.models.params.col_sample_by_tree,
            gamma=config.args.models.params.gamma,
            reg_lambda=config.args.models.params.reg_lambda,
            alpha=config.args.models.params.alpha,
            tree_method=tree_method,
            multi_strategy="multi_output_tree" if len(config.args.tasks.targets) > 1 else "one_output_per_tree",
        )

    else:
        raise

    model.set_params(
        # device="cpu",
        eval_metric=config.args.models.params.eval_metric,
        early_stopping_rounds=config.args.callbacks.early_stopping.patience if config.args.callbacks.early_stopping.use_callback else 0,
        callbacks=[],
    )

    return model
