from omegaconf import DictConfig, ListConfig
from typing import Optional

import catboost
from catboost import CatBoostClassifier, CatBoostRegressor


def prepareCatBoost(config: DictConfig | ListConfig) -> catboost.CatBoostRegressor | catboost.CatBoostClassifier:
    model_name = config.args.models.name

    if model_name == "CatBoostClassifier":
        model = CatBoostClassifier(
            objective=config.args.models.params.objective,
            random_state=config.args.seed,
            n_estimators=config.args.models.params.n_estimators,
            max_depth=config.args.models.params.max_depth,
            learning_rate=config.args.models.params.learning_rate,
            bagging_temperature=config.args.models.params.bagging_temperature,
            l2_leaf_reg=config.args.models.params.l2_leaf_reg,
            leaf_estimation_iterations=config.args.models.params.leaf_estimation_iterations,
            eval_metric=config.args.models.params.eval_metric,
            od_pval=config.args.models.params.od_pval,
            thread_count=16,
        )

    elif model_name == "CatBoostRegressor":
        model = CatBoostRegressor(
            objective=config.args.models.params.objective,
            random_state=config.args.seed,
            n_estimators=config.args.models.params.n_estimators,
            max_depth=config.args.models.params.max_depth,
            learning_rate=config.args.models.params.learning_rate,
            bagging_temperature=config.args.models.params.bagging_temperature,
            l2_leaf_reg=config.args.models.params.l2_leaf_reg,
            leaf_estimation_iterations=config.args.models.params.leaf_estimation_iterations,
            eval_metric=config.args.models.params.eval_metric,
            od_pval=config.args.models.params.od_pval,
            thread_count=1,
        )

    else:
        raise

    return model
