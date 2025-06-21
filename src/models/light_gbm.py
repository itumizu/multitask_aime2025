from omegaconf import DictConfig, ListConfig
from lightgbm import LGBMClassifier, LGBMRegressor


def prepareLightGBM(config: DictConfig | ListConfig) -> LGBMClassifier | LGBMRegressor:
    model_name = config.args.models.name

    if model_name == "LGBMClassifier":
        model = LGBMClassifier(
            objective=config.args.models.params.objective,
            random_state=config.args.seed,
            n_estimators=config.args.models.params.n_estimators,
            max_depth=config.args.models.params.max_depth,
            learning_rate=config.args.models.params.learning_rate,
            min_child_weight=config.args.models.params.min_child_weight,
            subsample=config.args.models.params.subsample,
            colsample_bytree=config.args.models.params.col_sample_by_tree,
            min_split_gain=config.args.models.params.min_split_gain,
            reg_lambda=config.args.models.params.reg_lambda,
            alpha=config.args.models.params.alpha,
            n_jobs=1,
            # verbose=0,
        )

    elif model_name == "LGBMRegressor":
        model = LGBMRegressor(
            objective=config.args.models.params.objective,
            random_state=config.args.seed,
            n_estimators=config.args.models.params.n_estimators,
            max_depth=config.args.models.params.max_depth,
            learning_rate=config.args.models.params.learning_rate,
            min_child_weight=config.args.models.params.min_child_weight,
            subsample=config.args.models.params.subsample,
            colsample_bytree=config.args.models.params.col_sample_by_tree,
            min_split_gain=config.args.models.params.min_split_gain,
            reg_lambda=config.args.models.params.reg_lambda,
            alpha=config.args.models.params.alpha,
            n_jobs=1,
        )

    else:
        raise

    return model
