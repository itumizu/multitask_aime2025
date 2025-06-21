REGRESSION_LOSS_FUNCTION_LIST = ["MSELoss", "HuberLoss"]
BINARY_CLASSIFICATION_LOSS_FUNCTION_LIST = ["BCEWithLogitsLoss", "BCELoss"]
MULTICLASS_CLASSIFICATION_LOSS_FUNCTION_LIST = ["CrossEntropyLoss"]
LOSS_FUNCTION_LIST = ["KLDivLoss"]


GRADIENT_BOOSTING_TREE_MODELS = ["XGBClassifier", "XGBRegressor"]
GRADIENT_BOOSTING_TREE_CLASSIFICATION_MODELS = ["XGBClassifier"]
CATBOOST_MODELS = ["CatBoostClassifier", "CatBoostRegressor"]
LIGHTGBM_MODELS = ["LGBMClassifier", "LGBMRegressor"]

# 評価指標名
METRICS_FOR_REGRESSION_LIST = ["rmse", "mae", "r2_score", "mse"]
METRICS_FOR_CLASSIFICATION_LIST = ["acc", "recall", "precision", "auroc", "mcc"]
TRANSFORMER_BASED_MODELS = [
    "FTTransformer",
    "FTMultiTransformer",
    "FTTransformerMultiBranches",
]

NEURAL_NETWORK_MODELS = TRANSFORMER_BASED_MODELS