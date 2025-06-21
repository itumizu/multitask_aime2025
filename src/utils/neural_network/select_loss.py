import torch
from omegaconf import DictConfig

from src.etc.constants import (
    BINARY_CLASSIFICATION_LOSS_FUNCTION_LIST,
    MULTICLASS_CLASSIFICATION_LOSS_FUNCTION_LIST,
    REGRESSION_LOSS_FUNCTION_LIST,
    LOSS_FUNCTION_LIST,
)


def selectLoss(loss_function_name: str):
    if (
        not loss_function_name
        in BINARY_CLASSIFICATION_LOSS_FUNCTION_LIST
        + MULTICLASS_CLASSIFICATION_LOSS_FUNCTION_LIST
        + REGRESSION_LOSS_FUNCTION_LIST
        + LOSS_FUNCTION_LIST
    ):
        raise ValueError(f"Undefined Loss Function in this code: {loss_function_name}")

    if loss_function_name == "MSELoss":
        loss_function = torch.nn.MSELoss()

    elif loss_function_name == "BCEWithLogitsLoss":
        loss_function = torch.nn.BCEWithLogitsLoss()

    elif loss_function_name == "HuberLoss":
        loss_function = torch.nn.HuberLoss()

    elif loss_function_name == "CrossEntropyLoss":
        loss_function = torch.nn.CrossEntropyLoss()

    elif loss_function_name == "KLDivLoss":
        loss_function = torch.nn.KLDivLoss()

    elif loss_function_name == "BCELoss":
        loss_function = torch.nn.BCELoss()

    else:
        raise

    return loss_function
