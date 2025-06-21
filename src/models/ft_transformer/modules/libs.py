from asyncio import tasks
import itertools
import math
import typing
import warnings
from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, cast
from omegaconf import DictConfig
from prefect import task

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch import Tensor, Value, tensor
from torch.nn.parameter import Parameter

_INTERNAL_ERROR = "Internal error"
_TransformerFFNActivation = Literal["ReLU", "ReGLU"]
_LINFORMER_KV_COMPRESSION_SHARING = Literal["headwise", "key-value"]
_FORWARD_BAD_ARGS_MESSAGE = "Based on the arguments passed to the constructor of FTTransformer, {}"


def _named_sequential(*modules) -> nn.Sequential:
    return nn.Sequential(OrderedDict(modules))
