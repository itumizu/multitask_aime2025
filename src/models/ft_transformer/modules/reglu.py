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


class _ReGLU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:

        if x.shape[-1] % 2:
            raise ValueError("For the ReGLU activation, the last input dimension" f" must be a multiple of 2, however: {x.shape[-1]=}")

        a, b = x.chunk(2, dim=-1)

        return a * F.relu(b)
