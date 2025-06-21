import itertools
from typing import Any, Dict, Iterable, List, Optional, Tuple, cast
from omegaconf import DictConfig

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter

from src.models.ft_transformer.modules.backbone_branches import FTTransformerMultiBranchesBackbone
from src.models.ft_transformer.modules.bottleneck_block_one_block import BottleneckBlockMultiTaskOneBlock
from src.models.ft_transformer.modules.bottleneck_block_two_task import BottleneckBlockMultiTask

from .modules.embedding import _CLSEmbedding, LinearEmbeddings, CategoricalEmbeddings
from .modules.libs import _INTERNAL_ERROR, _FORWARD_BAD_ARGS_MESSAGE


class FTTransformerMultiBranches(nn.Module):
    def __init__(
        self,
        params_shared: DictConfig,
        params_task: DictConfig,
        params_middle: DictConfig | None = None,
        _is_default: bool = False,
    ) -> None:
        """
        Args:
            n_cont_features: the number of continuous features.
            cat_cardinalities: the cardinalities of categorical features.
                Pass en empty list if there are no categorical features.
            _is_default: this is a technical argument, don't set it manually.
            params_shared: the keyword arguments for the `FTTransformerBackbone`.
        """
        n_cont_features = params_shared["n_cont_features"]
        cat_cardinalities = params_shared["cat_cardinalities"]

        if n_cont_features < 0:
            raise ValueError(f"n_cont_features must be non-negative, however: {n_cont_features=}")

        if n_cont_features == 0 and not cat_cardinalities:
            raise ValueError("At least one type of features must be presented, however:" f" {n_cont_features=}, {cat_cardinalities=}")

        if "n_tokens" in params_shared:
            raise ValueError('params_shared must not contain key "n_tokens"' " (the number of tokens will be inferred automatically)")

        super().__init__()

        d_block: int = params_shared["d_block"]
        self.device = torch.device(f"cuda:{params_shared['gpus'][0]}")

        self.params_task = params_task
        self.n_cont_features = n_cont_features
        self.cls_embedding = _CLSEmbedding(d_block)

        # >>> Feature embeddings (Figure 2a in the paper).
        self.cont_embeddings = LinearEmbeddings(n_cont_features, d_block) if n_cont_features > 0 else None
        self.cat_embeddings = CategoricalEmbeddings(cat_cardinalities, d_block, True) if cat_cardinalities else None
        # <<<

        # for shared layers
        self.backbone = FTTransformerMultiBranchesBackbone(
            **params_shared,
            n_tokens=(None if params_shared.get("linformer_kv_compression_ratio") is None else 1 + n_cont_features + len(cat_cardinalities)),
            is_hidden_module=True,
            d_out=None,
        )

        self.params_middle = None
        self.backbone_middle = None

        if params_middle:
            self.middle_backbone_dict = nn.ModuleDict()
            self.task_bottleneck_block_dict = nn.ModuleDict()

            self.params_middle = params_middle

            # for middle block in task branches
            for task_name, task_params in params_middle.items():
                task_name = task_params.task_name

                self.middle_backbone_dict[task_name] = FTTransformerMultiBranchesBackbone(
                    **task_params,
                    is_hidden_module=True,
                    is_sub_task=True,
                    n_tokens=(None if task_params.get("linformer_kv_compression_ratio") is None else 1 + n_cont_features + len(cat_cardinalities)),
                    d_out=None,
                )

        self.task_backbone_dict = nn.ModuleDict()

        # for task branches
        for task_name, task_params in params_task.items():
            task_name = task_params.task_name

            self.task_backbone_dict[task_name] = FTTransformerMultiBranchesBackbone(
                **task_params,
                is_sub_task=True,
                n_tokens=(None if task_params.get("linformer_kv_compression_ratio") is None else 1 + n_cont_features + len(cat_cardinalities)),
            )

        self.output = None

        self._is_default = _is_default
        self.bottleneck_block_middle = None
        self.task_bottleneck_block = None

        self.n_heads_common_in_bottleneck = params_shared.get("n_heads_common_in_bottleneck", None)

        if params_middle:
            if params_shared.get("add_bottleneck", ""):
                ffn_use_reglu = True
                ffn_d_hidden = int(d_block * cast(float, params_shared.ffn_d_hidden_multiplier))
                ffn_dropout = 0.0
                is_reversed = params_shared.get("is_reversed", True)

                self.bottleneck_block_middle = BottleneckBlockMultiTask(
                    d_block=d_block,
                    n_tokens=(None if task_params.get("linformer_kv_compression_ratio") is None else 1 + n_cont_features + len(cat_cardinalities)),
                    ffn_use_reglu=ffn_use_reglu,
                    ffn_d_hidden=ffn_d_hidden,
                    ffn_dropout=ffn_dropout,
                    mask_attention_n_heads=params_shared.n_heads_in_bottleneck,
                    is_reversed=is_reversed,
                    use_sparsemax=params_shared.get("use_sparsemax", False),
                    bottleneck_weights=params_shared.get("bottleneck_weights", [1.0, 1.0]),
                )

        if params_shared.get("add_bottleneck_task", ""):
            task_name = task_params.task_name
            is_reversed = params_shared.get("is_reversed", True)

            ffn_use_reglu = True
            ffn_d_hidden = int(d_block * cast(float, params_shared.ffn_d_hidden_multiplier))
            ffn_dropout = 0.0

            if params_shared.get("bottleneck_type", "") == "one_block":
                self.task_bottleneck_block = BottleneckBlockMultiTaskOneBlock(
                    d_block=d_block,
                    n_tokens=(None if task_params.get("linformer_kv_compression_ratio") is None else 1 + n_cont_features + len(cat_cardinalities)),
                    ffn_use_reglu=ffn_use_reglu,
                    ffn_d_hidden=ffn_d_hidden,
                    ffn_dropout=ffn_dropout,
                    mask_attention_n_heads=params_shared.n_heads_in_bottleneck,
                    is_reversed=is_reversed,
                )

            elif params_shared.get("bottleneck_type", "") == "two_block":
                self.task_bottleneck_block = BottleneckBlockMultiTask(
                    d_block=d_block,
                    n_tokens=(None if task_params.get("linformer_kv_compression_ratio") is None else 1 + n_cont_features + len(cat_cardinalities)),
                    ffn_use_reglu=ffn_use_reglu,
                    ffn_d_hidden=ffn_d_hidden,
                    ffn_dropout=ffn_dropout,
                    mask_attention_n_heads=params_shared.n_heads_in_bottleneck,
                    mask_common_attention_n_heads=params_shared.get("n_heads_common_in_bottleneck", None),
                    use_task_bottleneck_block=params_shared.get("use_task_bottleneck_block", True),
                    is_reversed=is_reversed,
                )

            else:
                raise

    def make_parameter_groups(self) -> List[Dict[str, Any]]:
        """Make parameter groups for optimizers.

        The difference with calling this method instead of
        `.parameters()` is that this method always sets `weight_decay=0.0`
        for some of the parameters.

        Returns:
            the parameter groups that can be passed to PyTorch optimizers.
        """

        def get_parameters(m: Optional[nn.Module]) -> Iterable[Parameter]:
            return () if m is None else m.parameters()

        zero_wd_group: Dict[str, Any] = {
            "params": set(
                itertools.chain(
                    get_parameters(self.cls_embedding),
                    get_parameters(self.cont_embeddings),
                    get_parameters(self.cat_embeddings),
                    # normalization
                    (p for name, p in self.named_parameters() if "normalization" in name),
                    # bias
                    (p for name, p in self.named_parameters() if name.endswith(".bias")),
                )
            ),
            "weight_decay": 0.0,
        }

        main_group: Dict[str, Any] = {"params": [p for p in self.parameters() if p not in zero_wd_group["params"]]}
        zero_wd_group["params"] = list(zero_wd_group["params"])
        return [main_group, zero_wd_group]

    _FORWARD_BAD_ARGS_MESSAGE = "Based on the arguments passed to the constructor of FTTransformer, {}"

    def forward(self, x_cont, x_cat) -> Tuple[Tensor, dict]:
        """Do the forward pass."""
        x_any = x_cat if x_cont is None else x_cont

        if x_any is None:
            raise ValueError("At least one of x_cont and x_cat must be provided.")

        if x_cat.size(1) == 0:
            x_cat = None

        x_embeddings: List[Tensor] = []

        if self.cls_embedding is not None:
            x_embeddings.append(self.cls_embedding(x_any.shape[:-1]))

        for argname, argvalue, module in [
            ("x_cont", x_cont, self.cont_embeddings),
            ("x_cat", x_cat, self.cat_embeddings),
        ]:
            if module is None:
                if argvalue is not None:
                    raise ValueError(_FORWARD_BAD_ARGS_MESSAGE.format(f"{argname} must be None"))
            else:
                if argvalue is None:
                    raise ValueError(_FORWARD_BAD_ARGS_MESSAGE.format(f"{argname} must not be None"))

                x_embeddings.append(module(argvalue))

        assert x_embeddings, _INTERNAL_ERROR

        x = torch.cat(x_embeddings, dim=1)

        # common
        x_common, attention_shared_dict = self.backbone(x)
        output_attention_dict = {"shared": attention_shared_dict}

        x_middle_task_dict = {}

        # task_specific
        if self.params_middle:
            for task_name, task_params in self.params_middle.items():
                task_name = task_params.task_name
                x_middle, attention_middle_dict = self.middle_backbone_dict[task_name](x_common)

                x_middle_task_dict[task_name] = x_middle
                output_attention_dict[task_name] = attention_middle_dict

        else:
            for task_params in self.params_task.values():
                task_name = task_params.task_name
                x_middle_task_dict[task_name] = x_common

        if self.bottleneck_block_middle:
            x_middle_bottleneck_block = self.bottleneck_block_middle(*x_middle_task_dict.values())

            for key, value in zip(self.params_middle.keys(), x_middle_bottleneck_block):
                x_middle_task_dict[key] = value

        x_tasks = []
        x_middle_bottleneck_block_dict = {}

        if self.task_bottleneck_block:
            if self.params_middle:
                middle_output = self.task_bottleneck_block(*x_middle_task_dict.values())

            else:
                middle_output = self.task_bottleneck_block(x_common, x_common)

            x_middle_bottleneck_block, attention_bottleneck_dict = middle_output[0:2], middle_output[2]

            task_name_list = list(self.params_task.keys())

            if self.n_heads_common_in_bottleneck:
                task_name_list.append("common")

            for task_name, attention_dict in zip(task_name_list, attention_bottleneck_dict.values()):
                output_attention_dict[f"BottleNeck_{task_name}"] = attention_dict

            for params, value in zip(self.params_task.values(), x_middle_bottleneck_block):
                x_middle_bottleneck_block_dict[params.task_name] = value

        for task_name, task_params in self.params_task.items():
            task_name = task_params.task_name
            middle_block_name = task_params.get("middle_block_name", "")

            if task_name in x_middle_bottleneck_block_dict:
                x_input = x_middle_bottleneck_block_dict[task_name]

            elif middle_block_name in x_middle_task_dict:
                x_input = x_middle_task_dict[middle_block_name]

            else:
                x_input = x_common

            x_task, attention_task_dict = self.task_backbone_dict[task_name](x_input)
            x_tasks.append(x_task)

            output_attention_dict[task_name] = attention_task_dict

        x = torch.cat(x_tasks, dim=1)

        return x, output_attention_dict
