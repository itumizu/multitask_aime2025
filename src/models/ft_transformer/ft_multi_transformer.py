import itertools
import warnings
from typing import Any, Dict, Iterable, List, Optional, Tuple
from omegaconf import DictConfig

import torch
import torch.nn as nn
import torch.optim
from torch import Tensor
from torch.nn.parameter import Parameter

from .modules.embedding import _CLSEmbedding, LinearEmbeddings, CategoricalEmbeddings
from .modules.backbone_multi import FTMultiTransformerBackbone
from .modules.libs import _named_sequential, _INTERNAL_ERROR, _FORWARD_BAD_ARGS_MESSAGE


class FTMultiTransformer(nn.Module):
    def __init__(
        self,
        *,
        tasks_params: DictConfig,
        n_cont_features: int,
        cat_cardinalities: List[int],
        _is_default: bool = False,
        **backbone_kwargs,
    ) -> None:
        """
        Args:
            n_cont_features: the number of continuous features.
            cat_cardinalities: the cardinalities of categorical features.
                Pass en empty list if there are no categorical features.
            _is_default: this is a technical argument, don't set it manually.
            backbone_kwargs: the keyword arguments for the `FTTransformerBackbone`.
        """
        if n_cont_features < 0:
            raise ValueError(f"n_cont_features must be non-negative, however: {n_cont_features=}")

        if n_cont_features == 0 and not cat_cardinalities:
            raise ValueError("At least one type of features must be presented, however:" f" {n_cont_features=}, {cat_cardinalities=}")

        if "n_tokens" in backbone_kwargs:
            raise ValueError('backbone_kwargs must not contain key "n_tokens"' " (the number of tokens will be inferred automatically)")

        super().__init__()

        d_block: int = backbone_kwargs["d_block"]
        self.device = torch.device(f"cuda:{backbone_kwargs['gpus'][0]}")
        self.is_merged_layers_by_tasks = False

        if "is_merged_layers_by_tasks" in backbone_kwargs:
            self.is_merged_layers_by_tasks = backbone_kwargs["is_merged_layers_by_tasks"]
            d_out = backbone_kwargs["d_out"]

        # merged_layer
        self.use_skip_connection_via_shared_layers_type = None
        if "use_skip_connection_via_shared_layers_type" in backbone_kwargs:

            if not self.is_merged_layers_by_tasks:
                raise ValueError("There is no merge layer.")

            self.use_skip_connection_via_shared_layers_type = backbone_kwargs["use_skip_connection_via_shared_layers_type"]

        self.tasks_params = tasks_params
        self.n_cont_features = n_cont_features
        self.cls_embedding = _CLSEmbedding(d_block)

        # >>> Feature embeddings (Figure 2a in the paper).
        self.cont_embeddings = LinearEmbeddings(n_cont_features, d_block) if n_cont_features > 0 else None
        self.cat_embeddings = CategoricalEmbeddings(cat_cardinalities, d_block, True) if cat_cardinalities else None
        # <<<

        self.task_backbone_dict = nn.ModuleDict()
        n_block_via_main_task_dict = {}

        # for sub task
        for task_name, task_params in tasks_params.items():
            task_name = task_params.task_name
            n_block_via_main_task = task_params["n_block_via_main_task"]
            n_block_via_main_task_dict[n_block_via_main_task] = [n_block_via_main_task_dict]

            if self.is_merged_layers_by_tasks:
                if "d_out" in task_params:
                    task_params["d_out"] = None

            self.task_backbone_dict[task_name] = FTMultiTransformerBackbone(
                **task_params,
                is_sub_task=True,
                n_tokens=(None if task_params.get("linformer_kv_compression_ratio") is None else 1 + n_cont_features + len(cat_cardinalities)),
            )

        if self.is_merged_layers_by_tasks:
            if "d_out" in backbone_kwargs:
                backbone_kwargs["d_out"] = None

        # for main task
        self.backbone = FTMultiTransformerBackbone(
            **backbone_kwargs,
            n_tokens=(None if backbone_kwargs.get("linformer_kv_compression_ratio") is None else 1 + n_cont_features + len(cat_cardinalities)),
            n_block_via_main_task=n_block_via_main_task_dict,
        )

        self.reduction_layer = None

        # reduce layers
        if self.use_skip_connection_via_shared_layers_type in ["add_with_reduce_layer", "concat_with_reduce_layer"]:
            if self.use_skip_connection_via_shared_layers_type == "add_with_reduce_layer":
                branch_num = len(tasks_params) + 1

            else:
                # with shared layers
                branch_num = len(tasks_params) + 2

            self.reduction_layer = _named_sequential(
                ("normalization", nn.LayerNorm(d_block * branch_num)),
                ("activation", nn.ReLU()),
                ("linear", nn.Linear(d_block * branch_num, d_block)),
            )

        self.output = None

        if self.is_merged_layers_by_tasks:
            branch_num = len(tasks_params) + 1

            if self.use_skip_connection_via_shared_layers_type in ["add_with_reduce_layer", "concat_with_reduce_layer"]:
                branch_num = 1

            elif self.use_skip_connection_via_shared_layers_type in ["concat"]:
                branch_num = len(tasks_params) + 2

            self.output = _named_sequential(
                ("normalization", nn.LayerNorm(d_block * branch_num)),
                ("activation", nn.ReLU()),
                ("linear", nn.Linear(d_block * branch_num, d_out)),
            )

        self._is_default = _is_default

    @classmethod
    def get_default_kwargs(cls, n_blocks: int = 3) -> Dict[str, Any]:
        """Get the default hyperparameters.

        Args:
            n_blocks: the number of blocks. The supported values are: 1, 2, 3, 4, 5, 6.
        Returns:
            the default keyword arguments for the constructor.
        """
        if n_blocks < 0 or n_blocks > 6:
            raise ValueError(
                "Default configurations are available" " only for the following values of n_blocks: 1, 2, 3, 4, 5, 6." f" However, {n_blocks=}"
            )
        return {
            "n_blocks": n_blocks,
            "d_block": [96, 128, 192, 256, 320, 384][n_blocks - 1],
            "attention_n_heads": 8,
            "attention_dropout": [0.1, 0.15, 0.2, 0.25, 0.3, 0.35][n_blocks - 1],
            "ffn_d_hidden": None,
            # "4 / 3" for ReGLU leads to almost the same number of parameters
            # as "2.0" for ReLU.
            "ffn_d_hidden_multiplier": 4 / 3,
            "ffn_dropout": [0.0, 0.05, 0.1, 0.15, 0.2, 0.25][n_blocks - 1],
            "residual_dropout": 0.0,
            "_is_default": True,
        }

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
                    itertools.chain.from_iterable(
                        m.parameters() for block in self.backbone.blocks for name, m in block.named_children() if name.endswith("_normalization")
                    ),
                    itertools.chain.from_iterable(
                        m.parameters()
                        for _, blocks_task in self.task_backbone_dict.items()
                        for block in blocks_task.blocks
                        for name, m in block.named_children()
                        if name.endswith("_normalization")
                    ),
                    (p for name, p in self.named_parameters() if name.endswith(".bias")),
                )
            ),
            "weight_decay": 0.0,
        }
        main_group: Dict[str, Any] = {"params": [p for p in self.parameters() if p not in zero_wd_group["params"]]}
        zero_wd_group["params"] = list(zero_wd_group["params"])
        return [main_group, zero_wd_group]

    def make_default_optimizer(self) -> torch.optim.AdamW:
        """Create the "default" `torch.nn.AdamW` suitable for the *default* FT-Transformer.

        Returns:
            the optimizer.
        """  # noqa: E501
        if not self._is_default:
            warnings.warn("The default opimizer is supposed to be used in a combination" " with the default FT-Transformer.")

        return torch.optim.AdamW(self.make_parameter_groups(), lr=1e-4, weight_decay=1e-5)

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
        x, x_tasks_dict, attention_shared_dict = self.backbone(x)

        x_tasks = [x]

        output_attention_dict = {"shared": attention_shared_dict}

        for task_name, task_params in self.tasks_params.items():
            task_name = task_params.task_name
            x_common = x_tasks_dict[task_params["n_block_via_main_task"]]
            x_task, x_task_dict, attention_task_dict = self.task_backbone_dict[task_name](x_common)
            x_tasks.append(x_task)
            output_attention_dict[task_name] = attention_task_dict

        x = torch.cat(x_tasks, dim=1)

        return x, output_attention_dict

    def predict(self, x):
        x_cont, x_cat = x[:, 0 : self.n_cont_features], x[:, self.n_cont_features :]

        x_cont = torch.tensor(x_cont, dtype=torch.float32, device=self.device)
        x_cat = torch.tensor(x_cat, dtype=torch.long, device=self.device)

        y_pred, _ = self.forward(x_cont, x_cat)
        return y_pred.detach().to("cpu").numpy()
