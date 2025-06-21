import itertools
import warnings
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim
from torch import Tensor
from torch.nn.parameter import Parameter

from .modules.embedding import _CLSEmbedding, LinearEmbeddings, CategoricalEmbeddings
from .modules.backbone import FTTransformerBackbone
from .modules.libs import _INTERNAL_ERROR


class FTTransformer(nn.Module):
    """The FT-Transformer model from Section 3.3 in the paper."""

    def __init__(
        self,
        *,
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

        self.n_cont_features = n_cont_features
        self.is_return_attention = True

        super().__init__()
        d_block: int = backbone_kwargs["d_block"]
        self.cls_embedding = _CLSEmbedding(d_block)
        self.device = torch.device(f"cuda:{backbone_kwargs['gpus'][0]}")

        # >>> Feature embeddings (Figure 2a in the paper).
        self.cont_embeddings = LinearEmbeddings(n_cont_features, d_block) if n_cont_features > 0 else None
        self.cat_embeddings = CategoricalEmbeddings(cat_cardinalities, d_block, True) if cat_cardinalities else None
        # <<<

        self.backbone = FTTransformerBackbone(
            **backbone_kwargs,
            n_tokens=(None if backbone_kwargs.get("linformer_kv_compression_ratio") is None else 1 + n_cont_features + len(cat_cardinalities)),
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

    def forward(self, x=None, x_cont: Optional[Tensor] = None, x_cat: Optional[Tensor] = None) -> Tuple[Tensor, dict]:
        """Do the forward pass."""
        x_any = x_cat if x_cont is None else x_cont

        if x_any is None:
            if x is None:
                raise ValueError("At least one of x_cont and x_cat must be provided.")

            else:
                if len(x.size()) == 1:
                    x = x.unsqueeze(0)

                x_cont = x[:, : self.n_cont_features]
                x_cat = x[:, self.n_cont_features :]

                # print(x_cont.device)
                # print(f"x_cat: ", x_cat.device)
                # print("x_cont: ", x_cont.shape)
                # print("x_cat: ", x_cat.shape)

                x_any = x_cat if x_cont is None else x_cont

        if x_cat.shape[1] == 0:
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
                    raise ValueError(FTTransformer._FORWARD_BAD_ARGS_MESSAGE.format(f"{argname} must be None"))

            else:
                if argvalue is None:
                    raise ValueError(FTTransformer._FORWARD_BAD_ARGS_MESSAGE.format(f"{argname} must not be None"))
                x_embeddings.append(module(argvalue))

        assert x_embeddings, _INTERNAL_ERROR

        x = torch.cat(x_embeddings, dim=1)
        x, attention_dict = self.backbone(x)
        output_attention_dict = {"shared": attention_dict, "tasks": {}}

        if self.is_return_attention:
            return x, output_attention_dict

        else:
            return x
