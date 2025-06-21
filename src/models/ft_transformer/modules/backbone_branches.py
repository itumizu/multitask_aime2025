import typing
from typing import Optional, Tuple, cast

import torch.nn as nn
import torch.optim
from torch import Tensor

from .bottleneck_block import BottleneckBlock
from .multi_head_attention import MultiheadAttention
from .libs import _TransformerFFNActivation, _LINFORMER_KV_COMPRESSION_SHARING, _named_sequential
from .reglu import _ReGLU


class FTTransformerMultiBranchesBackbone(nn.Module):
    """The backbone of FT-Transformer.

    The differences with Transformer from the paper
    ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) are as follows:

    - the so called "PreNorm" variation is used
        (`norm_first=True` in terms of `torch.nn.TransformerEncoderLayer`)
    - the very first normalization is skipped. This is **CRUCIAL** for FT-Transformer
        in the PreNorm configuration.

    **Examples**

    >>> batch_size = 2
    >>> n_tokens = 3
    >>> d_block = 16
    >>> x = torch.randn(batch_size, n_tokens, d_block)
    >>> d_out = 1
    >>> m = FTTransformerBackbone(
    ...     d_out=d_out,
    ...     n_blocks=2,
    ...     d_block=d_block,
    ...     attention_n_heads=8,
    ...     attention_dropout=0.2,
    ...     ffn_d_hidden=None,
    ...     ffn_d_hidden_multiplier=2.0,
    ...     ffn_dropout=0.1,
    ...     residual_dropout=0.0,
    ... )
    >>> m(x).shape
    torch.Size([2, 1])
    """

    def __init__(
        self,
        *,
        d_out: Optional[int],
        n_blocks: int,
        d_block: int,
        attention_n_heads: int,
        attention_dropout: float,
        ffn_d_hidden: Optional[int] = None,
        ffn_d_hidden_multiplier: Optional[float],
        ffn_dropout: float,
        ffn_activation: _TransformerFFNActivation = "ReGLU",
        residual_dropout: float,
        n_tokens: Optional[int] = None,
        linformer_kv_compression_ratio: Optional[float] = None,
        linformer_kv_compression_sharing: Optional[_LINFORMER_KV_COMPRESSION_SHARING] = None,
        n_block_via_main_task: dict = {},
        is_sub_task: bool = False,
        is_hidden_module: bool = False,
        **kwargs,
        # NOTE[DIFF]
        # In the paper, FT-Transformer uses the ReGLU activation.
        # Here, to illustrate the difference, ReLU activation is also supported
        # (in particular, see the docstring).
    ):
        """
        Args:
            d_out: the output size.
            n_blocks: the number of blocks.
            d_block: the block width
                (or, equivalently, the embedding size of each feature).
                Must be a multiple of `attention_n_heads`.
            attention_n_heads: the number of attention heads in `MultiheadAttention`.
            attention_dropout: the dropout rate in `MultiheadAttention`. Usually,
                positive values work better, even if the number of features is low.
            ffn_d_hidden: the hidden representation size after the activation in the
                feed-forward blocks (or, equivalently, the *input* size of the *second*
                linear layer in the feed-forward blocks). If ``ffn_use_reglu``
                is `True`, then the *output* size of the *first* linear layer
                will be set to ``2 * ffn_d_hidden``.
            ffn_d_hidden_multiplier: the alternative way to set `ffn_d_hidden` as
                `int(d_block * ffn_d_hidden_multiplier)`.
            ffn_dropout: the dropout rate for the hidden representation
                in the feed-forward blocks.
            ffn_activation: the activation used in the FFN blocks. To maintain (almost)
                the same number of parameters between different activations:
                <ffn_d_hidden_multiplier for ReGLU> = <2 / 3 * ffn_d_hidden_multiplier for ReLU>
                or
                <ffn_d_hidden_multiplier for ReLU> = <3 / 2 * ffn_d_hidden_multiplier for ReGLU>
            residual_dropout: the dropout rate for all residual branches.
            n_tokens: the argument for `MultiheadAttention`.
            linformer_kv_compression_ratio: the argument for `MultiheadAttention`.
            linformer_kv_compression_sharing: the argument for `MultiheadAttention`.
        """  # noqa: E501
        if ffn_activation not in typing.get_args(_TransformerFFNActivation):
            raise ValueError("ffn_activation must be one of" f" {typing.get_args(_TransformerFFNActivation)}." f" However: {ffn_activation=}")

        if ffn_d_hidden is None:
            if ffn_d_hidden_multiplier is None:
                raise ValueError("If ffn_d_hidden is None," " then ffn_d_hidden_multiplier must not be None")
            ffn_d_hidden = int(d_block * cast(float, ffn_d_hidden_multiplier))

        else:
            if ffn_d_hidden_multiplier is not None:
                raise ValueError("If ffn_d_hidden is not None," " then ffn_d_hidden_multiplier must be None")

        ffn_use_reglu = ffn_activation == "ReGLU"
        self.n_block_via_main_task_dict = n_block_via_main_task
        self.is_sub_task = is_sub_task
        self.is_hidden_module = is_hidden_module

        self.add_bottleneck = False
        self.use_attention_normalization = True

        if "add_bottleneck" in kwargs:
            self.add_bottleneck = kwargs["add_bottleneck"]

        if "use_attention_normalization" in kwargs:
            self.use_attention_normalization = kwargs["use_attention_normalization"]

        # temp
        self.gamma = 1.2

        super().__init__()
        self.blocks = nn.ModuleList()
        for layer_idx in range(n_blocks):
            # self.blocks.append(
            #     Block(
            #         d_block=d_block,
            #         attention_n_heads=attention_n_heads,
            #         attention_dropout=attention_dropout,
            #         residual_dropout=residual_dropout,
            #         ffn_d_hidden=ffn_d_hidden,
            #         ffn_dropout=ffn_dropout,
            #         ffn_use_reglu=ffn_use_reglu,
            #         i_block=layer_idx,
            #         n_blocks=n_blocks,
            #         n_tokens=n_tokens,
            #         linformer_kv_compression_ratio=linformer_kv_compression_ratio,
            #         linformer_kv_compression_sharing=linformer_kv_compression_sharing,
            #     )
            # )

            block_dict = {
                # >>> attention
                "attention": MultiheadAttention(
                    d_embedding=d_block,
                    n_heads=attention_n_heads,
                    dropout=attention_dropout,
                    n_tokens=n_tokens,
                    linformer_kv_compression_ratio=linformer_kv_compression_ratio,
                    linformer_kv_compression_sharing=linformer_kv_compression_sharing,
                ),
                "attention_residual_dropout": nn.Dropout(residual_dropout),
                # >>> feed-forward
                "ffn_normalization": nn.LayerNorm(d_block),
                "ffn": _named_sequential(
                    (
                        "linear1",
                        # ReGLU divides dimension by 2,
                        # so multiplying by 2 to compensate for this.
                        nn.Linear(d_block, ffn_d_hidden * (2 if ffn_use_reglu else 1)),
                    ),
                    ("activation", _ReGLU() if ffn_use_reglu else nn.ReLU()),
                    ("dropout", nn.Dropout(ffn_dropout)),
                    ("linear2", nn.Linear(ffn_d_hidden, d_block)),
                ),
                "ffn_residual_dropout": nn.Dropout(residual_dropout),
                # >>> output (for hook-based introspection)
                "output": nn.Identity(),
                # >>> the very first normalization
            }

            # if is_sub_task:
            #     if layer_idx == 0:

            # else:
            # if not layer_idx == 0:
            #     block_dict["attention_normalization"] = nn.LayerNorm(d_block)

            if is_sub_task and self.use_attention_normalization:
                block_dict["attention_normalization"] = nn.LayerNorm(d_block)

            else:
                if not layer_idx == 0:
                    block_dict["attention_normalization"] = nn.LayerNorm(d_block)

            if self.add_bottleneck and not layer_idx == 0:
                block_dict["bottleneck_block"] = BottleneckBlock(
                    d_block=d_block,
                    mask_attention_n_heads=kwargs["n_heads_in_bottleneck"],
                    ffn_d_hidden=1,
                    ffn_use_reglu=True,
                    ffn_dropout=0.0,
                    n_tokens=n_tokens,
                )

            self.blocks.append(nn.ModuleDict(block_dict))

        self.output = (
            None
            if d_out is None
            else _named_sequential(
                ("normalization", nn.LayerNorm(d_block)),
                ("activation", nn.ReLU()),
                ("linear", nn.Linear(d_block, d_out)),
            )
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, dict]:
        n_blocks = len(self.blocks)
        attention_dict = {}

        for i_block, block in enumerate(self.blocks):
            block = cast(nn.ModuleDict, block)
            x_identity = x

            # add bottleneck block
            if "bottleneck_block" in block:
                x, _, _ = block["bottleneck_block"](x, x)

            if "attention_normalization" in block:
                x = block["attention_normalization"](x)

            if self.is_hidden_module:
                x, attention = block["attention"](x, x)

            else:
                x, attention = block["attention"](x[:, :1] if i_block + 1 == n_blocks else x, x)

            x = block["attention_residual_dropout"](x)
            x = x_identity + x

            x_identity = x
            x = block["ffn_normalization"](x)
            x = block["ffn"](x)
            x = block["ffn_residual_dropout"](x)
            x = x_identity + x

            x = block["output"](x)
            attention_dict[i_block] = attention

        if not self.is_hidden_module:
            x = x[:, 0]  # The representation of [CLS]-token.

            if self.output is not None:
                x = self.output(x)

        return x, attention_dict
