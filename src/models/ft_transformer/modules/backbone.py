import typing
from typing import Optional, Tuple, cast
from prefect import task

import torch.nn as nn
from torch import Tensor

from .libs import _TransformerFFNActivation, _LINFORMER_KV_COMPRESSION_SHARING, _named_sequential
from .multi_head_attention import MultiheadAttention
from .reglu import _ReGLU


class FTTransformerBackbone(nn.Module):
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
        # NOTE[DIFF]
        # In the paper, FT-Transformer uses the ReGLU activation.
        # Here, to illustrate the difference, ReLU activation is also supported
        # (in particular, see the docstring).
        ffn_activation: _TransformerFFNActivation = "ReGLU",
        residual_dropout: float,
        n_tokens: Optional[int] = None,
        linformer_kv_compression_ratio: Optional[float] = None,
        linformer_kv_compression_sharing: Optional[_LINFORMER_KV_COMPRESSION_SHARING] = None,
        **kwargs,
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

        super().__init__()
        ffn_use_reglu = ffn_activation == "ReGLU"
        self.blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
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
                        **({} if layer_idx == 0 else {"attention_normalization": nn.LayerNorm(d_block)}),
                    }
                )
                for layer_idx in range(n_blocks)
            ]
        )

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
        """Do the forward pass."""
        if x.ndim != 3:
            raise ValueError(f"The input must have exactly three dimension, however: {x.ndim=}")

        n_blocks = len(self.blocks)
        attention_dict = {}

        for i_block, block in enumerate(self.blocks):
            block = cast(nn.ModuleDict, block)

            x_identity = x

            if "attention_normalization" in block:
                x = block["attention_normalization"](x)

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

        x = x[:, 0]  # The representation of [CLS]-token.

        if self.output is not None:
            x = self.output(x)

        return x, attention_dict
