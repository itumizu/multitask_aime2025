from asyncio import tasks
import itertools
import math
import typing
import warnings
from collections import OrderedDict
from omegaconf import DictConfig
from prefect import task

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch import Tensor, Value, tensor
from torch.nn.parameter import Parameter

from .libs import _named_sequential


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
_LINFORMER_KV_COMPRESSION_SHARING = Literal["headwise", "key-value"]


class MultiheadAttentionBottleneck(nn.Module):
    """Multihead (Self-/Cross-)Attention with an optional linear attention from ["Linformer: Self-Attention with Linear Complexity"](https://arxiv.org/abs/2006.04768).

    **Examples**

    >>> batch_size, n_tokens, d_embedding = 2, 3, 16
    >>> n_heads = 8
    >>> a = torch.randn(batch_size, n_tokens, d_embedding)
    >>> b = torch.randn(batch_size, n_tokens * 2, d_embedding)
    >>> m = MultiheadAttention(
    ...     d_embedding=d_embedding, n_heads=n_heads, dropout=0.2
    >>> )
    >>>
    >>> # Self-attention.
    >>> assert m(a, a).shape == a.shape
    >>>
    >>> # Cross-attention.
    >>> assert m(a, b).shape == a.shape
    >>>
    >>> # Linformer attention.
    >>> m = MultiheadAttention(
    ...     d_embedding=d_embedding,
    ...     n_heads=n_heads,
    ...     dropout=0.2,
    ...     n_tokens=n_tokens,
    ...     linformer_kv_compression_ratio=0.5,
    ...     linformer_kv_compression_sharing='headwise',
    >>> )
    >>> assert m(a, a).shape == a.shape
    """  # noqa: E501

    def __init__(
        self,
        *,
        d_embedding: int,
        n_heads: int,
        dropout: float,
        # Linformer arguments.
        n_tokens: Optional[int] = None,
        linformer_kv_compression_ratio: Optional[float] = None,
        linformer_kv_compression_sharing: Optional[_LINFORMER_KV_COMPRESSION_SHARING] = None,
        use_weights: bool = False,
        use_sparemax: bool = False,
    ) -> None:
        """
        Args:
            d_embedding: the embedding size for one token.
                Must be a multiple of `n_heads`.
            n_heads: the number of heads. If greater than 1, then the module will have
                an additional output layer (the so called "mixing" layer).
            dropout: the dropout rate for the attention probability map.
            n_tokens: the number of tokens
                (must be provided if `linformer_kv_compression_ratio` is not None)
            linformer_kv_compression_ratio: Linformer-style compression rate.
                Must be within the interval `(0.0, 1.0)`.
            linformer_kv_compression_sharing: Linformer compression sharing policy.
                Must be provided if `linformer_kv_compression_ratio` is not None.
                (non-shared Linformer compression is not supported; the "layerwise"
                sharing policy is not supported).
        """
        if n_heads < 1:
            raise ValueError(f"n_heads must be positive, however: {n_heads=}")

        if d_embedding % n_heads:
            raise ValueError("d_embedding must be a multiple of n_heads," f" however: {d_embedding=}, {n_heads=}")

        super().__init__()
        self.W_q = nn.Linear(d_embedding, d_embedding)
        self.W_k = nn.Linear(d_embedding, d_embedding)
        self.W_v = nn.Linear(d_embedding, d_embedding)
        self.W_out = nn.Linear(d_embedding, d_embedding) if n_heads > 1 else None
        self.dropout = nn.Dropout(dropout) if dropout else None
        self._n_heads = n_heads

        self.use_weights = use_weights
        self.use_sparemax = use_sparemax

        if self.use_sparemax:
            self.sparsemax = Sparsemax(dim=-1)

        if linformer_kv_compression_ratio is not None:
            if n_tokens is None:
                raise ValueError("If linformer_kv_compression_ratio is not None," " then n_tokens also must not be None")
            if linformer_kv_compression_sharing not in typing.get_args(_LINFORMER_KV_COMPRESSION_SHARING):
                raise ValueError(
                    "Valid values of linformer_kv_compression_sharing include:"
                    f" {typing.get_args(_LINFORMER_KV_COMPRESSION_SHARING)},"
                    f" however: {linformer_kv_compression_sharing=}"
                )
            if linformer_kv_compression_ratio <= 0.0 or linformer_kv_compression_ratio >= 1.0:
                raise ValueError(
                    "linformer_kv_compression_ratio must be from the open interval" f" (0.0, 1.0), however: {linformer_kv_compression_ratio=}"
                )

            def make_linformer_kv_compression():
                return nn.Linear(
                    n_tokens,
                    max(int(n_tokens * linformer_kv_compression_ratio), 1),
                    bias=False,
                )

            self.key_compression = make_linformer_kv_compression()
            self.value_compression = make_linformer_kv_compression() if linformer_kv_compression_sharing == "headwise" else None

        else:
            if n_tokens is not None:
                raise ValueError("If linformer_kv_compression_ratio is None," " then n_tokens also must be None")
            if linformer_kv_compression_sharing is not None:
                raise ValueError("If linformer_kv_compression_ratio is None," " then linformer_kv_compression_sharing also must be None")
            self.key_compression = None
            self.value_compression = None

        for m in [self.W_q, self.W_k, self.W_v]:
            nn.init.zeros_(m.bias)

        if self.W_out is not None:
            nn.init.zeros_(self.W_out.bias)

    def _reshape(self, x: Tensor) -> Tensor:
        batch_size, n_tokens, d = x.shape
        d_head = d // self._n_heads
        return x.reshape(batch_size, n_tokens, self._n_heads, d_head).transpose(1, 2).reshape(batch_size * self._n_heads, n_tokens, d_head)

    # 元の形状になるように実装
    def _unreshape(self, x: Tensor) -> Tensor:
        batch_size_n_heads, n_tokens, d_head = x.shape
        batch_size = batch_size_n_heads // self._n_heads
        # d = d_head * self._n_heads
        return x.reshape(batch_size, self._n_heads, n_tokens, d_head).transpose(1, 2).reshape(batch_size, n_tokens, n_tokens)

    def forward(self, x_q: Tensor, x_kv: Tensor, attention_probs_reversed=None) -> Tuple[Tensor, Tensor] | Tuple[Tensor, Tensor, Tensor]:
        """Do the forward pass."""
        q, k, v = self.W_q(x_q), self.W_k(x_kv), self.W_v(x_kv)

        if self.key_compression is not None:
            k = self.key_compression(k.transpose(1, 2)).transpose(1, 2)
            v = (self.key_compression if self.value_compression is None else self.value_compression)(v.transpose(1, 2)).transpose(1, 2)

        batch_size = len(q)
        d_head_key = k.shape[-1] // self._n_heads
        d_head_value = v.shape[-1] // self._n_heads
        n_q_tokens = q.shape[1]

        q = self._reshape(q)
        k = self._reshape(k)

        # if attention_probs is None:
        attention_logits = q @ k.transpose(1, 2) / math.sqrt(d_head_key)

        # sparsemax を使ってみる
        if self.use_sparemax:
            attention_probs = self.sparsemax(attention_logits)

        else:
            attention_probs = F.softmax(attention_logits, dim=-1)

        if self.dropout is not None:
            attention_probs = self.dropout(attention_probs)

        x = attention_probs @ self._reshape(v)

        return x, attention_probs


class BottleneckBlockMultiTaskOneBlock(nn.Module):
    def __init__(
        self,
        d_block,
        mask_attention_n_heads,
        n_tokens,
        ffn_d_hidden,
        ffn_use_reglu,
        ffn_dropout,
        use_sparemax=False,
        is_reversed=False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.attention_layer_a = MultiheadAttentionBottleneck(
            d_embedding=d_block, n_heads=mask_attention_n_heads, dropout=0.2, use_weights=True, n_tokens=n_tokens, use_sparemax=use_sparemax
        )

        self.linear_a = _named_sequential(
            (
                "linear1",
                nn.Linear(d_block, d_block),
            ),
            ("activation", nn.ReLU()),
        )

        self.attention_layer_b = MultiheadAttentionBottleneck(
            d_embedding=d_block, n_heads=mask_attention_n_heads, dropout=0.2, use_weights=True, n_tokens=n_tokens, use_sparemax=use_sparemax
        )

        self.linear_b = _named_sequential(
            (
                "linear1",
                nn.Linear(d_block, d_block),
            ),
            ("activation", nn.ReLU()),
        )

        self.W_out_a = nn.Linear(d_block, d_block) if mask_attention_n_heads > 1 else None
        self.W_out_b = nn.Linear(d_block, d_block) if mask_attention_n_heads > 1 else None

        self._n_heads = mask_attention_n_heads

    def forward(self, x_task_a: Tensor, x_task_b: Tensor):
        batch_size = len(x_task_a)
        d_head_value = x_task_a.shape[-1] // self._n_heads
        n_q_tokens = x_task_a.shape[1]

        x_task_a, x_attention_probs_task_a = self.attention_layer_a(x_task_a, x_task_a)
        x_task_b, x_attention_probs_task_b = self.attention_layer_b(x_task_b, x_task_b)

        if not x_attention_probs_task_b is None:
            x_task_a = (1 - x_attention_probs_task_b) @ x_task_a

        if not x_attention_probs_task_a is None:
            x_task_b = (1 - x_attention_probs_task_a) @ x_task_b

        x_task_a = (
            x_task_a.reshape(batch_size, self._n_heads, n_q_tokens, d_head_value)
            .transpose(1, 2)
            .reshape(batch_size, n_q_tokens, self._n_heads * d_head_value)
        )

        x_task_b = (
            x_task_b.reshape(batch_size, self._n_heads, n_q_tokens, d_head_value)
            .transpose(1, 2)
            .reshape(batch_size, n_q_tokens, self._n_heads * d_head_value)
        )

        if self.W_out_a is not None:
            x = self.W_out_a(x_task_a)

        if self.W_out_b is not None:
            x = self.W_out_b(x_task_b)

        x_task_a = self.linear_a(x_task_a)
        x_task_b = self.linear_b(x_task_b)

        attention_bottleneck_dict = {
            "task_a": {
                0: x_attention_probs_task_a,
            },
            "task_b": {
                0: x_attention_probs_task_b,
            },
        }

        return x_task_a, x_task_b, attention_bottleneck_dict
