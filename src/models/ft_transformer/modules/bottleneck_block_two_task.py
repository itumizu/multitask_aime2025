import torch.nn as nn
from torch import Tensor

from .multi_head_attention import MultiheadAttention
from .libs import _named_sequential

class BottleneckBlockMultiTask(nn.Module):
    def __init__(
        self,
        d_block,
        mask_attention_n_heads,
        n_tokens,
        ffn_d_hidden,
        ffn_use_reglu,
        ffn_dropout,
        mask_common_attention_n_heads=None,
        is_reversed=True,
        use_sparsemax=False,
        use_task_bottleneck_block=True,
        bottleneck_weights=[1.0, 1.0],
        detach_task: str = "",
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.is_reversed = is_reversed
        self.bottleneck_weights = bottleneck_weights
        self.detach_task = detach_task
        self.use_task_bottleneck_block = use_task_bottleneck_block

        if mask_common_attention_n_heads:
            self.mask_common_attention_n_heads = mask_common_attention_n_heads

        else:
            self.mask_common_attention_n_heads = None

        if self.use_task_bottleneck_block:
            self.attention_layer_a = MultiheadAttention(
                d_embedding=d_block, n_heads=mask_attention_n_heads, dropout=0.2, use_weights=True, n_tokens=n_tokens, use_sparsemax=use_sparsemax
            )

            self.linear_a = _named_sequential(
                (
                    "linear1",
                    nn.Linear(d_block, d_block),
                ),
                ("activation", nn.ReLU()),
            )

            self.attention_layer_reversed_a = MultiheadAttention(
                d_embedding=d_block, n_heads=mask_attention_n_heads, dropout=0.2, use_weights=True, n_tokens=n_tokens
            )

            self.linear_reversed_a = _named_sequential(
                (
                    "linear1",
                    nn.Linear(d_block, d_block),
                ),
                ("activation", nn.ReLU()),
            )

            self.attention_layer_b = MultiheadAttention(
                d_embedding=d_block, n_heads=mask_attention_n_heads, dropout=0.2, use_weights=True, n_tokens=n_tokens, use_sparsemax=use_sparsemax
            )

            self.linear_b = _named_sequential(
                (
                    "linear1",
                    nn.Linear(d_block, d_block),
                ),
                ("activation", nn.ReLU()),
            )

            self.attention_layer_reversed_b = MultiheadAttention(
                d_embedding=d_block, n_heads=mask_attention_n_heads, dropout=0.2, use_weights=True, n_tokens=n_tokens
            )

            self.linear_reversed_b = _named_sequential(
                (
                    "linear1",
                    nn.Linear(d_block, d_block),
                ),
                ("activation", nn.ReLU()),
            )

        if self.mask_common_attention_n_heads:
            self.attention_layer_c = MultiheadAttention(
                d_embedding=d_block, n_heads=self.mask_common_attention_n_heads, dropout=0.2, use_weights=True, n_tokens=n_tokens
            )

        # self.linear_c = _named_sequential(
        #     (
        #         "linear1",
        #         nn.Linear(d_block, d_block),
        #     ),
        #     ("activation", nn.ReLU()),
        # )

    def forward(self, x_task_a: Tensor, x_task_b: Tensor):
        if self.use_task_bottleneck_block:
            x_task_a, x_attention_probs_task_a, attention_weights_task_a_first = self.attention_layer_a(x_task_a, x_task_a)
            x_task_a = self.linear_a(x_task_a)

            x_task_b, x_attention_probs_task_b, attention_weights_task_b_first = self.attention_layer_b(x_task_b, x_task_b)
            x_task_b = self.linear_b(x_task_b)

            if self.detach_task == "a":
                x_attention_probs_task_a = x_attention_probs_task_a.detach()
                x_attention_probs_task_b = x_attention_probs_task_b.detach()

            # elif self.detach_task == "b":
            # x_attention_probs_task_a = x_attention_probs_task_a.detach()

            else:
                pass

            # if self.detach_task == "a":
            #     x_task_a, x_attention_probs_task_a, attention_weights_task_a_second = self.attention_layer_reversed_a(x_task_a, x_task_a)
            #     x_task_a = self.linear_reversed_a(x_task_a)

            #     x_task_b, x_attention_probs_task_b, attention_weights_task_b_second = self.attention_layer_reversed_b(x_task_b, x_task_b)
            #     x_task_b = self.linear_reversed_b(x_task_b)

        if self.mask_common_attention_n_heads:
            x_common, _, attention_weights_task_c = self.attention_layer_c(x_task_a, x_task_b)

            # x_common = self.linear_c(x_common)
            x_task_a = x_task_a + x_common
            x_task_b = x_task_b + x_common

        if self.use_task_bottleneck_block:
            # else:
            if self.is_reversed:
                x_task_a, x_attention_probs_task_a, attention_weights_task_a_second = self.attention_layer_reversed_a(
                    x_task_a, x_task_a, (1 - x_attention_probs_task_b) * self.bottleneck_weights[0]
                )

                # x_task_a, x_attention_probs_task_a, attention_weights_task_a_second = self.attention_layer_reversed_a(
                #     x_task_a, x_task_a, x_attention_probs_task_b * self.bottleneck_weights[0]
                # )

            else:
                x_task_a, x_attention_probs_task_a, attention_weights_task_a_second = self.attention_layer_reversed_a(x_task_a, x_task_a)

            x_task_a = self.linear_reversed_a(x_task_a)

            if self.is_reversed:
                x_task_b, x_attention_probs_task_b, attention_weights_task_b_second = self.attention_layer_reversed_b(
                    x_task_b, x_task_b, (1 - x_attention_probs_task_a) * self.bottleneck_weights[1]
                )

                # x_task_b, x_attention_probs_task_b, attention_weights_task_b_second = self.attention_layer_reversed_b(
                #     x_task_b, x_task_b, x_attention_probs_task_a * self.bottleneck_weights[1]
                # )
                # x_task_b, x_attention_probs_task_b, attention_weights_task_b_second = self.attention_layer_reversed_b(x_task_b, x_task_b)

            else:
                x_task_b, x_attention_probs_task_b, attention_weights_task_b_second = self.attention_layer_reversed_b(x_task_b, x_task_b)

            x_task_b = self.linear_reversed_b(x_task_b)

            attention_bottleneck_dict = {
                "task_a": {
                    0: attention_weights_task_a_first,
                    1: attention_weights_task_a_second,
                },
                "task_b": {
                    0: attention_weights_task_b_first,
                    1: attention_weights_task_b_second,
                },
            }

        else:
            attention_bottleneck_dict = {}

        if self.mask_common_attention_n_heads:
            attention_bottleneck_dict["task_c"] = {0: attention_weights_task_c}

        return x_task_a, x_task_b, attention_bottleneck_dict
