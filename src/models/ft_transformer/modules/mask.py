import numpy as np

import torch
import torch.nn as nn
from torch.nn import BatchNorm1d, Linear
from .sparsemax import Sparsemax


def initialize_non_glu(module, input_dim, output_dim):
    gain_value = np.sqrt((input_dim + output_dim) / np.sqrt(4 * input_dim))
    torch.nn.init.xavier_normal_(module.weight, gain=gain_value)
    # torch.nn.init.zeros_(module.bias)
    return


class Mask(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        group_dim,
        device: torch.device,
        virtual_batch_size=128,
        gamma=1.3,
        momentum=0.02,
        mask_type="softmax",
    ):
        """
        Initialize an attention transformer.

        Parameters
        ----------
        input_dim : int
            Input size
        group_dim : int
            Number of groups for features
        virtual_batch_size : int
            Batch size for Ghost Batch Normalization
        momentum : float
            Float value between 0 and 1 which will be used for momentum in batch norm
        mask_type : str
            Either "sparsemax" or "entmax" : this is the masking function to use
        """
        super(Mask, self).__init__()

        self.mask_type = mask_type
        # self.fc = Linear(input_dim, group_dim, bias=False)
        # initialize_non_glu(self.fc, input_dim, group_dim)

        # self.bn = GBN(group_dim, virtual_batch_size=virtual_batch_size, momentum=momentum)

        if mask_type == "sparsemax":
            # Sparsemax
            self.selector = Sparsemax(dim=-1)

        else:
            self.selector = torch.nn.Softmax(dim=-1)

        # self.bn = nn.BatchNorm1d(59)

    def forward(self, processed_feat, prior):
        # processed_feat = self.fc(processed_feat)
        # processed_feat = self.bn(processed_feat)
        x = torch.mul(processed_feat, prior)
        x = self.selector(x)

        return x
