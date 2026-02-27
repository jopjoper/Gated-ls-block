# ==============================================================================
# Copyright (c) 2024 - Present. Your Name or Organization.
# All rights reserved.
#
# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/licenses/MIT
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
Gated LSConv Block for HRNet Integration
==========================================

This module provides a specialized bottleneck block designed to integrate Large-Separable
Convolutions (LSConv) into the HRNet (High-Resolution Network) architecture.

Key Components:
    - LayerNormGeneral: A custom LayerNorm implementation supporting the (B, C, H, W) format.
    - GatedLSBlock_BCHW: A drop-in replacement for HRNet's BasicBlock, utilizing a Gated Linear Unit (GLU)
      mechanism combined with LSConv for efficient spatial feature mixing.

Reference:
    - This design is inspired by ConvNeXt (https://arxiv.org/abs/2201.03545) and modern MLP architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from mmengine.model import BaseModule

# Attempt to import the custom LSConv module.
# Ensure `lsnet.py` exists in the same directory or in the Python path.
try:
    from .lsnet import LSConv
except ImportError:
    # Fallback import for direct execution or different directory structures
    try:
        from mmpose.models.backbones.lsnet import LSConv
    except ImportError:
        raise ImportError(
            "Could not import 'LSConv'. Please ensure 'lsnet.py' is located in the "
            "current directory or under 'mmpose/models/backbones/'."
        )


class LayerNormGeneral(nn.Module):
    r"""
    A generic Layer Normalization layer that operates on the channel dimension (dim=1)
    of a 4D tensor with shape (Batch, Channels, Height, Width).

    This implementation is specifically designed for spatial attention blocks and
    ConvNeXt-like designs where normalization is applied before the depth-wise convolution.

    Args:
        normalized_shape (int or list): The shape of the input tensor that is normalized over.
                                         Since this operates on dim=1, only the channel dimension
                                         (int) is required.
        eps (float): A value added to the denominator for numerical stability. Defaults to 1e-6.
    """

    def __init__(self, normalized_shape: Union[int, list, tuple], eps: float = 1e-6, **kwargs):
        super().__init__()
        if isinstance(normalized_shape, (list, tuple)):
            dim = normalized_shape[0]
        else:
            dim = normalized_shape

        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Normalized tensor of shape (B, C, H, W).
        """
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class GatedLSBlock(BaseModule):
    r"""
    A Gated Linear Unit (GLU) block integrated with LSConv, designed as a drop-in replacement
    for standard residual blocks in HRNet.

    This block follows the modern design paradigm:
    `Norm -> Linear (Expand) -> Gating -> Depth-wise Conv (LSConv) -> Linear (Project) -> Residual`

    It specifically adapts to the HRNet interface by accepting unused parameters like `norm_cfg`
    and `conv_cfg` for API compatibility, while using its own internal normalization and convolution.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int): Stride of the block. If > 1, downsampling is applied via the shortcut.
        downsample (nn.Module, optional): External downsampling module. If None, it is auto-constructed
                                          if stride != 1 or in_channels != out_channels.
        expansion_ratio (float): Ratio to expand the channels in the hidden layer. Defaults to 8/3 (ConvNeXt ratio).
        conv_ratio (float): Ratio of channels that will be passed through the LSConv (the rest are identity). Defaults to 1.0.
        drop_path (float): Stochastic depth rate. Defaults to 0.0.
        norm_cfg (dict, optional): Ignored. Kept for HRNet API compatibility.
        conv_cfg (dict, optional): Ignored. Kept for HRNet API compatibility.
        with_cp (bool, optional): Ignored. Kept for HRNet API compatibility.
        init_cfg (dict, optional): Initialization config for mmengine.
    """

    expansion: int = 1

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            expansion_ratio: float = 8 / 3,
            conv_ratio: float = 1.0,
            drop_path: float = 0.0,
            norm_cfg: Optional[dict] = None,  # Compatibility placeholder
            conv_cfg: Optional[dict] = None,  # Compatibility placeholder
            with_cp: bool = False,  # Compatibility placeholder
            init_cfg: Optional[dict] = None,
            **kwargs
    ):
        super().__init__(init_cfg)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.drop_prob = drop_path

        # --- Handle Downsampling for HRNet Compatibility ---
        self.downsample = downsample
        if self.downsample is None and (stride != 1 or in_channels != out_channels):
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        # --- Core Block Logic ---
        self.norm = LayerNormGeneral(in_channels, eps=1e-6)

        # Channel expansion (using 1x1 Conv)
        hidden_dim = int(expansion_ratio * in_channels)
        self.fc1 = nn.Conv2d(in_channels, hidden_dim * 2, kernel_size=1)
        self.act = nn.GELU()

        # Channel splitting for Gating and Convolution
        conv_channels = int(conv_ratio * in_channels)
        # Split logic: [Gating, Identity, LSConv]
        self.split_indices = (hidden_dim, hidden_dim - conv_channels, conv_channels)

        # Large Kernel Convolution (LSConv)
        # Note: LSConv is assumed to have a fixed kernel size (e.g., 7x7) internally
        self.conv = LSConv(conv_channels)

        # Channel projection back to output
        self.fc2 = nn.Conv2d(hidden_dim, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input feature map of shape (B, C_in, H, W).

        Returns:
            torch.Tensor: Output feature map of shape (B, C_out, H', W').
        """
        shortcut = x
        if self.downsample is not None:
            shortcut = self.downsample(x)

        # --- Forward Pass ---
        x = self.norm(x)

        # Expand and split
        x = self.fc1(x)
        g, i, c = torch.split(x, self.split_indices, dim=1)

        # Apply LSConv to the designated portion
        c = self.conv(c)

        # Gating mechanism: Gate * (Identity + Conv)
        x = self.act(g) * torch.cat((i, c), dim=1)

        # Project back
        x = self.fc2(x)

        # --- Stochastic Depth (Drop Path) ---
        if self.training and self.drop_prob > 0:
            keep_prob = 1 - self.drop_prob
            # Create binary mask
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
            random_tensor.floor_()  # Binarize
            x = x.div(keep_prob) * random_tensor

        return x + shortcut