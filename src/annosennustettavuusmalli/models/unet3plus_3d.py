from collections import namedtuple
from typing import List

import torch
import torch.nn as nn

from annosennustettavuusmalli.models.basic_blocks import ConvBlock

""" 
Tekijä: Akseli Leino

This module creates a input for decoder node from other nodes using skip connections.
Module is meant to receive encoder and decoder nodes as on input.
There is no skip connection for depth = depth_max, as the final encoder is converted directly to decoder.
Module then pools or upsamples the nodes depending on the current depth.
These nodes are then put through convolution to create N*depth sized stack.
Output is a stack of feature maps, which are further convolved in decoder nodes.
"""


class SkipConnectionStack(nn.Module):
    def __init__(
        self,
        current_depth: int,
        total_depth: int,
        in_channels: List[int],
        pool_size: tuple,
        kernel_size: tuple,
        out_channels: int = 64,
        act_func: str = "relu",
    ) -> None:
        super(SkipConnectionStack, self).__init__()
        self.current_depth = current_depth
        self.total_depth = total_depth
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.filters = out_channels
        self.act_func = act_func

        # For each depth, n = total_depth sized SkipConnectionStack is created.
        self.convolution = nn.ModuleList(
            [
                ConvBlock(
                    in_channels=in_channels[i + 1]
                    if i <= self.current_depth or i == (self.total_depth - 1)
                    else self.filters * self.total_depth,
                    out_channels=self.filters,
                    conv_layers=1,
                    kernel_size=self.kernel_size,
                    padding="same",
                    act_func=self.act_func,
                    is_bn=True,
                )
                for i in range(self.total_depth)
            ]
        )

        # Adds Maxpool or Upsample based on the current_depth of the network. Kernel size is calculated based on the distance from pooling/upsampling module to current_depth. There's gonna be unused Upsample at i == current_depth
        # with upsampling scale_factor of 1.0, but that's intentional to preserve relationship between depth and ModuleList indices.

        self.poolings = nn.ModuleList(
            [
                nn.MaxPool3d(
                    kernel_size=(
                        self.pool_size[0] ** abs(self.current_depth - i),
                        self.pool_size[1] ** abs(self.current_depth - i),
                        self.pool_size[2] ** abs(self.current_depth - i),
                    )
                )
                if i < self.current_depth
                else nn.Upsample(
                    scale_factor=(
                        self.pool_size[0] ** abs(self.current_depth - i),
                        self.pool_size[1] ** abs(self.current_depth - i),
                        self.pool_size[2] ** abs(self.current_depth - i),
                    ),
                    mode="trilinear",
                )
                for i in range(self.total_depth)
            ]
        )

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        for depth_iter in range(self.total_depth):
            if depth_iter != self.current_depth:
                x[depth_iter] = self.poolings[depth_iter](x[depth_iter])
            x[depth_iter] = self.convolution[depth_iter](x[depth_iter])

        return torch.cat(x, dim=1)


""" This module creates an output for deep supervision from the decoder stacks.
"""


class DeepSupervisionOutput(nn.Module):
    def __init__(
        self,
        current_depth: int,
        total_depth: int,
        in_channels: int,
        pool_size: tuple,
        kernel_size: tuple,
        is_bn: bool = False,
        is_act: bool = True,
    ) -> None:
        super(DeepSupervisionOutput, self).__init__()
        self.current_depth = current_depth
        self.total_depth = total_depth
        self.in_channels = in_channels
        self.pool_size = pool_size
        self.is_bn = is_bn
        self.is_act = is_act
        self.kernel_size = kernel_size

        self.convolution = nn.Conv3d(
            in_channels=in_channels,
            out_channels=1,
            kernel_size=self.kernel_size,
            padding="same",
            bias=False if is_bn else True,
        )
        self.upsampling = nn.Upsample(
            scale_factor=(
                self.pool_size[0] ** self.current_depth,
                self.pool_size[1] ** self.current_depth,
                self.pool_size[2] ** self.current_depth,
            ),
            mode="trilinear",
        )

        if is_act:
            self.act = (
                nn.ReLU()
            )  # Not affected by act_func, as we don't want negative dose values

        if is_bn:
            self.bn = nn.BatchNorm3d(
                num_features=1
            )  # one because there's only one deep supervision channel per depth to output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convolution(x)
        x = self.upsampling(x)

        if self.is_bn:
            x = self.bn(x)
        if self.is_act:
            x = self.act(x)

        return x


""" This is the main Unet3+ model.

Architrecture without skip connections depth = 5

    Input
        |
        Encoder 0               Skip 0               Decoder 0 - Deepsup 0 (output - final prediction)
            |                                        |
            Encoder 1           Skip 1           Decoder 1 - Deepsup 1 (side output)
                |                                |   
                Encoder 2       Skip 2        Decoder 1 - Deepsup 2 (side output)
                     |                        |    
                    Encoder 3   Skip 3   Decoder 3 - Deepsup 3 (side output)
                        |                | 
                        Encoder(Decoder) 4 - Deepsup 4 (side output)

Skip connection example (Skip 1) with [64, 128, 256, 512, 1024] filters


    Encoder 0 -> Pooling (/4) -> Conv(in_channels = 64, out_channels = 64)   
    Encoder 1 -> Upsample(*1) -> Conv(in_channels = 128, out_channels = 64)  
    Decoder 2 -> Upsample(*2) -> Conv(in_channels = 320, out_channels = 64)  -----> Concatenate all to Depth*64
    Decoder 3 -> Upsample(*4) -> Conv(in_channels = 320, out_channels = 64)
    Decoder 4 -> Upsample(*8) -> Conv(in_channels = 1024, out_channels = 64)

"""


class UNet3plus_3d(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        filters: List[int] = [16, 32, 64, 128, 256],
        kernel_size: int = 3,
        skip_filters: int = 64,
        pool_size: int = 2,
        conv_layers: int = 2,
        act_func: str = "ReLU",
        patch_size: tuple = (256, 256, 16),
    ) -> None:

        super(UNet3plus_3d, self).__init__()
        self.depth = len(filters)

        # If input is 2D slices, 3D kernels are reduced to have redundant third dimension
        if patch_size[2] == 1:
            self.pool_size = (pool_size, pool_size, 1)
            self.kernel_size = (kernel_size, kernel_size, 1)
        else:
            self.pool_size = (pool_size, pool_size, pool_size)
            self.kernel_size = (kernel_size, kernel_size, kernel_size)

        pooling_reduction = pool_size ** len(filters)

        if patch_size[2] > 1:
            if patch_size[2] % pooling_reduction != 0:
                raise ValueError(
                    "z-dimension of patch_size violates with selected pooling size and UNet depth"
                )

        self.conv_layers = conv_layers
        self.filters = (
            filters.copy()
        )  # Otherwise insertion will change the original list!
        self.skip_filters = skip_filters
        self.decoder_filters = self.filters.copy()
        self.filters.insert(0, in_channels)
        self.decoder_filters.reverse()
        self.act_func = act_func

        # This is a list of encoder convolution stacks
        self.conv_encoder = nn.ModuleList(
            [
                ConvBlock(
                    in_channels=self.filters[i],
                    out_channels=self.filters[i + 1],
                    conv_layers=self.conv_layers,
                    kernel_size=self.kernel_size,
                    padding="same",
                    act_func=self.act_func,
                    is_bn=True,
                )
                for i in range(self.depth)
            ]
        )

        # This is a list of decoder convolution stacks
        self.conv_decoder = nn.ModuleList(
            [
                ConvBlock(
                    in_channels=self.skip_filters * self.depth,
                    out_channels=self.skip_filters * self.depth,
                    conv_layers=1,
                    kernel_size=self.kernel_size,
                    padding="same",
                    act_func=self.act_func,
                    is_bn=True,
                )
                for i in range(self.depth - 1)
            ]
        )

        # This is list of pooling modules used between encoder convolutions
        self.pooling_encoder = nn.ModuleList(
            [nn.MaxPool3d(self.pool_size) for i in range(self.depth - 1)]
        )

        # This is a list of skip connections. Inputs for these are encoder stacks, and outputs are decoder stacks.
        self.skip_conv_stacks = nn.ModuleList(
            [
                SkipConnectionStack(
                    current_depth=i,
                    total_depth=self.depth,
                    pool_size=self.pool_size,
                    kernel_size=self.kernel_size,
                    in_channels=self.filters,
                    out_channels=self.skip_filters,
                    act_func=self.act_func,
                )
                for i in range(self.depth - 1)
            ]
        )

        # This is a list of deep supervision modules that take decoder convolution stacks into convolution module, resulting in output layers
        self.deep_supervision_outputs = nn.ModuleList(
            [
                DeepSupervisionOutput(
                    in_channels=self.skip_filters * self.depth
                    if i < (self.depth - 1)
                    else self.filters[-1],
                    current_depth=i,
                    total_depth=self.depth,
                    pool_size=self.pool_size,
                    kernel_size=self.kernel_size,
                    is_bn=False,
                    is_act=True,
                )
                for i in range(self.depth)
            ]
        )

        # Create output type once to avoid redefining a dynamic namedtuple in each forward call.
        self.Output = namedtuple(
            "Output",
            ["main_output"]
            + ["deep_supervision_{}".format(i + 1) for i in range(self.depth - 1)],
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:

        encoder_nodes = []

        # We'll append first encoder to list
        encoder_nodes.append(self.conv_encoder[0](x))

        # Then we connect pooling layer to last value of encoder list, and append a new encoder node.
        # The new node is the last one in the next iteration.
        for i in range(self.depth - 1):
            temp = self.pooling_encoder[i](encoder_nodes[-1])
            encoder_nodes.append(self.conv_encoder[i + 1](temp))

        skip_nodes = []
        decoder_nodes = []

        # After encoder nodes, we will connect the skip nodes and decoder nodes in reverse order, from UNet bottom to surface.
        # However, connections are always inserted to index = 0 to "reverse the reverse" and maintain equality between list indices and UNet depth.
        # We start from the bottom, because all skip nodes need the nodes below them.
        # Each decoder node takes skip node as input. Each skip node takes a list of encoder nodes above and same level, and decoder nodes below as input.
        # First (deepest) decoder is same as last encoder node.
        # In the first iteration only encoder nodes will be used.
        # In the next round, decoder node has been appended with the output of the skip connection. Next skip connection will take one less encoder node, and the newly created decoder node as input.

        decoder_nodes.insert(0, encoder_nodes[-1])
        for depth_iter in reversed(range(self.depth - 1)):
            skip_nodes.insert(
                0,
                self.skip_conv_stacks[depth_iter](
                    encoder_nodes[: depth_iter + 1] + decoder_nodes[:]
                ),
            )  # List is mutable, so [:] prevents passing parameter by reference
            decoder_nodes.insert(0, self.conv_decoder[depth_iter](skip_nodes[0]))

        deep_supervision_nodes = []

        for i in range(self.depth):
            deep_supervision_nodes.append(
                self.deep_supervision_outputs[i](decoder_nodes[i])
            )

        return self.Output(*deep_supervision_nodes)
