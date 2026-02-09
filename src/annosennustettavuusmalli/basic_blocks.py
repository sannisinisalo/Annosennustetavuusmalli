import torch
import torch.nn as nn
from typing import List

""" 
Tekijä: Akseli Leino

This is a very standard convolution block module that creates a n-sized stack of CNN-BN-ReLU layers
These are used at encoder, decoder, and output nodes, and skip connections.
"""
class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, conv_layers: List[int], kernel_size: tuple, padding: int, act_func: str, is_bn: bool = True) -> None:
        super(ConvBlock, self).__init__()
        self.conv_layers = conv_layers
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.act_func = act_func
        self.is_bn = is_bn

        self.convolution = nn.ModuleList([nn.Conv3d(in_channels = self.in_channels if i == 0 else self.out_channels, 
                                                    out_channels = self.out_channels, 
                                                    kernel_size = self.kernel_size,
                                                    padding = self.padding,
                                                    bias = False if self.is_bn else True) for i in range(conv_layers)])
        if is_bn:
            self.batch_norm = nn.ModuleList([nn.BatchNorm3d(num_features = out_channels) for i in range(conv_layers)])
        
        self.act = nn.ModuleList([getattr(nn, act_func)() for i in range(conv_layers)])


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(self.conv_layers):
            x = self.convolution[i](x)
            if self.is_bn:
                x = self.batch_norm[i](x)
            x = self.act[i](x)
        return x

"""
WIP - Don't use yet.
This is a residual block introduced in original ResNet paper that can be used in place of ConvBlock. 
"""
class ResNetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int=3, num_blocks: int=1, act_func='ReLU', padding='same') -> None:
        super(ResNetBlock, self).__init__()
        self.blocks = nn.Sequential()

        for block_idx in range(num_blocks):
            block = nn.Sequential(nn.Conv3d(in_channels if block_idx == 0 else out_channels, out_channels//4, kernel_size=1, padding=padding, bias=False),
                                          nn.BatchNorm3d(num_features = out_channels//4),
                                          getattr(nn, act_func)(inplace=True),
                                          nn.Conv3d(out_channels//4, out_channels//4, kernel_size, padding=padding, bias=False),
                                          nn.BatchNorm3d(num_features = out_channels//4),
                                          getattr(nn, act_func)(inplace=True),
                                          nn.Conv3d(out_channels//4, out_channels, kernel_size=1, padding=padding, bias=False),
                                          nn.BatchNorm3d(num_features = out_channels))
            self.blocks.add_module(f'ResNetBlock_{block_idx}', block)
        
        self.out_act = getattr(nn, act_func)(inplace=True)

        self.expand_identity_filters = None
        if in_channels != out_channels:
            self.expand_identity_filters = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, padding=padding, bias=False),
                nn.BatchNorm3d(num_features = out_channels)
            )

        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.expand_identity_filters is not None:
            identity = self.expand_identity_filters(x)
        else:
            identity = x.clone()

        out = self.blocks(x)

        out += identity
        out = self.out_act(out)
        
        return out


class DepthwiseSeparableConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding='same', is_bn=True, bias=True, conv_layers: int=2, act_func='ReLU'):
        super(DepthwiseSeparableConv3d, self).__init__()
        self.kernel_size = kernel_size
        self.blocks = nn.Sequential()
        self.padding = padding
        self.num_blocks = conv_layers
        self.is_bn = is_bn
        
        for block_idx in range(self.num_blocks):
            for i in range(1):
                intermediate_channels = in_channels if block_idx == 0 and i == 0 else out_channels
                block = nn.Sequential(nn.Conv3d(intermediate_channels, 
                                                intermediate_channels, 
                                                kernel_size=self.kernel_size, 
                                                padding=self.padding,
                                                groups=intermediate_channels, 
                                                bias=bias),
                                      nn.Conv3d(intermediate_channels, 
                                                out_channels, 
                                                kernel_size=1, 
                                                stride=1, 
                                                padding='same', 
                                                bias=False))
                if self.is_bn:
                    block.add_module(f'BN_{block_idx}_{i}',  nn.BatchNorm3d(num_features = out_channels))
                block.add_module(f'Act_{block_idx}_{i}', getattr(nn, act_func)(inplace=True))                    
                self.blocks.add_module(f'DepthwiseSeparableConvBlock_{block_idx}_{i}', block)
                
    def forward(self, x):
        x = self.blocks(x)
        return x



class SeparableConvolution3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding='same', stride=1, dilation=2, bias=True, is_relu=True, is_bn=True, act_func='ReLU'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.bias = bias
        self.is_relu = is_relu
        self.is_bn = is_bn

        self.act = getattr(nn, act_func)() if self.is_relu else nn.Identity()
        self.depthwise = nn.Conv3d(self.in_channels, self.in_channels, self.kernel_size, self.stride, self.padding, self.dilation, groups=self.in_channels, bias=self.bias)
        self.pointwise = nn.Conv3d(self.in_channels, self.out_channels, kernel_size=1, stride=1, dilation=1, bias=self.bias)
        self.bn = nn.BatchNorm3d(self.out_channels) if self.is_bn else nn.Identity()

    def forward(self, x):
        x = self.act(x)
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return x

class SeparableBlock3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, conv_layers, padding='same', stride=1, dilation=1, bias=True, is_relu=True, is_bn=True, act_func='ReLU'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.conv_layers = conv_layers
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.bias = bias
        self.is_relu = is_relu
        self.is_bn = is_bn

        self.layers = []
        for i in range(self.conv_layers):
            in_channels = self.in_channels if i == 0 else self.out_channels
            self.layers.append(SeparableConvolution3d(in_channels, self.out_channels, self.kernel_size))

        self.sequence = nn.Sequential(*self.layers)
        self.resconv = nn.Conv3d(self.in_channels, self.out_channels, kernel_size=1)
        self.res_bn = nn.BatchNorm3d(self.out_channels) if self.is_bn else nn.Identity()
        self.res_act = getattr(nn, act_func)() if is_relu else nn.Identity()

    def forward(self, x):
        residual = self.res_act(x)
        residual = self.resconv(x)
        residual = self.res_bn(residual)
        x = self.sequence(x)
        x = x+residual

        return x
        
        

    