import torch

"""
Tekijä: Akseli Leino
"""


def init_weights_kaiming(m):
    """Initializes convolutional kernels with kaiming initialization as suggested by Kaiming He et al."""
    if isinstance(m, torch.nn.Conv3d):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
