import numpy as np
import torch
import copy

"""
Tekijä: Akseli Leino
"""

def integer_mask_to_binary(arr, num_bits):
    """Input mask is coded as binary turned to 10-base to preserve overlaps. This function turns the 10-base information back to binary.

    Args: - arr (torch.Tensor, np.ndarray): Array with any shape, can be tensor or numpy.
          - num_bits (int): Value that defines how many bits are used (also output dimension)
    
    Returns: - bin_nums (torch.Tensor, np.ndarray): Input array represented as binary. Additional dimension is last. (i.e., input shaped (x, y, z) leads to output (x, y, z, num_bits)
    
    """

    if isinstance(arr, np.ndarray):
        arr = arr.copy() 
        arr[arr < 1] = 0
        arr = arr.astype(np.int32)
        bin_nums = ((arr.reshape(arr.shape + (1,)) & (2**np.arange(num_bits))) != 0).astype(int)
    elif isinstance(arr, torch.Tensor):
        arr = arr.clone()
        arr[arr < 1] = 0
        arr = arr.to(torch.int32)
        bin_nums = ((arr.reshape(arr.shape + (1,)) & (2**torch.arange(num_bits))) != 0).int()
    else:
        raise TypeError("Input must be a NumPy array or a PyTorch tensor.")
    
    return bin_nums

