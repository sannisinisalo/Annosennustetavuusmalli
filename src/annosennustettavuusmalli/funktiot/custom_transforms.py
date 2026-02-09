import numpy as np
import torchio as tio
from torchvision.transforms.functional import affine
from torchvision.transforms.functional import InterpolationMode
import matplotlib.pyplot as plt
from einops import rearrange
from .integer_mask_to_binary import integer_mask_to_binary
import torch
from scipy.ndimage import distance_transform_edt

"""
Tekijä: Akseli Leino
"""

class ProbabilityMapTransform(tio.transforms.Transform):
    """Custom transform that creates probability map that can be used for weighted sampling.
    The idea of the transform is to create a volume with same size as patient volume. Each slice will be constant value that is determined by OARs that are present.
    """
    def apply_transform(self, subject: tio.Subject)->tio.Subject:
        mask = subject['mask'][tio.DATA]
        probability_map = torch.empty_like(mask)
        
        # Images are (c, w, h, d), we want to iterate depth.
        for i in range(probability_map.shape[3]):
            current_slice = mask[:, :, :, i]
        
            if torch.any(current_slice == 1):
                probability_map[:, :, :, i] = 0.6
            else:
                probability_map[:, :, :, i] = 0.2
        
        subject['probability_map'].set_data(probability_map)
        
        return subject
    

class CreateDistanceToPTV(tio.transforms.Transform):
    """Custom transform for TorchIO that creates distance to PTV map.
    """
    def apply_transform(self, subject: tio.Subject)->tio.Subject:
        mask = subject['mask'][tio.DATA]
        not_PTV_mask = mask != 1 # distance_transform_edt calcualates distance to 0, not 1. Thus we need mask that is NOT PTV.
        body_mask = mask != -1 # Outside of body is -1
        slice_thickness_mm = 3 # Change this to your slice thickness - Cyberknife prostatoille 1
        distance_to_PTV = torch.multiply(torch.tensor(distance_transform_edt(not_PTV_mask, sampling=(1,slice_thickness_mm,subject.pixel_spacing,subject.pixel_spacing))), body_mask)
        subject['distance_to_PTV'].set_data(distance_to_PTV/400) #400 is an arbitrary number to approximately normalize the distances to [0, 1]
        return subject

class DoseScalingTransform(tio.transforms.Transform):
    """Custom transform for TorchIO that scales the dose data.
    """
    def apply_transform(self, subject: tio.Subject)->tio.Subject:
        dose = subject['dose'][tio.DATA]
        dose_multiplier = subject['dose_multiplier']
        dose_tensor_transformed = dose.data*dose_multiplier
        subject['dose'].set_data(dose_tensor_transformed)
        return subject

class FlipRightTransform(tio.transforms.Transform):
    """This is custom transform to TorchIO that flips dose, mask, and CT data. Basically this enables combining left and right data to same dataset (i.e. right dataset is put through this to make it "left")
    """
    def apply_transform(self, subject: tio.Subject)->tio.Subject:
        if subject['flip'] == True:
            # Data has dimensions (c, w, h, d), thus we want to flip the second (1) dimension.
            transformed_dose = subject['dose'][tio.DATA].flip(1)
            transformed_mask = subject['mask'][tio.DATA].flip(1)
            transformed_ct = subject['ct'][tio.DATA].flip(1)
            subject['dose'].set_data(transformed_dose)
            subject['ct'].set_data(transformed_ct)
            subject['mask'].set_data(transformed_mask)
            subject['original_mask'].set_data(transformed_mask)
        return subject


class PixelSizingTransform(tio.transforms.Transform):
    """This is transform for standardizing pixel spacing throughout the dataset.

    The affine function of torchvision is only meant for 2D images. However, as we want to resize only pixel size in xy-slice plane, it does not matter that we have 3D data.
    """
    def apply_transform(self, subject: tio.Subject)->tio.Subject:
        dose = subject['dose'][tio.DATA]
        mask = subject['mask'][tio.DATA]
        ct = subject['ct'][tio.DATA]

        scale = subject['pixel_spacing']
        scale = scale/2 # Divison by two is due to x2 downsampling made for the data

        # As affine needs to have H and W as last dimensions, we need to rearrange tensors so that depth (or z-axis) is not last.
        ct = rearrange(ct, 'c w h d -> c d w h')
        mask = rearrange(mask, 'c w h d -> c d w h')
        dose = rearrange(dose, 'c w h d -> c d w h')

        transformed_dose = affine(dose, angle = 0, translate = (0,0), shear = 0, scale = scale, interpolation = InterpolationMode.BILINEAR, fill = 0)
        transformed_ct = affine(ct, angle = 0, translate = (0,0), shear = 0, scale = scale, interpolation = InterpolationMode.BILINEAR, fill = 0)
        transformed_mask = affine(mask, angle = 0, translate = (0,0), shear = 0, scale = scale, interpolation = InterpolationMode.NEAREST, fill = -1)

        transformed_ct = rearrange(transformed_ct, 'c d w h -> c w h d')
        transformed_mask = rearrange(transformed_mask, 'c d w h -> c w h d')
        transformed_dose = rearrange(transformed_dose, 'c d w h -> c w h d')
        
        subject['dose'].set_data(transformed_dose)
        subject['ct'].set_data(transformed_ct)
        subject['mask'].set_data(transformed_mask)

        return subject

class CreateInputMask(tio.transforms.Transform):
    """Input mask is stored as binary data converted to 10-base. This transform creates a mask that has overlaps removed and leaves higher priority masks on top.
    """
    def apply_transform(self, subject: tio.Subject)->tio.Subject:
        mask = subject['mask'][tio.DATA]
        rounded_mask = torch.round(mask).to(dtype=torch.int)
        binary_mask = integer_mask_to_binary(rounded_mask, num_bits=11)
        final_mask = torch.zeros(mask.shape)

        # NOTE: THIS IS THE PRIORITY ORDER YOU WANT FOR YOUR OARs, LAST ONE WILL BE USED ON TOP OF EVERYTHING ELSE IN CASE OF OVERLAPPING
        # Current order is so that important one have high priority, PTV almost highest, and brachial plexus and LAD on top of everything
        # due to their nature.
        # This order leaves highest priorities - TODO: read these from config
        for channel in [10, 9, 8, 6, 4, 3, 2, 1, 0, 5, 7]:
            final_mask[binary_mask[:, :, :, :, channel] == 1] = channel + 1
        final_mask[torch.isclose(rounded_mask[:, :, :, :], torch.tensor([-1], dtype=torch.int))] = -1

        subject['mask'].set_data(final_mask)
        subject['original_mask'].set_data(rounded_mask)

        return subject
