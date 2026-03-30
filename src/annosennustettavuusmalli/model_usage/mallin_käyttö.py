# -*- coding: utf-8 -*-
"""
Luotu Ti 17.03.2026
Tekijä: Sanni Sinisalo

Koodi, jolla kehitettyä mallia visualisoidaan
"""

import os
import torch
import torchio as tio
from pydicom import dcmread
from einops import rearrange
from scipy.ndimage import zoom, distance_transform_edt
from torchvision.transforms.functional import affine, InterpolationMode
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent)) 

from annosennustettavuusmalli.models.unet3plus_3d import UNet3plus_3d
from annosennustettavuusmalli.utils.custom_transforms import (
    DoseScalingTransform,
    PixelSizingTransform,
    CreateInputMask,
    CreateDistanceToPTV,
    ProbabilityMapTransform
)


# Polut
CASE_PATH = r"C:\Users\User01\GRADU\Aineisto\VN0ds\Patient1_VN0"
MODEL_PATH = r"C:\Users\User01\GRADU\GitHub-koodit\Annosennustetavuusmalli\src\annosennustettavuusmalli\training\trained_models\gregarious-chimp-691_epoch_16.pth"

device = torch.device("cpu")


# Ladataan data samalla tavalla kuin generate_datasets.py:ssä
ct_path = os.path.join(CASE_PATH, "ct")
mask_path = os.path.join(CASE_PATH, "maskids")
dose_path = os.path.join(CASE_PATH, "doseds")

dose_file = os.listdir(dose_path)[0]
ds_dose = dcmread(os.path.join(dose_path, dose_file))
num_samples = int(ds_dose.NumberOfFrames)
dose_multiplier = float(ds_dose.DoseGridScaling)

ct_file = os.listdir(ct_path)[0]
ds_ct = dcmread(os.path.join(ct_path, ct_file))
pixel_spacing = float(list(ds_ct.PixelSpacing)[0])

ct_data = tio.ScalarImage(ct_path)
mask_data = tio.ScalarImage(mask_path)
dose_data = tio.ScalarImage(dose_path)

ct_data.set_data(ct_data.data.float())
dose_data.set_data(dose_data.data.float())

subject = tio.Subject(
    ct = ct_data,
    mask = mask_data,
    original_mask = tio.ScalarImage(mask_path),
    dose = dose_data,
    name = "inference_case",
    dose_multiplier = dose_multiplier,
    num_samples = num_samples,
    pixel_spacing = pixel_spacing,
    distance_to_PTV = tio.ScalarImage(mask_path),
    probability_map = tio.ScalarImage(mask_path),
)



# Transformit
rescale_dose = DoseScalingTransform()
rescale_ct = tio.RescaleIntensity(
    out_min_max=(0, 4),
    in_min_max=(-1024, 3072),
    include=["ct"]
)
rescale_pixels = PixelSizingTransform()
create_final_mask = CreateInputMask()
create_distance_to_PTV = CreateDistanceToPTV()
create_probability_map = ProbabilityMapTransform()

transforms = tio.Compose((
    rescale_dose,
    rescale_ct,
    rescale_pixels,
    create_final_mask,
    create_probability_map,
    create_distance_to_PTV
))

dataset = tio.SubjectsDataset([subject], transform=transforms)



# Alustetaan malli
model = UNet3plus_3d(
    in_channels=3,
    out_channels=1,
    filters=[64, 128, 256, 512, 1024],
    kernel_size=3,
    skip_filters=64,
    pool_size=2,
    conv_layers=2,
    act_func="ReLU",
    patch_size=(256, 256, 1)
)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()


# Inference
with torch.no_grad():
    for subject in dataset:

        sampler = tio.data.GridSampler(subject, (256, 256, 1))
        loader = torch.utils.data.DataLoader(sampler, batch_size=4)


        aggregator = tio.inference.GridAggregator(sampler)

        for batch in loader:
            ct = batch["ct"][tio.DATA].float().to(device)
            # mask shape: (B, 1, W, H, D)
            mask = batch["mask"][tio.DATA].float().to(device)

            # PTV = kaikki positiiviset arvot
            ptv_mask = (mask > 0).float()
            
            # Distance map jokaiselle batchin slice:lle
            distance_list = []
            
            for b in range(ptv_mask.shape[0]):
                ptv_np = ptv_mask[b, 0].cpu().numpy()  # (W, H, 1)
                ptv_np2d = ptv_np[:, :, 0]
            
                dist2d = distance_transform_edt(ptv_np2d == 0)
                dist2d = dist2d / 400.0
            
                dist_tensor = torch.tensor(dist2d, dtype=torch.float32)[None, :, :, None]
                distance_list.append(dist_tensor)
            
            distance = torch.stack(distance_list, dim=0).to(device)
            
            # Probability map
            probability = torch.where(ptv_mask > 0, 0.75, 0.25)


            locations = batch[tio.LOCATION]

            inputs = torch.cat((ct, distance, probability), dim=1)


            outputs = model(inputs).main_output

            aggregator.add_batch(outputs, locations)

        pred_dose = aggregator.get_output_tensor()


# Postprocess


# Mallin output TorchIO-tilassa
pred_dose = aggregator.get_output_tensor()  # shape (1, W, H, D)

# Tee siitä TorchIO-kuva käyttäen samaa affinea kuin subject["ct"]:llä
pred_dose_img = tio.ScalarImage(
    tensor=pred_dose,
    affine=subject["ct"].affine
)

# Lataa alkuperäinen CT (se, johon haluat visualisoida)
original_ct_path = os.path.join(CASE_PATH, "vanha ct")
original_ct = tio.ScalarImage(original_ct_path)

# Resamplaa annos alkuperäisen CT:n ruudukkoon
pred_on_orig = tio.Resample(original_ct)(pred_dose_img)

ct_np = original_ct.data.numpy().squeeze()        # (W, H, D)
dose_np = pred_on_orig.data.numpy().squeeze()     # (W, H, D)

# Muoto (D, H, W) matplotlibia varten
ct_np = np.transpose(ct_np, (2, 1, 0))
dose_np = np.transpose(dose_np, (2, 1, 0))

print("CT:", ct_np.shape)
print("Dose:", dose_np.shape)

num_slices = ct_np.shape[0]

for i in range(num_slices):
    plt.figure(figsize=(8,8))
    plt.imshow(ct_np[i], cmap="gray")
    plt.imshow(dose_np[i], cmap="jet", alpha=0.4)
    plt.title(f"Slice {i+1}/{num_slices}")
    plt.axis("off")
    plt.show()

