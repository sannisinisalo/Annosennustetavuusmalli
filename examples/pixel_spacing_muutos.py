# -*- coding: utf-8 -*-
"""
Luotu To 12.2.2026
Tekijä: Sanni Sinisalo

Koodi, jolla potilaiden CT-kuvien ja maskin pixel_spacing voidaan muokata 
halutuun
"""

import os
import SimpleITK as sitk
import numpy as np

# Haluttu pixel spacing (XY)
TARGET_SPACING_2D = np.array([1.953125, 1.953125])

# Polku potilasaineistoon
BASE_PATH = r"C:/Users/User01/GRADU/Aineisto/VN0ds"


def needs_resampling_2d(image, target_spacing):
    spacing = np.array(image.GetSpacing()[:2])
    return not np.allclose(spacing, target_spacing, rtol=1e-4, atol=1e-4)


def resample_2d(image, target_spacing, is_mask=False):
    original_spacing = np.array(image.GetSpacing())
    original_size = np.array(image.GetSize(), dtype=int)

    # 2D laskenta vain XY-suunnassa
    new_size_xy = (original_size[:2] * (original_spacing[:2] / target_spacing)).astype(int)
    new_size_xy[new_size_xy < 1] = 1

    # Interpolointi
    interpolator = sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear

    resampler = sitk.ResampleImageFilter()

    # 3D spacing: XY muuttuu, Z säilyy
    new_spacing = [target_spacing[0], target_spacing[1], original_spacing[2]]
    resampler.SetOutputSpacing(new_spacing)

    # 3D size: XY muuttuu, Z = 1
    new_size = [int(new_size_xy[0]), int(new_size_xy[1]), 1]
    resampler.SetSize(new_size)

    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetInterpolator(interpolator)

    return resampler.Execute(image)


def process_folder_2d(input_folder, output_folder, is_mask=False):
    os.makedirs(output_folder, exist_ok=True)

    for fname in os.listdir(input_folder):
        if not fname.lower().endswith(".dcm"):
            continue

        input_file = os.path.join(input_folder, fname)
        output_file = os.path.join(output_folder, fname)

        img = sitk.ReadImage(input_file)

        if not needs_resampling_2d(img, TARGET_SPACING_2D):
            sitk.WriteImage(img, output_file)
            continue

        print(f"  Resamplataan 2D-leike: {input_file}")
        resampled = resample_2d(img, TARGET_SPACING_2D, is_mask=is_mask)

        sitk.WriteImage(resampled, output_file)


def main():
    for folder in os.listdir(BASE_PATH):
        if folder.startswith("Patient") and folder.endswith("_VN0"):
            patient_dir = os.path.join(BASE_PATH, folder)
            print(f"Käsitellään potilas: {patient_dir}")

            ct_dir = os.path.join(patient_dir, "ct")
            mask_dir = os.path.join(patient_dir, "maskids")

            ct_out = os.path.join(patient_dir, "ct_resampled")
            mask_out = os.path.join(patient_dir, "maskids_resampled")

            if os.path.isdir(ct_dir):
                process_folder_2d(ct_dir, ct_out, is_mask=False)

            if os.path.isdir(mask_dir):
                process_folder_2d(mask_dir, mask_out, is_mask=True)


if __name__ == "__main__":
    main()