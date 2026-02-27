# -*- coding: utf-8 -*-
"""
Luotu Ke 18.02.2026
Tekijä: Sanni Sinisalo

Koodi, jolla voi vertailla kahden eri maskin dimensioita ja metadataa
"""

import os
import numpy as np
import pydicom
import matplotlib.pyplot as plt
import pandas as pd


# -------------------------------------------------
# 1. Lue DICOM-sarja kansiosta ja muodosta 3D-volyymi
# -------------------------------------------------
def load_dicom_series(folder):
    files = [pydicom.dcmread(os.path.join(folder, f)) 
             for f in os.listdir(folder) 
             if f.endswith(".dcm")]

    # Järjestetään slice orderiin (ImagePositionPatient z)
    files.sort(key=lambda x: float(x.ImagePositionPatient[2]))

    volume = np.stack([f.pixel_array for f in files])

    return volume, files


# -------------------------------------------------
# 2. Kerää keskeinen metadata
# -------------------------------------------------
def extract_metadata(dicom_files):
    first = dicom_files[0]
    
    metadata = {
        "Rows": first.Rows,
        "Columns": first.Columns,
        "PixelSpacing": first.PixelSpacing,
        "SliceThickness": getattr(first, "SliceThickness", None),
        "ImageOrientationPatient": first.ImageOrientationPatient,
        "ImagePositionPatient (first slice)": first.ImagePositionPatient,
        "Modality": first.Modality,
        "SeriesDescription": getattr(first, "SeriesDescription", None)
    }
    
    return metadata


# -------------------------------------------------
# 3. Dice-kerroin
# -------------------------------------------------
def dice_coefficient(mask1, mask2):
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)

    intersection = np.logical_and(mask1, mask2).sum()
    return 2. * intersection / (mask1.sum() + mask2.sum())


# -------------------------------------------------
# 4. VISUAALINEN VERTAILU
# -------------------------------------------------
def show_comparison(mask1, mask2, slice_index=None):
    if slice_index is None:
        slice_index = mask1.shape[0] // 2

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(mask1[slice_index], cmap='gray')
    axes[0].set_title("Maski 1")

    axes[1].imshow(mask2[slice_index], cmap='gray')
    axes[1].set_title("Maski 2")

    diff = mask1[slice_index] - mask2[slice_index]
    axes[2].imshow(diff, cmap='grey')
    axes[2].set_title("Erotus (Mask1 - Mask2)")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


# -------------------------------------------------
# 5. PÄÄOHJELMA
# -------------------------------------------------
folder1 = r"C:\Users\User01\GRADU\Aineisto\VN0ds\Patient1_VN0\ct"
folder2 = r"C:\Users\User01\GRADU\Aineisto\VN0ds\Patient1_VN0\maskids"

mask1, files1 = load_dicom_series(folder1)
mask2, files2 = load_dicom_series(folder2)

print("=== METADATA MASKI 1 (Nova) ===")
meta1 = extract_metadata(files1)
print(pd.Series(meta1))

print("\n=== METADATA MASKI 2 (KYS) ===")
meta2 = extract_metadata(files2)
print(pd.Series(meta2))

# Tarkista että dimensiot täsmäävät
if mask1.shape != mask2.shape:
    raise ValueError("Maskien dimensiot eivät täsmää!")

# Laske Dice
dice = dice_coefficient(mask1, mask2)
print(f"\nDice coefficient: {dice:.4f}")

# Näytä visuaalinen vertailu
show_comparison(mask1, mask2)
