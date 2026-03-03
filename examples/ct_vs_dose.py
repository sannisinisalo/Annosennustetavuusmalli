# -*- coding: utf-8 -*-
"""
CT–annosgeometrian debuggaus
Täytä CT_DIR ja DOSE_PATH ennen ajoa.
"""

import os
import pydicom
import SimpleITK as sitk


# POLUT

CT_DIR = r"C:\Users\User01\GRADU\Aineisto\VN0ds\Patient1_VN0\vanha ct"
DOSE_PATH = r"C:\Users\User01\GRADU\Aineisto\VN0ds\Patient1_VN0\dose\RD.1.2.246.352.221.4972727230104878982.15806810084685865633.dcm"    


# APUFUNKTIOITA

def z_extent(img):
    origin = img.GetOrigin()
    spacing = img.GetSpacing()
    size = img.GetSize()
    z0 = origin[2]
    z1 = origin[2] + spacing[2] * (size[2] - 1)
    return z0, z1

def load_ct_files(ct_dir):
    files = []
    for f in os.listdir(ct_dir):
        if f.lower().endswith(".dcm"):
            try:
                ds = pydicom.dcmread(os.path.join(ct_dir, f), stop_before_pixels=True)
                if ds.Modality == "CT":
                    files.append(os.path.join(ct_dir, f))
            except:
                pass
    return files


# 3) LATAA CT:T

ct_files = load_ct_files(CT_DIR)
if not ct_files:
    raise RuntimeError("CT-tiedostoja ei löytynyt annetusta kansiosta.")

# Lajittele viipaleet oikeaan järjestykseen
ct_files_sorted = sorted(
    ct_files,
    key=lambda f: pydicom.dcmread(f, stop_before_pixels=True).ImagePositionPatient[2]
)


# DEBUG-TULOSTEET



# --- CT DICOM z-koordinaatit ---
ct_positions = [
    float(pydicom.dcmread(f, stop_before_pixels=True).ImagePositionPatient[2])
    for f in ct_files_sorted
]

print("CT DICOM z-positions (sorted):")
print(ct_positions)
print("CT DICOM z-extent:", ct_positions[0], ct_positions[-1])

# --- SimpleITK CT RAW ---
ct_img_raw = sitk.ImageSeriesReader()
ct_img_raw.SetFileNames(ct_files_sorted)
ct_img_raw = ct_img_raw.Execute()

print("\nCT SimpleITK (raw):")
print("  spacing:", ct_img_raw.GetSpacing())
print("  origin:", ct_img_raw.GetOrigin())
print("  size:", ct_img_raw.GetSize())
print("  z-extent:", z_extent(ct_img_raw))

# --- SimpleITK CT LPS ---
ct_img_lps = sitk.DICOMOrient(ct_img_raw, "LPS")

print("\nCT SimpleITK (LPS):")
print("  spacing:", ct_img_lps.GetSpacing())
print("  origin:", ct_img_lps.GetOrigin())
print("  size:", ct_img_lps.GetSize())
print("  z-extent:", z_extent(ct_img_lps))

# --- Dose RAW ---
dose_img_raw = sitk.ReadImage(DOSE_PATH, sitk.sitkFloat32)

print("\nDose SimpleITK (raw):")
print("  spacing:", dose_img_raw.GetSpacing())
print("  origin:", dose_img_raw.GetOrigin())
print("  size:", dose_img_raw.GetSize())
print("  z-extent:", z_extent(dose_img_raw))

# --- Dose LPS ---
dose_img_lps = sitk.DICOMOrient(dose_img_raw, "LPS")

print("\nDose SimpleITK (LPS):")
print("  spacing:", dose_img_lps.GetSpacing())
print("  origin:", dose_img_lps.GetOrigin())
print("  size:", dose_img_lps.GetSize())
print("  z-extent:", z_extent(dose_img_lps))

