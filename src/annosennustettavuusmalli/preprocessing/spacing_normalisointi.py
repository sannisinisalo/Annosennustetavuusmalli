# -*- coding: utf-8 -*-
"""
Luotu To 26.02.2026 
Tekijä: Sanni Sinisalo

Koodi muuttamaan mallin tarvitsemien tiedostojen pixel_spacing samaksi
"""

# -*- coding: utf-8 -*-
"""
Luotu Ke 26.2.2026
Tekijä: ChatGPT
Tarkoitus: Resample CT, Dose ja maskit samaan spacingiin turvallisesti.
"""

import os
from pathlib import Path
import shutil

import pydicom
import SimpleITK as sitk
from src.annosennustettavuusmalli.preprocessing.luokat import Patient, AllPatients  # oletan että luokat.py on importattavissa

TARGET_SPACING = [1.9531248, 1.9531248]

def resample_image(itk_image, target_spacing, is_label=False):
    """Resample SimpleITK image to target spacing."""
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    new_spacing = list(original_spacing)
    new_spacing[0:2] = target_spacing  # x, y

    new_size = [
        int(round(original_size[i] * (original_spacing[i] / new_spacing[i])))
        for i in range(3)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(itk_image.GetDirection())
    resampler.SetOutputOrigin(itk_image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)

    if is_label:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resampler.SetInterpolator(sitk.sitkLinear)

    return resampler.Execute(itk_image)

def process_ct_and_mask(patient: Patient):
    patient.ct_dir.mkdir(parents=True, exist_ok=True)
    patient.mask_dir.mkdir(parents=True, exist_ok=True)

    # CT-kuvat
    for ct_file in patient.ct_files:
        ds = pydicom.dcmread(ct_file)
        spacing = ds.PixelSpacing
        if list(spacing) != TARGET_SPACING:
            # Muutetaan SimpleITK:llä geometrisesti oikein
            itk_image = sitk.ReadImage(str(ct_file))
            resampled = resample_image(itk_image, TARGET_SPACING)
            sitk.WriteImage(resampled, str(patient.ct_dir / ct_file.name))
        else:
            shutil.copy(ct_file, patient.ct_dir / ct_file.name)

    # Maskit
    mask_files = list((patient.original_dir / "mask*").glob("*"))
    for mask_file in mask_files:
        ds = pydicom.dcmread(mask_file)
        spacing = ds.PixelSpacing
        if list(spacing) != TARGET_SPACING:
            itk_image = sitk.ReadImage(str(mask_file))
            resampled = resample_image(itk_image, TARGET_SPACING, is_label=True)
            sitk.WriteImage(resampled, str(patient.mask_dir / mask_file.name))
        else:
            shutil.copy(mask_file, patient.mask_dir / mask_file.name)

def process_rtdose(patient: Patient):
    patient.doseds_dir.mkdir(parents=True, exist_ok=True)
    dose_file = patient.rtdose_file
    ds = pydicom.dcmread(dose_file)
    spacing = list(ds.PixelSpacing)
    if spacing != TARGET_SPACING:
        itk_image = sitk.ReadImage(str(dose_file))
        resampled = resample_image(itk_image, TARGET_SPACING)
        sitk.WriteImage(resampled, str(patient.doseds_dir / dose_file.name))
    else:
        shutil.copy(dose_file, patient.doseds_dir / dose_file.name)

def main():
    all_patients = AllPatients(processed_dataset="VN0ds", original_dataset="VN0")
    for patient in all_patients.sorted_by_number():
        print(f"Processing patient {patient.patient_folder}")
        process_ct_and_mask(patient)
        process_rtdose(patient)
        print(f"Done patient {patient.patient_folder}")

if __name__ == "__main__":
    main()