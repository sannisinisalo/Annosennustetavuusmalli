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
import warnings
import SimpleITK as sitk

from luokat import AllPatients

# -----------------------------
# Asetukset
# -----------------------------
TARGET_SPACING = (1.953125, 1.953125, 2.0)  # mm, x,y,z
INTERPOLATOR_CT_DOSE = sitk.sitkLinear
INTERPOLATOR_MASK = sitk.sitkNearestNeighbor

# -----------------------------
# Funktiot
# -----------------------------
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def resample_image(itk_img, new_spacing_xy, interpolator=sitk.sitkLinear):
    """
    Resample a 2D image slice (x,y) safely.
    """
    spacing_orig = itk_img.GetSpacing()
    size_orig = itk_img.GetSize()

    # Uusi koko x,y, säilytä z
    new_size = [
        int(round(size_orig[0] * (spacing_orig[0] / new_spacing_xy[0]))),
        int(round(size_orig[1] * (spacing_orig[1] / new_spacing_xy[1]))
        )
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing((new_spacing_xy[0], new_spacing_xy[1]))
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(itk_img.GetDirection())
    resampler.SetOutputOrigin(itk_img.GetOrigin())
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(0)

    return resampler.Execute(itk_img)

# -----------------------------
# Suorita resample kaikille potilaille
# -----------------------------
if __name__ == "__main__":
    dataset_original = "VN0"
    dataset_processed = "VN0ds"
    all_patients = AllPatients(processed_dataset=dataset_processed, original_dataset=dataset_original)

    for patient in all_patients.sorted_by_number():
        print(f"\nProcessing patient {patient.patient_folder}...")

        # Luo output-kansiot uusille normalisoiduille tiedostoille
        ct_out_dir = patient.ct_dir / "ct_norm"
        dose_out_dir = patient.doseds_dir / "doseds_norm"
        mask_out_dir = patient.maskds_dir / "maskids_norm"
        for folder in [ct_out_dir, dose_out_dir, mask_out_dir]:
            ensure_dir(folder)

        # -----------------------------
        # 1. CT
        # -----------------------------
        ct_sitk_list = []
        for f in patient.ct_dir.glob("*"):  # Lähdetään koko CT-kansiosta
            try:
                itk_img = sitk.ReadImage(str(f))
                print(f"CT {f.name} spacing ennen: {itk_img.GetSpacing()}")
                itk_resampled = resample_image(itk_img, TARGET_SPACING, INTERPOLATOR_CT_DOSE)
                print(f"CT {f.name} spacing jälkeen: {itk_resampled.GetSpacing()}\n")
                out_path = ct_out_dir / f.name
                sitk.WriteImage(itk_resampled, str(out_path))
                ct_sitk_list.append(itk_resampled)
            except Exception as e:
                warnings.warn(f"{patient.patient_folder}: CT {f.name} resample epäonnistui: {e}")

        if not ct_sitk_list:
            warnings.warn(f"{patient.patient_folder}: CT:tä ei löytynyt, hypätään dose/mask")
            continue
        ct_ref = ct_sitk_list[0]

        # -----------------------------
        # 2. RTDose
        # -----------------------------
        try:
            dose_file = patient.doseds_dir.glob("*")  # Kaikki doseds tiedostot
            for f in dose_file:
                dose_itk = sitk.ReadImage(str(f))
                dose_resampled = resample_image(dose_itk, TARGET_SPACING, INTERPOLATOR_CT_DOSE)
                out_path = dose_out_dir / f.name
                sitk.WriteImage(dose_resampled, str(out_path))
        except Exception as e:
            warnings.warn(f"{patient.patient_folder}: Dose resample epäonnistui: {e}")

        # -----------------------------
        # 3. Maskit
        # -----------------------------
        for f in patient.maskds_dir.glob("*"):  # Kaikki maskit
            try:
                mask_itk = sitk.ReadImage(str(f))
                print(f"Mask {f.name} spacing ennen: {mask_itk.GetSpacing()}")
                mask_resampled = resample_image(mask_itk, TARGET_SPACING, INTERPOLATOR_MASK)
                print(f"Mask {f.name} spacing jälkeen: {mask_resampled.GetSpacing()}\n")
                out_path = mask_out_dir / f.name
                sitk.WriteImage(mask_resampled, str(out_path))
            except Exception as e:
                warnings.warn(f"{patient.patient_folder}: Mask {f.name} resample epäonnistui: {e}")

        print(f"{patient.patient_folder} done.")