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

import SimpleITK as sitk
import numpy as np
from pathlib import Path
import pydicom
from luokat import AllPatients

TARGET_SPACING_XY = [1.953125, 1.953125]


# -----------------------------
# Apufunktio: resample
# -----------------------------
def resample_image(itk_image, target_spacing_xy, interpolator):
    original_pixel_type = itk_image.GetPixelID()
    itk_image = sitk.Cast(itk_image, sitk.sitkFloat32)
    
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    new_spacing = [
        target_spacing_xy[0],
        target_spacing_xy[1],
        original_spacing[2]
    ]

    new_size = [
        int(round(original_size[0] * original_spacing[0] / new_spacing[0])),
        int(round(original_size[1] * original_spacing[1] / new_spacing[1])),
        original_size[2]
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(interpolator)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(itk_image.GetDirection())
    resampler.SetOutputOrigin(itk_image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)

    resampled = resampler.Execute(itk_image)
    return sitk.Cast(resampled, original_pixel_type)


# -----------------------------
# CT:n käsittely
# -----------------------------
def process_ct(patient):
    print("  -> CT")

    out_dir = patient.modified_dir / "ct_norm"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(patient.ct_dir.glob("*.dcm"))
    if not files:
        print("     Ei CT-tiedostoja!")
        return

    reader = sitk.ImageSeriesReader()
    reader.SetFileNames([str(f) for f in files])
    image = reader.Execute()

    original_spacing_xy = image.GetSpacing()[:2]

    if np.allclose(original_spacing_xy, TARGET_SPACING_XY, atol=1e-3):
        print("     spacing jo oikein")
        # Kirjoitetaan vain kopio
        for i, f in enumerate(files):
            sitk.WriteImage(sitk.ReadImage(str(f)), str(out_dir / f.name))
        return

    resampled = resample_image(image, TARGET_SPACING_XY, sitk.sitkLinear)

    for i in range(resampled.GetDepth()):
        slice_i = resampled[:, :, i]
        sitk.WriteImage(slice_i, str(out_dir / f"CT_{i:04d}.dcm"))

    print("     valmis")


# -----------------------------
# Maskien käsittely
# -----------------------------
def process_masks(patient):
    print("  -> Maskit")

    out_dir = patient.modified_dir / "mask_norm"
    out_dir.mkdir(parents=True, exist_ok=True)

    mask_files = sorted(patient.mask_dir.glob("*.dcm"))
    if not mask_files:
        print("     Ei maskeja!")
        return

    for mask_path in mask_files:
        image = sitk.ReadImage(str(mask_path))
        original_spacing_xy = image.GetSpacing()[:2]

        if np.allclose(original_spacing_xy, TARGET_SPACING_XY, atol=1e-3):
            sitk.WriteImage(image, str(out_dir / mask_path.name))
            continue

        resampled = resample_image(image, TARGET_SPACING_XY, sitk.sitkNearestNeighbor)
        sitk.WriteImage(resampled, str(out_dir / mask_path.name))

    print("     valmis")


# -----------------------------
# RTDose käsittely
# -----------------------------
def process_dose(patient):
    print("  -> RTDose")

    out_dir = patient.modified_dir / "dose_norm"
    out_dir.mkdir(parents=True, exist_ok=True)

    dose_path = patient.doseds_dir
    ds = pydicom.dcmread(dose_path)

    original_spacing_xy = [float(x) for x in ds.PixelSpacing]
    if np.allclose(original_spacing_xy, TARGET_SPACING_XY, atol=1e-3):
        ds.save_as(out_dir / dose_path.name)
        print("     spacing jo oikein")
        return

    # Muunna array floatiksi
    dose_array = ds.pixel_array.astype(np.float32) * float(ds.DoseGridScaling)

    image = sitk.GetImageFromArray(dose_array)
    spacing = [
        float(ds.PixelSpacing[1]),
        float(ds.PixelSpacing[0]),
        float(ds.GridFrameOffsetVector[1] - ds.GridFrameOffsetVector[0])
    ]
    image.SetSpacing(spacing)

    resampled = resample_image(image, TARGET_SPACING_XY, sitk.sitkLinear)
    resampled_array = sitk.GetArrayFromImage(resampled)

    # Uusi scaling ja uint16
    max_dose = np.max(resampled_array)
    new_scaling = max_dose / 65535.0
    stored_array = (resampled_array / new_scaling).astype(np.uint16)

    # Päivitä DICOM
    ds.Rows = stored_array.shape[1]
    ds.Columns = stored_array.shape[2]
    ds.PixelSpacing = [str(TARGET_SPACING_XY[0]), str(TARGET_SPACING_XY[1])]
    ds.DoseGridScaling = new_scaling
    ds.PixelData = stored_array.tobytes()
    z_spacing = resampled.GetSpacing()[2]
    ds.GridFrameOffsetVector = [str(i * z_spacing) for i in range(stored_array.shape[0])]

    ds.save_as(out_dir / dose_path.name)
    print("     valmis")


# -----------------------------
# Yksi potilas
# -----------------------------
def process_patient(patient):
    print(f"Käsitellään: {patient.patient_folder}")
    process_ct(patient)
    process_masks(patient)
    process_dose(patient)
    print("Potilas valmis.")


# -----------------------------
# Pääohjelma
# -----------------------------
def main():
    patients = AllPatients(
        processed_dataset="VN0ds",
        original_dataset="VN0"
    )

    for patient in patients.sorted_by_number():
        process_patient(patient)

    print("\nKaikki potilaat käsitelty.")


if __name__ == "__main__":
    main()