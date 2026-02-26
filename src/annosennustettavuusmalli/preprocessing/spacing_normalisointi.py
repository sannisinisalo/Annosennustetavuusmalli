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

import shutil
import pydicom
import SimpleITK as sitk
from luokat import Patient, AllPatients

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

# --------------------------
# CT ja maskit
# --------------------------
def process_ct_and_mask(patient: Patient):
    # CT-kuvat
    for ct_file in patient.ct_dir.glob("*.dcm"):
        ds = pydicom.dcmread(ct_file)
        if list(ds.PixelSpacing) != TARGET_SPACING:
            itk_image = sitk.ReadImage(str(ct_file))
            resampled = resample_image(itk_image, TARGET_SPACING)
            sitk.WriteImage(resampled, str(ct_file))  # tallennetaan päälle
    # Maskit
    for mask_file in patient.maskds_dir.glob("*.dcm"):
        ds = pydicom.dcmread(mask_file)
        if list(ds.PixelSpacing) != TARGET_SPACING:
            itk_image = sitk.ReadImage(str(mask_file))
            resampled = resample_image(itk_image, TARGET_SPACING, is_label=True)
            sitk.WriteImage(resampled, str(mask_file))  # tallennetaan päälle

# --------------------------
# RTDose
# --------------------------
def process_rtdose(patient: Patient):
    dose_file = list(patient.doseds_dir.glob("*.dcm"))[0]  # oletetaan yksi dose
    ds = pydicom.dcmread(dose_file, force=True)
    spacing = list(ds.PixelSpacing)

    if spacing != TARGET_SPACING:
        dose_array = ds.pixel_array.astype("float32")
        itk_image = sitk.GetImageFromArray(dose_array)
        original_spacing = [float(ds.PixelSpacing[0]), float(ds.PixelSpacing[1]),
                            float(ds.GridFrameOffsetVector[1] - ds.GridFrameOffsetVector[0])]
        itk_image.SetSpacing(original_spacing[::-1])
        resampled_itk = resample_image(itk_image, TARGET_SPACING)
        resampled_array = sitk.GetArrayFromImage(resampled_itk)

        ds.Rows, ds.Columns = resampled_array.shape[1], resampled_array.shape[2]
        ds.PixelSpacing = TARGET_SPACING
        ds.PixelData = resampled_array.astype("float32").tobytes()
        ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

        ds.save_as(dose_file)  # tallennetaan päälle

# --------------------------
# Koko potilasjoukko
# --------------------------
def main():
    all_patients = AllPatients(processed_dataset="VN0ds", original_dataset="VN0")
    for patient in all_patients.sorted_by_number():
        print(f"Processing patient {patient.patient_folder}")
        process_ct_and_mask(patient)
        process_rtdose(patient)
        print(f"Done patient {patient.patient_folder}")

if __name__ == "__main__":
    main()