# -*- coding: utf-8 -*-
"""
Luotu To 29.1.2026
Tekijä: Sanni Sinisalo

Koodi RTDose tiedoston upsamplaamiseen.
"""

import SimpleITK as sitk
import pydicom
import numpy as np
from luokat import AllPatients


def load_ct_series_from_files(ct_files):
    """
    Rakentaa 3D-CT-kuvan listasta 2D-viipaleita
    """
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(ct_files)
    return reader.Execute()


if __name__ == "__main__":
    all_patients = AllPatients(processed_dataset="VN0ds", original_dataset="VN0")

    for idx, patient in enumerate(all_patients.sorted_by_number(), start=1):
        print(f"\n[{idx}/{len(all_patients.patients)}] Käsitellään potilas: {patient.patient_folder}")

        # Polut luokan kautta
        ct_files = list(patient.ct_files)
        dose_path = None
        try:
            dose_path = patient.rtdose_file
        except FileNotFoundError:
            print(" Dose-tiedostoa ei löytynyt → ohitetaan")
            continue

        patient_out = patient.dose_dir
        patient_out.mkdir(parents=True, exist_ok=True)

        if not ct_files:
            print(" CT-viipaleita ei löytynyt → ohitetaan")
            continue

        try:
            # --- LUE CT ---
            ct_img = load_ct_series_from_files(ct_files)
            print(f" CT ladattu ({ct_img.GetSize()[2]} viipaletta)")

            # --- LUE DOSE ---
            ds = pydicom.dcmread(dose_path)
            dose_scaling = float(ds.DoseGridScaling)
            dose_img = sitk.ReadImage(dose_path, sitk.sitkFloat32) * dose_scaling
            print(" Dose ladattu")

            # --- REFERENSSI ---
            dose_size = dose_img.GetSize()  # (X,Y,Z)
            dose_spacing = dose_img.GetSpacing()
            dose_origin = dose_img.GetOrigin()
            dose_direction = dose_img.GetDirection()

            ct_size = ct_img.GetSize()
            ct_spacing = ct_img.GetSpacing()
            ct_origin = ct_img.GetOrigin()
            ct_direction = ct_img.GetDirection()

            reference = sitk.Image([ct_size[0], ct_size[1], dose_size[2]], sitk.sitkFloat32)
            reference.SetSpacing([ct_spacing[0], ct_spacing[1], dose_spacing[2]])
            reference.SetOrigin([ct_origin[0], ct_origin[1], dose_origin[2]])
            reference.SetDirection(dose_direction)

            # --- RESAMPLAA DOSE ---
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(reference)
            resampler.SetInterpolator(sitk.sitkLinear)
            resampler.SetDefaultPixelValue(0.0)
            dose_resampled = resampler.Execute(dose_img)
            print(" Resamplaus valmis")

            # --- TALLENNUS ---
            dose_array = sitk.GetArrayFromImage(dose_resampled)
            new_scaling = 0.001
            stored_values = np.round(dose_array / new_scaling).astype(np.uint16)

            ds.PixelData = stored_values.tobytes()
            ds.Rows = stored_values.shape[1]
            ds.Columns = stored_values.shape[2]
            ds.NumberOfFrames = stored_values.shape[0]
            ds.BitsAllocated = 16
            ds.BitsStored = 16
            ds.HighBit = 15
            ds.PixelRepresentation = 0
            ds.DoseGridScaling = new_scaling

            out_path = patient_out / dose_path.name
            ds.save_as(out_path)

            ds2 = pydicom.dcmread(out_path)
            dose_check = ds2.pixel_array * float(ds2.DoseGridScaling)
            print(dose_check.shape)
            print(dose_check.min(), dose_check.max())

            print(" Tallennettu onnistuneesti")

        except Exception as e:
            print(f" Virhe potilaalla {patient.patient_folder}: {e}")
