# -*- coding: utf-8 -*-
"""
Luotu To 29.1.2026

Tekijä: Sanni Sinisalo

Koodi RTDose tiedoston upsamplaamiseen.
"""

import os
import re
import SimpleITK as sitk
import pydicom
import numpy as np
from pydicom.dataset import FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid
from luokat import BASE_DIR


def patient_number(name):
    m = re.search(r"\d+", name)
    return int(m.group()) if m else 999999


def find_dose_file(folder):
    for f in os.listdir(folder):
        if f.startswith("RD") and f.endswith(".dcm"):
            return folder / f
    return None


def find_ct_files(folder):
    """
    Palauttaa listan CT-viipaleista (täydet polut)
    """
    ct_files = []
    for f in os.listdir(folder):
        if f.endswith(".dcm") and not f.startswith("RD"):
            try:
                ds = pydicom.dcmread(folder / f, stop_before_pixels=True)
                if ds.Modality == "CT":
                    ct_files.append(folder / f)
            except Exception:
                pass
    return ct_files


def load_ct_series_from_files(ct_files):
    """
    Rakentaa 3D-CT-kuvan listasta 2D-viipaleita
    """
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(ct_files)
    return reader.Execute()


if __name__ == "__main__":
    INPUT_ROOT = BASE_DIR / "VN0"
    OUTPUT_ROOT = BASE_DIR / "VN0ds"
    
    # POTILAAT NUMEROJÄRJESTYKSESSÄ
    patients = [
        p 
        for p in os.listdir(INPUT_ROOT)
        if os.path.isdir(INPUT_ROOT / p) and p.lower().startswith("patient")
    ]
    patients.sort(key=patient_number)
    total = len(patients)
    
    
    for idx, patient in enumerate(patients, start=1):
    
        print(f"\n[{idx}/{total}] Käsitellään potilas: {patient}")
    
        patient_in = INPUT_ROOT / patient
        patient_out = OUTPUT_ROOT / patient / "dose"
        os.makedirs(patient_out, exist_ok=True)
    
        dose_path = find_dose_file(patient_in)
        ct_files = find_ct_files(patient_in)
    
        if dose_path is None:
            print(" Dose-tiedostoa ei löytynyt → ohitetaan")
            continue
    
        if len(ct_files) == 0:
            print(" CT-viipaleita ei löytynyt → ohitetaan")
            continue
    
        try:
            # --- LUE CT ---
            ct_img = load_ct_series_from_files(ct_files)
            print(f" CT ladattu ({ct_img.GetSize()[2]} viipaletta)")
    
            # --- LUE DOSE ---
            ds = pydicom.dcmread(dose_path)
            dose_scaling = float(ds.DoseGridScaling)
    
            dose_img = sitk.ReadImage(dose_path, sitk.sitkFloat32)
            dose_img = dose_img * dose_scaling
            print(" Dose ladattu")
    
            # --- REFERENSSI ---
            dose_size = dose_img.GetSize()  # (X, Y, Z)
            dose_spacing = dose_img.GetSpacing() # (sx, sy, sz)
            dose_origin = dose_img.GetOrigin()
            dose_direction = dose_img.GetDirection()
            
            ct_size = ct_img.GetSize() # (X, Y, Z)
            ct_spacing = ct_img.GetSpacing()
            ct_origin = ct_img.GetOrigin()
            ct_direction = ct_img.GetDirection()
            
            # Luo referenssikuva X/Y = CT, Z = dose
            reference = sitk.Image(
                [ct_size[0], ct_size[1], dose_size[2]], sitk.sitkFloat32
                )
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
            ds.Rows = stored_values.shape[1]       # Y
            ds.Columns = stored_values.shape[2]    # X
            ds.NumberOfFrames = stored_values.shape[0]  # Z
            
            ds.BitsAllocated = 16
            ds.BitsStored = 16
            ds.HighBit = 15
            ds.PixelRepresentation = 0  # unsigned
            ds.DoseGridScaling = new_scaling
            ds.PixelSpacing = [ct_spacing[1], ct_spacing[0]]  # HUOM DICOM järjestys!
            ds.SliceThickness = dose_spacing[2]
            ds.GridFrameOffsetVector = [
                i * dose_spacing[2] for i in range(stored_values.shape[0])
            ]
    
            out_path = os.path.join(patient_out, os.path.basename(dose_path))
            
            if not hasattr(ds, "file_meta") or ds.file_meta is None:
                ds.file_meta = FileMetaDataset()
            
            ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
            ds.file_meta.MediaStorageSOPClassUID = ds.SOPClassUID
            ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
            ds.file_meta.ImplementationClassUID = generate_uid()
            
            ds.save_as(out_path, write_like_original=False)
            
            ds2 = pydicom.dcmread(out_path)
            dose_check = ds2.pixel_array * float(ds2.DoseGridScaling)
            print(dose_check.shape)
            print(dose_check.min(), dose_check.max())
    
            print(" Tallennettu onnistuneesti")
    
        except Exception as e:
            print(f" Virhe potilaalla {patient}: {e}")