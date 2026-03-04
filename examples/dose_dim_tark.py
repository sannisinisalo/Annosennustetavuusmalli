# -*- coding: utf-8 -*-
"""
Luotu Ke 18.02.2026
Tekijä: Sanni Sinisalo

Koodi, jolla voi vertailla kahden eri RTDose tiedoston dimensioita ja metadataa
"""

import pydicom
import numpy as np
import pandas as pd


# -------------------------------------------------
# 1. Lue RTDose ja palauta fyysinen annos (Gy)
# -------------------------------------------------
def load_rtdose(filepath):
    ds = pydicom.dcmread(filepath)

    if ds.Modality != "RTDOSE":
        raise ValueError("Tiedosto ei ole RTDose.")

    dose_grid_scaling = float(ds.DoseGridScaling)
    dose_array = ds.pixel_array.astype(np.float32) * dose_grid_scaling

    return dose_array, ds


# -------------------------------------------------
# 2. Poimi tärkeä metadata
# -------------------------------------------------
def extract_rtdose_metadata(ds):

    metadata = {
        "Modality": ds.Modality,
        "DoseUnits": ds.DoseUnits,
        "Rows": ds.Rows,
        "Columns": ds.Columns,
        "NumberOfFrames": getattr(ds, "NumberOfFrames", None),
        "PixelSpacing": ds.PixelSpacing,
        "SliceThickness": getattr(ds, "SliceThickness", None),
        "GridFrameOffsetVector (len)": len(ds.GridFrameOffsetVector),
        "ImagePositionPatient": ds.ImagePositionPatient,
        "ImageOrientationPatient": ds.ImageOrientationPatient
    }

    return metadata


# -------------------------------------------------
# 3. Annosstatistiikka
# -------------------------------------------------
def dose_statistics(dose):

    stats = {
        "Min (Gy)": np.min(dose),
        "Max (Gy)": np.max(dose),
        "Mean (Gy)": np.mean(dose),
        "Median (Gy)": np.median(dose),
        "Std (Gy)": np.std(dose)
    }

    return stats


# -------------------------------------------------
# 4. Erotusanalyysi
# -------------------------------------------------
def dose_difference_analysis(dose1, dose2):

    if dose1.shape != dose2.shape:
        raise ValueError("Dose-gridien dimensiot eivät täsmää.")

    diff = dose1 - dose2

    stats = {
        "Mean difference (Gy)": np.mean(diff),
        "Max absolute difference (Gy)": np.max(np.abs(diff)),
        "RMSE (Gy)": np.sqrt(np.mean(diff**2))
    }

    return stats


# -------------------------------------------------
# 5. PÄÄOHJELMA
# -------------------------------------------------
file1 = r"C:\Users\User01\GRADU\Aineisto\VN0ds\Patient1_VN0\dose\RD.1.2.246.352.221.4972727230104878982.15806810084685865633.dcm"
file2 = r"C:\Users\User01\GRADU\Aineisto\VN0ds\Patient60_VN0\dose\RD.1.2.246.352.221.5448679770797535778.17401886329480751763.dcm"

dose1, ds1 = load_rtdose(file1)
dose2, ds2 = load_rtdose(file2)

print("=== METADATA RTDOSE 1 ===")
meta1 = extract_rtdose_metadata(ds1)
print(pd.Series(meta1))

print("\n=== METADATA RTDOSE 2 ===")
meta2 = extract_rtdose_metadata(ds2)
print(pd.Series(meta2))

print("\n=== ANNOSSTATISTIIKKA 1 ===")
print(pd.Series(dose_statistics(dose1)))

print("\n=== ANNOSSTATISTIIKKA 2 ===")
print(pd.Series(dose_statistics(dose2)))


