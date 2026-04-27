# -*- coding: utf-8 -*-
"""
Luotu Ke 18.02.2026
Tekijä: Sanni Sinisalo

Koodi, jolla voi vertailla kahden eri RTDose tiedoston dimensioita ja metadataa
"""

import pydicom
import numpy as np
import pandas as pd


# Lue RTDose ja palauta fyysinen annos (Gy)
def load_rtdose(filepath):
    ds = pydicom.dcmread(filepath)

    if ds.Modality != "RTDOSE":
        raise ValueError("Tiedosto ei ole RTDose.")

    dose_grid_scaling = float(ds.DoseGridScaling)
    dose_array = ds.pixel_array.astype(np.float32) * dose_grid_scaling

    return dose_array, ds


# Poimi tärkeä metadata
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


# PÄÄOHJELMA
file1 = r"C:\Users\User01\GRADU\Aineisto\VN0ds\Patient1_VN0\doseds\RD.1.2.246.352.221.4972727230104878982.15806810084685865633.dcm"
file2 = r"C:\Users\User01\GRADU\Aineisto\VN0ds\Patient10_VN0\doseds\RD.1.2.246.352.221.4814537308602544922.6316583374286193844.dcm"

dose1, ds1 = load_rtdose(file1)
dose2, ds2 = load_rtdose(file2)

print(" METADATA RTDOSE 1 ")
meta1 = extract_rtdose_metadata(ds1)
print(pd.Series(meta1))

print("\n METADATA RTDOSE 2 ")
meta2 = extract_rtdose_metadata(ds2)
print(pd.Series(meta2))



