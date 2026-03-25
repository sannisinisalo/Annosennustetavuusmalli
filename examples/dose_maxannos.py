# -*- coding: utf-8 -*-
"""
Luotu To 19.03.2026

Tekijä: Sanni Sinisalo
"""

import os
import re
import pydicom
import numpy as np

# Pääkansio
base_path = r"C:\Users\User01\GRADU\Aineisto\VN0ds"

def get_patient_number(folder_name):
    """
    Poimii potilaan numeron nimestä, esim:
    Patient12_VN0 -> 12
    """
    match = re.search(r'Patient(\d+)', folder_name)
    return int(match.group(1)) if match else float('inf')

# Haetaan potilaskansiot ja järjestetään numeron mukaan
patient_folders = [
    f for f in os.listdir(base_path)
    if os.path.isdir(os.path.join(base_path, f))
]

patient_folders_sorted = sorted(patient_folders, key=get_patient_number)

# Käydään potilaat läpi
for patient in patient_folders_sorted:
    dose_folder = os.path.join(base_path, patient, "dose")

    if not os.path.exists(dose_folder):
        print(f"{patient}: dose-kansiota ei löydy")
        continue

    # Oletetaan yksi DICOM-tiedosto
    dicom_files = [
        f for f in os.listdir(dose_folder)
        if f.lower().endswith(".dcm")
    ]

    if not dicom_files:
        print(f"{patient}: ei DICOM-tiedostoa")
        continue

    dicom_path = os.path.join(dose_folder, dicom_files[0])

    try:
        ds = pydicom.dcmread(dicom_path)

        # Dose data (3D)
        dose_array = ds.pixel_array.astype(np.float32)

        # Skaalaus
        scaling = getattr(ds, "DoseGridScaling", 1.0)
        dose_array *= scaling

        max_dose = np.max(dose_array)

        print(f"{patient}: maksimi annos = {max_dose:.4f} Gy")

    except Exception as e:
        print(f"{patient}: virhe tiedoston käsittelyssä -> {e}")