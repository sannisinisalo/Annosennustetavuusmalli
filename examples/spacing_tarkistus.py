# -*- coding: utf-8 -*-
"""
Luotu Ke 11.2.2026
Tekijä: Sanni Sinisalo

Koodi, jolla tarkastellaan RTDose, CT ja maskien pixel_spacing
"""

import os
import re
import SimpleITK as sitk

base_path = r"C:/Users/User01/GRADU/Aineisto/VN0ds"

# Funktio, joka poimii potilasnumeron nimestä (esim. Patient12_VN0 → 12)
def extract_patient_number(name):
    match = re.search(r'Patient(\d+)', name)
    return int(match.group(1)) if match else float('inf')

patients = [d for d in os.listdir(base_path) 
            if os.path.isdir(os.path.join(base_path, d))]

# 🔹 Lajitellaan numeron perusteella
patients = sorted(patients, key=extract_patient_number)

for patient in patients:
    patient_path = os.path.join(base_path, patient)
    
    ct_path = os.path.join(patient_path, "ct")
    mask_path = os.path.join(patient_path, "maskids")
    dose_path = os.path.join(patient_path, "doseds")
    
    # Lue dose (yksi 3D-kuva)
    dose_files = [f for f in os.listdir(dose_path) if os.path.isfile(os.path.join(dose_path, f))]
    if not dose_files:
        print(f"{patient}: Dose-kansio tyhjä!")
        continue
    dose_img = sitk.ReadImage(os.path.join(dose_path, dose_files[0]))
    dose_spacing = dose_img.GetSpacing()  # (x, y, z)
    
    # Lue CT
    ct_files = sorted([f for f in os.listdir(ct_path) if os.path.isfile(os.path.join(ct_path, f))])
    if not ct_files:
        print(f"{patient}: CT-kansio tyhjä!")
        continue
    first_ct = sitk.ReadImage(os.path.join(ct_path, ct_files[0]))
    ct_spacing = (first_ct.GetSpacing()[0], first_ct.GetSpacing()[1], dose_spacing[2])
    
    # Lue maski
    mask_files = sorted([f for f in os.listdir(mask_path) if os.path.isfile(os.path.join(mask_path, f))])
    if not mask_files:
        print(f"{patient}: Maski-kansio tyhjä!")
        continue
    first_mask = sitk.ReadImage(os.path.join(mask_path, mask_files[0]))
    mask_spacing = (first_mask.GetSpacing()[0], first_mask.GetSpacing()[1], dose_spacing[2])
    
    # Tarkista, ovatko spacingit samoja
    spacing_match = (ct_spacing == mask_spacing == dose_spacing)
    
    # Tulosta
    print(f"{patient}:")
    print(f"  CT spacing:    {ct_spacing}")
    print(f"  Mask spacing:  {mask_spacing}")
    print(f"  Dose spacing:  {dose_spacing}")
    if not spacing_match:
        print("  !!! VAROITUS: Spacing ei täsmää !!!")
    print("-" * 60)
