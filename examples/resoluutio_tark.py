# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 10:14:58 2026

@author: User01
"""

import os
import re
import pydicom

# Polku pääkansioon
base_path = r"C:\Users\User01\GRADU\Aineisto\VN0ds"

def extract_patient_number(folder_name):
    """Palauttaa potilaan numeron PatientX_VN0 -muodossa"""
    match = re.search(r'Patient(\d+)_VN0', folder_name)
    return int(match.group(1)) if match else float('inf')  # jos ei löydy, laitetaan loppuun

# Hae kaikki potilaskansiot
all_patients = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]

# Järjestetään numeron mukaan
all_patients.sort(key=extract_patient_number)

# Käydään läpi jokainen potilas
for patient_folder in all_patients:
    patient_path = os.path.join(base_path, patient_folder)
    
    print(f"\nPotilas: {patient_folder}")
    
    for subfolder in ['ct', 'maskids', 'doseds']:
        subfolder_path = os.path.join(patient_path, subfolder)
        
        if os.path.exists(subfolder_path) and os.listdir(subfolder_path):
            first_file = os.path.join(subfolder_path, os.listdir(subfolder_path)[0])
            
            try:
                ds = pydicom.dcmread(first_file)
                
                resolution = (int(ds.Rows), int(ds.Columns)) if 'Rows' in ds and 'Columns' in ds else "N/A"
                pixel_spacing = ds.PixelSpacing if 'PixelSpacing' in ds else "N/A"
                slice_thickness = ds.SliceThickness if 'SliceThickness' in ds else "N/A"
                
                print(f"  Kansiosta '{subfolder}':   Resolution: {resolution}, Pixel Spacing: {pixel_spacing}, Slice Thickness: {slice_thickness}")
            
            except Exception as e:
                print(f"    Virhe luettaessa tiedostoa {first_file}: {e}")
        else:
            print(f"  Kansiota '{subfolder}' ei löytynyt tai se on tyhjä.")