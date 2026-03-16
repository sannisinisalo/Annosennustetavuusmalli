# -*- coding: utf-8 -*-
"""
Luotu to 26.02.2026
Tekijä: Sanni Sinisalo

Koodi, jolla voidaan lukea ja listat potilaiden tiedostojen resoluutio, pixel_spacing ja slice_thickness
"""

import os
import re
import pydicom


base_path = r"C:\Users\User01\GRADU\Aineisto\VN0ds"

def extract_patient_number(folder_name):
    """
    Palauttaa potilaan numeron PatientX_VN0 -muodossa
    """
    match = re.search(r'Patient(\d+)_VN0', folder_name)
    return int(match.group(1)) if match else float('inf') 

all_patients = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]

all_patients.sort(key=extract_patient_number)

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