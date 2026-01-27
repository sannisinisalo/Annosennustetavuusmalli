# -*- coding: utf-8 -*-
"""
Luotu Ti 27.1.2026

Tiedostojen koon pienentämiseen käytetty koodi
"""

import os
from os.path import isfile, join
import glob
from pydicom import dcmread
from pydicom.multival import MultiValue
import re
import numpy as np
from utils import remove_overlap
import warnings
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from pathlib import Path

"""
Discard reasons:
Left
ANON1832: No dose, rtstruct, ct, or mask
ANON1845, ANON1861, ANON0090, ANON0091, ANON1792, ANON0132: Missing critical structures (PTV, heart, lungs or contra breast)
"""

BASEDIR = Path("C:\\Users\\User01\\GRADU\\Aineisto\\")

def downsample_dicom_folder(dataset):
    if dataset == 'L':
        SOURCE_PATH = BASEDIR / 'VN0'
        DESTINATION_PATH = BASEDIR / 'VN0ds'
    elif dataset == 'R':
        SOURCE_PATH = BASEDIR / 'ON0'
        DESTINATION_PATH = BASEDIR / 'ON0ds'
    elif dataset == 'LAX':
        SOURCE_PATH = BASEDIR / 'VN+'
        DESTINATION_PATH = BASEDIR / 'VN+ds'
    elif dataset == 'RAX':
        SOURCE_PATH = BASEDIR / 'ON+'
        DESTINATION_PATH = BASEDIR / 'ON+ds'
    else:
        raise ValueError("Tuntematon dataset-parametri. Käytä: 'L', 'R', 'LAX', 'RAX'.")
    

    patient_folders = [p for p in SOURCE_PATH.iterdir()
                       if p.is_dir() and p.name.startswith("Patient")]

    ct_files = []
    rd_file = None
    rp_file = None
    rs_file = None
    
    for f in patient_folders.iterdir():
        if f.is_dir():
            continue  # esim. "vanha ct"
        if f.name.startswith("RD"):
            rd_file = f
        elif f.name.startswith("RP"):
            rp_file = f
        elif f.name.startswith("RS"):
            rs_file = f
        else:
            ct_files.append(f)
            
    dest_patient = DESTINATION_PATH / patient_folders.name

    (dest_patient / "ct").mkdir(parents=True, exist_ok=True)
    (dest_patient / "dose").mkdir(exist_ok=True)
    (dest_patient / "struct").mkdir(exist_ok=True)
    (dest_patient / "plan").mkdir(exist_ok=True)
    (dest_patient / "maskids").mkdir(exist_ok=True)
    
    for i, subject in enumerate(unique_subjects):
            print(f"{i}: {subject}")
            subject_folders = [s for s in study_folders if r.search(s)]
            # There should be CT, mask "CT", dose, and RTstruct for each subject.
            if len(subject_folders) == 0:
                pass
            else:
                if len(subject_folders) != 6:
                    warnings.warn("Subject " + subject + " has != 5 folders. Ensure manually that ct, dose, and mask are present. Patient not processed.")
                    continue

                r_ct = re.compile('(?!.*Mask.*Mask).+CT.+CT.+CT')
                #r_ct = re.compile('(?!.*Mask.*Mask).+CT.+CT.+DVH.+CT')
                r_mask = re.compile('.+Mask.+Mask')
                r_dose = re.compile('.+Dose')

                ct_path = [s for s in subject_folders if r_ct.search(s)]
                mask_path = [s for s in subject_folders if r_mask.search(s)]
                dose_path = [s for s in subject_folders if r_dose.search(s)]
                
                dcm_path_ct = os.listdir(ct_path[0])
                dcmfile_path_dose = dose_path[0] + os.listdir(dose_path[0])[0]
                dcm_path_mask = os.listdir(mask_path[0])
                
                ### Ensin tehdään potilaskansio
                try:
                    os.mkdir(DESTINATION_PATH + subject)
                except FileExistsError:
                    pass

                
                ### Tallennetaan dose ekana. Tämä hieman erilainen kuin muut, koska kaikki leikkeet samassa filessä.
                dcm_dose = dcmread(rd_file)
                dose_down = zoom(dcm_dose.pixel_array, zoom=(1, 0.5, 0.5), order=1)
                
                _, dcm_dose.Rows, dcm_dose.Columns = dose_down.shape
                dcm_dose.PixelSpacing = MultiValue(
                    float, [float(x)*2 for x in dcm_dose.PixelSpacing]
                )
                
                dcm_dose.PixelData = dose_down.tobytes()
                dcm_dose.save_as(dest_patient / "dose" / rd_file.name)

                try:
                    os.mkdir(DESTINATION_PATH + subject + "/dose/")
                except FileExistsError:
                    pass
                
                
                ### Sitten CT
                try:
                    os.mkdir(DESTINATION_PATH + subject + "/ct/")
                except FileExistsError:
                    pass
                
                # Tässä joudutaan iteroimaan läpi koko kansio ja avataan ja tallennetaan yksittäiset leikkeet yksitellen.
                for ct_file in ct_files:
                    dcm = dcmread(ct_file)
                    down = zoom(dcm.pixel_array, zoom=(0.5, 0.5), order=1)
                
                    dcm.PixelData = down.tobytes()
                    dcm.Rows, dcm.Columns = down.shape
                    dcm.PixelSpacing = MultiValue(float, [float(x)*2 for x in dcm.PixelSpacing])
                
                    dcm.save_as(dest_patient / "ct" / ct_file.name)

                
                ### Maskit täysin samalla tavalla kuin CT
                mask_src = dest_patient / "maski"
                mask_dst = dest_patient / "maskids"
                
                mask_files = list(mask_src.glob("*.dcm"))
                
                for mf in mask_files:
                        dcm_mask = dcmread(mf)
                        down = zoom(dcm_mask.pixel_array, zoom=(0.5, 0.5), order=0)
                    
                        dcm_mask.PixelData = down.tobytes()
                        dcm_mask.Rows, dcm_mask.Columns = down.shape
                        dcm_mask.PixelSpacing = MultiValue(
                            float, [float(x)*2 for x in dcm_mask.PixelSpacing]
                        )
                    
                        dcm_mask.save_as(mask_dst / mf.name)

    
                # Haetaan tallennetaan pienennetty data takaisin DICOMiin
                dcm_dose.PixelData = dcm_dose_downsampled.tobytes()
                dcm_dose.save_as(DESTINATION_PATH + subject + "/dose/" + os.listdir(dose_path[0])[0])


print("PROCESSING...")
downsample_dicom_folder('LAX')
print("DONE")

# n_L = 279
