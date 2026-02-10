# -*- coding: utf-8 -*-
"""
Luotu Ma 26.1.2026
Tekijä: Akseli Leino
Muokkaaja. Sanni Sinisalo

Ensimmäinen testi Akselin annosennustettavuusmallin ajamiseen. 
Koodiin tehdyt muokkaukset:
    - Muutettu lukemaan oikeita tiedostopolkuja
    - 
"""

import torchio as tio
import glob
import re
import os
import numpy as np
import random
from pydicom import dcmread
from .custom_transforms import DoseScalingTransform, PixelSizingTransform, FlipRightTransform, CreateInputMask, CreateDistanceToPTV, ProbabilityMapTransform
import math
import pickle
from scipy.ndimage import zoom
import torch


def generate_datasets(path: str, reduce_samples: float, split: tuple = (0.7, 0.1)):
    print(">>> ENTERED generate_datasets <<<")

    # Hae kaikki potilaskansiot
    all_items = glob.glob(path, recursive=True)
    study_folders = [s for s in all_items if os.path.isdir(s)]
    print(f"Found {len(study_folders)} study folders")

    # Split data
    random.seed(68)
    random.shuffle(study_folders)
    dataset_size = len(study_folders)
    n_train = int(dataset_size * split[0])
    n_val = int(dataset_size * split[1])
    train_folders = study_folders[:n_train]
    val_folders = study_folders[n_train:n_train + n_val]
    test_folders = study_folders[n_train + n_val:]

    train_subjects_list = []
    val_subjects_list = []
    test_subjects_list = []

    for j, folders in enumerate([train_folders, val_folders, test_folders]):
        for folder in folders:
            subject_name = os.path.basename(os.path.normpath(folder))

            # Polut CT, mask ja dose kansioihin
            ct_dir = os.path.join(folder, 'ct')
            mask_dir = os.path.join(folder, 'maskids')
            dose_dir = os.path.join(folder, 'doseds')

            ct_files = sorted(glob.glob(os.path.join(ct_dir, '*')))
            mask_files = sorted(glob.glob(os.path.join(mask_dir, '*')))
            dose_files = sorted(glob.glob(os.path.join(dose_dir, '*')))
    

            # Tarkista, että tiedostoja löytyy
            if len(ct_files) == 0 or len(mask_files) == 0 or len(dose_files) == 0:
                print("Missing data for subject:", subject_name)
                continue

            # --- LUE METADATA ---
            # Dose: yksi monikehyksinen DICOM
            dose_file = dose_files[0]
            ds_dose = dcmread(dose_file)
            num_samples = int(ds_dose.NumberOfFrames)
            dose_multiplier = float(ds_dose.DoseGridScaling)

            # CT: metadata ensimmäisestä tiedostosta
            ct_file = ct_files[0]
            ds_ct = dcmread(ct_file)
            pixel_spacing = float(ds_ct.PixelSpacing[0])

            # --- LUE TORCHIO-KUVAT ---
            # CT ja maski: kansioina → TorchIO kasaa sarjan (monta slicea)
            ct_data = tio.ScalarImage(ct_dir)
            mask_data = tio.ScalarImage(mask_dir)

            # Dose: yksittäinen monikehyksinen DICOM
            dose_data = tio.ScalarImage(dose_file)

            # Luo Subject
            new_subject = tio.Subject(
                ct=ct_data,
                mask=mask_data,
                original_mask=mask_data,
                dose=dose_data,
                name=subject_name,
                dose_multiplier=dose_multiplier,
                num_samples=int(num_samples / reduce_samples),
                pixel_spacing=pixel_spacing,
                distance_to_PTV=mask_data,      # placeholder
                probability_map=mask_data        # placeholder
            )
            

            if j == 0:
                train_subjects_list.append(new_subject)
            elif j == 1:
                val_subjects_list.append(new_subject)
            else:
                test_subjects_list.append(new_subject)

    # --- Transforms ---
    rescale_ct = tio.RescaleIntensity(out_min_max=(0, 4), in_min_max=(-1024, 3072), include=['ct'])
    resample_to_ct = tio.Resample(target='ct')

    rand_affine = tio.transforms.RandomAffine(
        degrees=(0, 0, 10),
        translation=(30, 70, 0),
        image_interpolation='nearest',
        default_pad_value='otsu'
    )
    rescale_dose = DoseScalingTransform()
    rescale_pixels = PixelSizingTransform()
    create_final_mask = CreateInputMask()
    create_distance_to_PTV = CreateDistanceToPTV()
    create_probability_map = ProbabilityMapTransform()

    
    train_transforms = tio.Compose((
        rescale_dose,
        rescale_ct,
        resample_to_ct, 
        rescale_pixels,
        create_final_mask,
        create_probability_map,
        create_distance_to_PTV,
        rand_affine
    ))

    transforms = tio.Compose((
        rescale_dose,
        rescale_ct,
        resample_to_ct, 
        rescale_pixels,
        create_final_mask,
        create_probability_map,
        create_distance_to_PTV
    ))

    train_set = tio.SubjectsDataset(train_subjects_list, transform=train_transforms)
    val_set = tio.SubjectsDataset(val_subjects_list, transform=transforms)
    test_set = tio.SubjectsDataset(test_subjects_list, transform=transforms)

    print("train_set:", train_set)
    print("val_set:", val_set)
    print("test_set:", test_set)
    print("Loaded", len(train_set), "training subjects")

    return train_set, val_set, test_set


