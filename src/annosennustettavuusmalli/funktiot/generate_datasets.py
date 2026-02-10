# -*- coding: utf-8 -*-
"""
Luotu Ma 26.1.2026

Perustuu Akseli Leinon alkuperäiseen koodiin.

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

    all_items = glob.glob(path, recursive=True)
    study_folders = [s for s in all_items if os.path.isdir(s)]

    print(f"Found {len(study_folders)} study folders")

    train_subjects_list = []
    val_subjects_list = []
    test_subjects_list = []

    random.seed(68)
    random.shuffle(study_folders)

    dataset_size = len(study_folders)
    n_train = int(dataset_size * split[0])
    n_val = int(dataset_size * split[1])

    train_folders = study_folders[:n_train]
    val_folders = study_folders[n_train:n_train + n_val]
    test_folders = study_folders[n_train + n_val:]

    for j, folders in enumerate([train_folders, val_folders, test_folders]):
        for folder in folders:
            subject_name = os.path.basename(os.path.normpath(folder))
            print("Processing subject folder:", subject_name)

            ct_path = os.path.join(folder, 'ct')
            mask_path = os.path.join(folder, 'maski')
            dose_path = os.path.join(folder, 'doseds')

            if not (os.path.exists(ct_path) and os.path.exists(mask_path) and os.path.exists(dose_path)):
                print("Missing data for subject:", subject_name)
                print("ct:", ct_path)
                print("mask:", mask_path)
                print("dose:", dose_path)
                continue

            # --- Dose ---
            dose_files = os.listdir(dose_path)
            assert len(dose_files) == 1, f"Expected 1 dose file, found {len(dose_files)} in {dose_path}"

            dose_file = os.path.join(dose_path, dose_files[0])
            ds_dose = dcmread(dose_file)
            num_samples = int(ds_dose.NumberOfFrames)
            dose_multiplier = float(ds_dose.DoseGridScaling)

            # --- CT ---
            ct_files = os.listdir(ct_path)
            assert len(ct_files) > 0, f"No CT files found in {ct_path}"

            ct_file = os.path.join(ct_path, ct_files[0])
            ds_ct = dcmread(ct_file)
            pixel_spacing = float(list(ds_ct.PixelSpacing)[0])

            # --- TorchIO data ---
            ct_data = tio.ScalarImage(ct_path)
            mask_data = tio.ScalarImage(mask_path)
            dose_data = tio.ScalarImage(dose_path)

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

    # --- Transforms (UNCHANGED) ---
    rescale_ct = tio.RescaleIntensity(out_min_max=(0, 4), in_min_max=(-1024, 3072), include=['ct'])
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
        rescale_pixels,
        create_final_mask,
        create_probability_map,
        create_distance_to_PTV,
        rand_affine
    ))

    transforms = tio.Compose((
        rescale_dose,
        rescale_ct,
        rescale_pixels,
        create_final_mask,
        create_probability_map,
        create_distance_to_PTV
    ))

    train_set = tio.SubjectsDataset(train_subjects_list, transform=train_transforms)
    val_set = tio.SubjectsDataset(val_subjects_list, transform=transforms)
    test_set = tio.SubjectsDataset(test_subjects_list, transform=transforms)

    return train_set, val_set, test_set