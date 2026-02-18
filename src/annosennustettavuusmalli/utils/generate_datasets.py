# -*- coding: utf-8 -*-
"""
Tekijä: Akseli Leino
Muokkaaja. Sanni Sinisalo
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
import copy


def generate_datasets(path: str, reduce_samples:float, split:tuple = (0.7, 0.1))->tuple([tio.SubjectsDataset, tio.SubjectsDataset, tio.SubjectsDataset]):
    all_items = glob.glob(path, recursive=True)
    study_folders = glob.glob(path)  # nämä ovat jo potilaskansiot
    unique_subjects = [os.path.basename(os.path.normpath(s)) for s in study_folders]
    train_subjects_list = []
    val_subjects_list = []
    test_subjects_list = []

    random.seed(68)
    random.shuffle(unique_subjects)
    dataset_size = len(unique_subjects)

    # default split (0.7, 0.1) means 70% training, 10% validation, rest to testing
    train_names = unique_subjects[:int(dataset_size*split[0])]
    val_names = unique_subjects[int(dataset_size*split[0]):int(dataset_size*(split[0]+split[1]))]
    test_names = unique_subjects[int(dataset_size*(split[0]+split[1])):]
    
    for j, subject_set in enumerate([train_names, val_names, test_names]):
        for subject in subject_set:
            subject_path = [s for s in study_folders if subject in s][0]
            
            ct_path = os.path.join(subject_path, 'ct')
            mask_path = os.path.join(subject_path, 'maskids')
            dose_path = os.path.join(subject_path, 'doseds')
     
            # Dose is saved as large integers and needs to be rescaled back to get dose in Gy
            dose_file = os.listdir(dose_path)[0]
            ds_dose = dcmread(os.path.join(dose_path, dose_file))
            num_samples = int(ds_dose.NumberOfFrames)
            dose_multiplier = float(ds_dose.DoseGridScaling)

            ct_file = os.listdir(ct_path)[0]
            ds_ct = dcmread(os.path.join(ct_path, ct_file))
            pixel_spacing = float(list(ds_ct.PixelSpacing)[0])

            ct_data = tio.ScalarImage(ct_path)
            mask_data = tio.ScalarImage(mask_path)
            dose_data = tio.ScalarImage(dose_path)
            
            new_subject = tio.Subject(
                ct = ct_data,
                mask = mask_data,
                original_mask = mask_data,
                dose = dose_data,
                name = subject,
                dose_multiplier = dose_multiplier,
                num_samples = int(num_samples/reduce_samples),
                pixel_spacing = pixel_spacing,
                distance_to_PTV = tio.ScalarImage(tensor=torch.zeros_like(mask_data.data)), # Muutettu, jotta maski sai järkeviä arvoja
                probability_map = tio.ScalarImage(tensor=torch.zeros_like(mask_data.data))
                ) # Muutetttu, jotta maski sai järkeviä arvoja 
            
            if j == 0:
                train_subjects_list.append(new_subject)
            elif j == 1:
                val_subjects_list.append(new_subject)
            elif j == 2:
                test_subjects_list.append(new_subject)
    
    print("Mask dtype:", new_subject['mask'][tio.DATA].dtype) # Debuglisäys
    
    #rescale_mask = tio.RescaleIntensity(out_min_max=(-0.2, 1), in_min_max = (-1, 10), include = ['mask'])
    rescale_ct = tio.RescaleIntensity(out_min_max=(0, 4), in_min_max = (-1024, 3072), include = ['ct'])
    rand_affine = tio.transforms.RandomAffine(degrees = (0, 0, 10), translation = (30, 70, 0), image_interpolation = 'nearest', default_pad_value = 'otsu') # padding should equal to value outside of body, change if it's not minimum
    rescale_dose = DoseScalingTransform()
    rescale_pixels = PixelSizingTransform()
    create_final_mask = CreateInputMask()
    create_distance_to_PTV = CreateDistanceToPTV()
    create_probability_map = ProbabilityMapTransform()
    
    # The order of the transforms is important! Padding of the size transformations are made with the assumption that data is already scaled. Thus, rescale transforms must be befor rescale pixels.
    # Also, create_final_mask must be AFTER resizing pixels, as it creates 'original_mask', which is not at the moment handled by PixelSizingTransform.
    
    resample = tio.Resample('ct') # Lisätty, jotta dose:lla sama pixel_spacing kuin CT:llä
    train_transforms = tio.Compose((
        resample,
        rescale_dose, 
        rescale_ct, 
        rescale_pixels, 
        create_final_mask, 
        create_probability_map, 
        create_distance_to_PTV, 
        rand_affine
        ))
    transforms = tio.Compose((
        resample,
        rescale_dose, 
        rescale_ct, 
        rescale_pixels, 
        create_final_mask, 
        create_probability_map, 
        create_distance_to_PTV))
    
    train_set = tio.SubjectsDataset(train_subjects_list, transform = train_transforms)
    val_set = tio.SubjectsDataset(val_subjects_list, transform = transforms)    
    test_set = tio.SubjectsDataset(test_subjects_list, transform = transforms)
    
    return train_set, val_set, test_set