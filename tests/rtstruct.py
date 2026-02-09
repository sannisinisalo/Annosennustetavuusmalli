# -*- coding: utf-8 -*-
"""
Tekijä: Sanni Sinisalo

Automatisoitu RS-skaalaus kaikille potilaille, pääkansiot löydetään automaattisesti
"""

import os
import glob
import numpy as np
import cv2
import pydicom
from rt_utils import RTStructBuilder
import re

# -----------------------------
# APUTOIMINNOT
# -----------------------------

def sort_patients_numerically(patient_list):
    """
    Järjestää potilaat numeron mukaan PatientX_... muodossa
    """
    def extract_number(name):
        m = re.search(r'Patient(\d+)_', name)
        return int(m.group(1)) if m else 0
    return sorted(patient_list, key=extract_number)

def get_ct_shape_and_slice_count(ct_path):
    files = sorted(glob.glob(os.path.join(ct_path, "*.dcm")))
    if not files:
        raise FileNotFoundError(f"Ei DICOMeja polussa: {ct_path}")
    first = pydicom.dcmread(files[0], stop_before_pixels=True)
    return (first.Rows, first.Columns), len(files)


def scale_mask_to_ct(mask_bool, new_rows_cols, slice_count_expected):
    """
    Skaalaa boolean-maskin (mask_bool) pikseliresoluution new_rows_cols:iin
    ja varmistaa, että slice-määrä vastaa slice_count_expected.
    Palauttaa (rows, cols, slices)-muotoisen bool-arrayn.
    """

    if mask_bool.ndim != 3:
        raise ValueError(f"ROI-maskin tulee olla 3D. ndim={mask_bool.ndim}")

    # 1️⃣ Korjaa slice-määrä
    slices_in = mask_bool.shape[2] if mask_bool.shape[2] == slice_count_expected else None

    if mask_bool.shape[2] > slice_count_expected:
        # Rajaa ylimääräiset slice:t loppuun
        mask_bool = mask_bool[:, :, :slice_count_expected]
    elif mask_bool.shape[2] < slice_count_expected:
        # Lisää tyhjiä slicejä loppuun
        diff = slice_count_expected - mask_bool.shape[2]
        extra = np.zeros((mask_bool.shape[0], mask_bool.shape[1], diff), dtype=bool)
        mask_bool = np.concatenate([mask_bool, extra], axis=2)

    # 2️⃣ Skaalaa jokainen slice
    new_rows, new_cols = new_rows_cols
    scaled_uint8 = np.zeros((new_rows, new_cols, slice_count_expected), dtype=np.uint8)

    for i in range(slice_count_expected):
        scaled_uint8[:, :, i] = cv2.resize(
            mask_bool[:, :, i].astype(np.uint8),
            (new_cols, new_rows),
            interpolation=cv2.INTER_NEAREST
        )

    scaled_bool = scaled_uint8.astype(bool)
    return scaled_bool



def resample_rtstruct(original_rt_path, original_ct_path, new_ct_path, output_rt_path):
    print(f"\n[INFO] Käsitellään RS: {original_rt_path}")
    print(f"       Alkuperäinen CT: {original_ct_path}")
    print(f"       Skaalattu CT: {new_ct_path}")

    # Yritä ladata RS
    try:
        rtstruct = RTStructBuilder.create_from(
            dicom_series_path=original_ct_path,
            rt_struct_path=original_rt_path
        )
    except Exception as e:
        print(f"[WARN] RS ei vastaa CT:tä, ohitetaan: {original_rt_path}. Virhe: {e}")
        return

    roi_names = rtstruct.get_roi_names()
    if not roi_names:
        print(f"[WARN] Potilaalla ei ROI:ita, ohitetaan: {original_rt_path}")
        return

    new_rtstruct = RTStructBuilder.create_new(dicom_series_path=new_ct_path)
    (new_rows, new_cols), new_slice_count = get_ct_shape_and_slice_count(new_ct_path)

    for roi_name in roi_names:
        try:
            mask = rtstruct.get_roi_mask_by_name(roi_name)

        except Exception:
            # ROI:lla ei ole ContourSequencea → ei voi muuttaa maskiksi
            print(f"[WARN] ROI '{roi_name}' ei sisällä ContourSequencea, ohitetaan.")
            continue

    # Skaalaa maski
    scaled_mask = scale_mask_to_ct(mask, (new_rows, new_cols), new_slice_count)

    # Lisää uuteen RS:ään
    new_rtstruct.add_roi(mask=scaled_mask, name=roi_name)


    new_rtstruct.save(output_rt_path)
    print(f"[OK] Tallennettu uusi RS: {output_rt_path}")



# -----------------------------
# MONIPOTILAS-PROSESSI AUTOMAATTISESTI
# -----------------------------

def process_all_patients(base_dir):
    """
    base_dir: Aineisto-kansion polku
    Löytää kaikki pääkansiot, etsii niiden ds-version ja potilaat automaattisesti.
    """

    # Hae kaikki pääkansiot
    all_entries = os.listdir(base_dir)
    base_folders = [f for f in all_entries if os.path.isdir(os.path.join(base_dir, f)) and not f.endswith('ds')]

    print(f"[INFO] Löydetty pääkansiot: {base_folders}")

    for folder in base_folders:
        orig_base = os.path.join(base_dir, folder)
        ds_base_name = folder + "ds"
        ds_base = os.path.join(base_dir, ds_base_name)

        if not os.path.exists(ds_base):
            print(f"[WARN] ds-versiota ei löydy: {ds_base}")
            continue

        # Käy läpi potilaat alkuperäisessä kansiossa
        patients = sort_patients_numerically(os.listdir(orig_base))
        for patient_name in patients:
            patient_orig_ct = os.path.join(orig_base, patient_name)
            patient_scaled_ct = os.path.join(ds_base, patient_name, "ct")
            patient_rs_folder = os.path.join(ds_base, patient_name, "struct")

            if not os.path.exists(patient_orig_ct):
                print(f"[WARN] Ei alkuperäisiä CT:itä: {patient_orig_ct}")
                continue
            if not os.path.exists(patient_scaled_ct):
                print(f"[WARN] Ei skaalattuja CT:itä: {patient_scaled_ct}")
                continue
            if not os.path.exists(patient_rs_folder):
                print(f"[WARN] Ei RS-kansiota: {patient_rs_folder}")
                continue

            # Etsi RS-tiedosto
            rs_files = glob.glob(os.path.join(patient_rs_folder, "RS*.dcm"))
            if not rs_files:
                print(f"[WARN] Ei RS-tiedostoa: {patient_rs_folder}")
                continue

            original_rt = rs_files[0]
            output_rt = os.path.join(patient_rs_folder, "RS_SKAALATTU.dcm")
            # Jos tiedosto on jo olemassa, ohita
            if os.path.exists(output_rt):
                
                print(f"[INFO] RS_SKAALATTU.dcm löytyy jo, ohitetaan: {output_rt}")
                continue

            try:
                resample_rtstruct(
                    original_rt_path=original_rt,
                    original_ct_path=patient_orig_ct,
                    new_ct_path=patient_scaled_ct,
                    output_rt_path=output_rt
                )
            except Exception as e:
                print(f"[ERROR] Potilaan käsittely epäonnistui ({patient_name}): {e}")


# -----------------------------
# PÄÄOHJELMA
# -----------------------------

if __name__ == "__main__":
    BASE_DIR = r"C:\Users\User01\GRADU\Aineisto"
    process_all_patients(BASE_DIR)
