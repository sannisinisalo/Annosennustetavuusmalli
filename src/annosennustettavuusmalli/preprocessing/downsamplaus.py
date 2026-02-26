# -*- coding: utf-8 -*-
"""
Luotu Ti 27.1.2026
Tekijä: Akseli Leino
Muokkaaja: Sanni Sinisalo

Tiedostojen koon pienentämiseen käytetty koodi
"""

import glob
import os
import re
import warnings
from pathlib import Path
from typing import List

import numpy as np
from pydicom import dcmread
from pydicom.multival import MultiValue
from scipy.ndimage import zoom
from luokat import AllPatients


def ensure_dir(path: Path) -> None:
    """
    Create directory if it does not exist
    """
    path.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    print("PROCESSING...")

    # Choose dataset: "L", "R", "LAX", "RAX"
    dataset = "L"

    # Map dataset to source and destination names
    dataset_map = {
        "L": ("VN0", "VN0ds"),
        "R": ("ON0", "ON0ds"),
        "LAX": ("VN+", "VN+ds"),
        "RAX": ("ON+", "ON+ds"),
    }
    if dataset not in dataset_map:
        raise ValueError("Tuntematon dataset-parametri. Käytä: 'L', 'R', 'LAX', 'RAX'.")


    source_dataset, processed_dataset = dataset_map[dataset]

    all_patients = AllPatients(
        processed_dataset=processed_dataset, original_dataset=source_dataset
        )

for patient in all_patients.sorted_by_number():
        patient_name = patient.patient_folder
        print(f"Processing {patient_name}...")

        # --- Luo output-kansiot ---
        folders = {
            "ct": patient.ct_dir,
            "dose": patient.dose_dir,
            "doseds": patient.doseds_dir,
            "plan": patient.plan_dir,
            "struct": patient.struct_dir,
            "mask": patient.maskds_dir,
        }
        for d in folders.values():
            ensure_dir(d)

        # --- CT downsamplaus ---
        for f in patient.ct_files:
            try:
                ds = dcmread(f)
                arr_down = zoom(ds.pixel_array, zoom=(0.5, 0.5), order=1)
                ds.PixelData = arr_down.tobytes()
                ds.Rows, ds.Columns = arr_down.shape
                if hasattr(ds, "PixelSpacing"):
                    ds.PixelSpacing = MultiValue(
                        float, [float(x)*2 for x in ds.PixelSpacing]
                    )
                ds.save_as(folders["ct"] / f.name)
            except Exception as e:
                warnings.warn(f"{patient_name}: CT-tiedoston {f.name} käsittely epäonnistui: {e}")

        # --- Dose downsamplaus ---
        try:
            dose_file = patient.rtdose_file
            ds_dose = dcmread(dose_file)
            arr_dose = ds_dose.pixel_array
            arr_dose_down = zoom(arr_dose, zoom=(1, 0.5, 0.5), order=1)
            ds_dose.Rows, ds_dose.Columns = arr_dose_down.shape[1], arr_dose_down.shape[2]
            if hasattr(ds_dose, "PixelSpacing"):
                ds_dose.PixelSpacing = MultiValue(
                    float, [float(x)*2 for x in ds_dose.PixelSpacing]
                )
            ds_dose.PixelData = arr_dose_down.tobytes()
            ds_dose.save_as(folders["doseds"] / dose_file.name)
        except FileNotFoundError:
            warnings.warn(f"{patient_name}: Dose-tiedostoa ei löytynyt → ohitetaan")
            arr_dose_down = None
            ds_dose = None
        except Exception as e:
            warnings.warn(f"{patient_name}: Dose-tiedoston käsittely epäonnistui: {e}")
            arr_dose_down = None
            ds_dose = None

        # --- Maskien downsamplaus ---
        mask_dir = patient.mask_dir
        if mask_dir.exists():
            for f in mask_dir.glob("*.dcm"):
                try:
                    ds_mask = dcmread(f)
                    mask_orig = ds_mask.pixel_array
                    if arr_dose_down is not None:
                        zoom_y = arr_dose_down.shape[1] / mask_orig.shape[0]
                        zoom_x = arr_dose_down.shape[2] / mask_orig.shape[1]
                        mask_down = zoom(mask_orig, zoom=(zoom_y, zoom_x), order=0)

                        ds_mask.PixelData = mask_down.tobytes()
                        ds_mask.Rows, ds_mask.Columns = mask_down.shape
                        if hasattr(ds_mask, "PixelSpacing"):
                            ds_mask.PixelSpacing = MultiValue(
                                float, [float(x)*2 for x in ds_mask.PixelSpacing]
                            )

                        # Sovelletaan maski doseen slice-reversoinnilla
                        if hasattr(ds_mask, "InstanceNumber"):
                            idx = arr_dose_down.shape[0] - ds_mask.InstanceNumber
                            mask_min = np.min(mask_down)
                            arr_dose_down[idx, :, :] *= np.isin(mask_down, mask_min, invert=True)

                    ds_mask.save_as(folders["mask"] / f.name)
                except Exception as e:
                    warnings.warn(f"{patient_name}: Mask-tiedoston {f.name} käsittely epäonnistui: {e}")

        # --- Kopioi RP ja RS ---
        try:
            ds = dcmread(patient.rtplan_file, stop_before_pixels=True)
            ds.save_as(folders["plan"] / patient.rtplan_file.name)
            ds = dcmread(patient.rtstruct_file, stop_before_pixels=True)
            ds.save_as(folders["struct"] / patient.rtstruct_file.name)
        except FileNotFoundError:
            warnings.warn(f"{patient_name}: RP/RS-tiedostoja ei löytynyt → ohitetaan")
        except Exception as e:
            warnings.warn(f"{patient_name}: RP/RS-tiedoston kopiointi epäonnistui: {e}")

print("DONE")