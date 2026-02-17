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
import os
import re
import warnings
from pathlib import Path
from typing import List

import numpy as np
from pydicom import dcmread
from pydicom.multival import MultiValue
from scipy.ndimage import zoom  # type: ignore

from ..config import BASEDIR


def _ensure_dir(path: str | Path) -> None:
def _ensure_dir(path: str | Path) -> None:
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def _find_patient_folders(source_root: str | Path) -> List[str]:
def _find_patient_folders(source_root: str | Path) -> List[str]:
    """Find all patient folders starting with 'Patient' and sort numerically."""
    candidates = [
        p for p in glob.glob(os.path.join(source_root, "*")) if os.path.isdir(p)
    ]
    patients = [
        p for p in candidates if os.path.basename(p).lower().startswith("patient")
    ]

    def patient_sort_key(path):
        name = os.path.basename(path)
        m = re.search(r"(\d+)", name)
        return int(m.group(1)) if m else float("inf")
        m = re.search(r"(\d+)", name)
        return int(m.group(1)) if m else float("inf")

    return sorted(patients, key=patient_sort_key)


def _find_dicom_by_modality(folder: str, modality: str) -> List[str]:
    """Find all DICOM files of a given modality in a folder."""
    files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f))
    ]
    dicoms = []
    for f in files:
        try:
            ds = dcmread(f, stop_before_pixels=True)
            if getattr(ds, "Modality", None) == modality:
                dicoms.append(f)
        except Exception:
            continue
    return dicoms


if __name__ == "__main__":
    print("PROCESSING...")

    # Choose dataset: 'L', 'R', 'LAX', 'RAX'
    dataset = "L"

    # Map dataset to source and destination paths
    if dataset == "L":
        SOURCE_PATH = BASEDIR / "VN0"
        DESTINATION_PATH = BASEDIR / "VN0ds"
    elif dataset == "R":
        SOURCE_PATH = BASEDIR / "ON0"
        DESTINATION_PATH = BASEDIR / "ON0ds"
    elif dataset == "LAX":
        SOURCE_PATH = BASEDIR / "VN+"
        DESTINATION_PATH = BASEDIR / "VN+ds"
    elif dataset == "RAX":
        SOURCE_PATH = BASEDIR / "ON+"
        DESTINATION_PATH = BASEDIR / "ON+ds"
    if dataset == "L":
        SOURCE_PATH = BASEDIR / "VN0"
        DESTINATION_PATH = BASEDIR / "VN0ds"
    elif dataset == "R":
        SOURCE_PATH = BASEDIR / "ON0"
        DESTINATION_PATH = BASEDIR / "ON0ds"
    elif dataset == "LAX":
        SOURCE_PATH = BASEDIR / "VN+"
        DESTINATION_PATH = BASEDIR / "VN+ds"
    elif dataset == "RAX":
        SOURCE_PATH = BASEDIR / "ON+"
        DESTINATION_PATH = BASEDIR / "ON+ds"
    else:
        raise ValueError("Tuntematon dataset-parametri. Käytä: 'L', 'R', 'LAX', 'RAX'.")

    _ensure_dir(DESTINATION_PATH)
    patients = _find_patient_folders(SOURCE_PATH)
    if not patients:
        print("Ei löytynyt potilaskansioita (Patient*).")
    else:
        for p in patients:
            patient_name = os.path.basename(p)
            print(f"Processing {patient_name}...")

            # Find DICOM files by modality
            ct_files = _find_dicom_by_modality(p, "CT")
            rd_files = _find_dicom_by_modality(p, "RTDOSE")
            rp_files = _find_dicom_by_modality(p, "RTPLAN")
            rs_files = _find_dicom_by_modality(p, "RTSTRUCT")

            # Create output folders
            folders = {
                "ct": os.path.join(DESTINATION_PATH, patient_name, "ct"),
                "dose": os.path.join(
                    DESTINATION_PATH, patient_name, "dose"
                ),  # alkuperäinen dose
                "doseds": os.path.join(
                    DESTINATION_PATH, patient_name, "doseds"
                ),  # uusi downsampled/copy dose
                "plan": os.path.join(DESTINATION_PATH, patient_name, "plan"),
                "struct": os.path.join(DESTINATION_PATH, patient_name, "struct"),
                "mask": os.path.join(DESTINATION_PATH, patient_name, "maskids"),
            }
            for d in folders.values():
                _ensure_dir(d)

            # Downsample CT slices
            for f in ct_files:
                try:
                    ds = dcmread(f)
                    arr_down = zoom(ds.pixel_array, zoom=(0.5, 0.5), order=1)
                    ds.PixelData = arr_down.tobytes()
                    ds.Rows, ds.Columns = arr_down.shape
                    if hasattr(ds, "PixelSpacing"):
                        ds.PixelSpacing = MultiValue(
                            float, [float(x) * 2 for x in ds.PixelSpacing]
                        )
                    ds.save_as(os.path.join(folders["ct"], os.path.basename(f)))
                except Exception as e:
                    warnings.warn(
                        f"{patient_name}: CT-tiedoston {os.path.basename(f)} käsittely epäonnistui: {e}"
                    )

            # Downsample RTDOSE (first file only)
            # 1. Ladataan alkuperäinen dose
            dose_src_folder = os.path.join(DESTINATION_PATH, patient_name, "dose")
            dose_files = [
                os.path.join(dose_src_folder, f)
                for f in os.listdir(dose_src_folder)
                if os.path.isfile(os.path.join(dose_src_folder, f))
            ]

            arr_dose_down = None
            ds_dose = None
            if dose_files:
                try:
                    ds_dose = dcmread(dose_files[0])
                    arr_dose = ds_dose.pixel_array
                    # 2. Downsample Y/X (Z pysyy samana)
                    arr_dose_down = zoom(arr_dose, zoom=(1, 0.5, 0.5), order=1)
                    ds_dose.Rows, ds_dose.Columns = (
                        arr_dose_down.shape[1],
                        arr_dose_down.shape[2],
                    )
                    if hasattr(ds_dose, "PixelSpacing"):
                        ds_dose.PixelSpacing = MultiValue(
                            float, [float(x) * 2 for x in ds_dose.PixelSpacing]
                        )
                except Exception as e:
                    warnings.warn(
                        f"{patient_name}: RD-tiedoston käsittely epäonnistui: {e}"
                    )

            # 3. Ladataan maskit ja downsampleataan samaan kokoon
            mask_src_folder = os.path.join(DESTINATION_PATH, patient_name, "maski")
            mask_files = [
                os.path.join(mask_src_folder, f)
                for f in os.listdir(mask_src_folder)
                if os.path.isfile(os.path.join(mask_src_folder, f))
            ]

            for f in mask_files:
                try:
                    ds_mask = dcmread(f)
                    mask_orig = ds_mask.pixel_array

                    # Downsample mask samaan kokoon kuin dose
                    zoom_y = arr_dose_down.shape[1] / mask_orig.shape[0]
                    zoom_x = arr_dose_down.shape[2] / mask_orig.shape[1]
                    mask_down = zoom(mask_orig, zoom=(zoom_y, zoom_x), order=0)

                    ds_mask.PixelData = mask_down.tobytes()
                    ds_mask.Rows, ds_mask.Columns = mask_down.shape
                    if hasattr(ds_mask, "PixelSpacing"):
                        ds_mask.PixelSpacing = MultiValue(
                            float, [float(x) * 2 for x in ds_mask.PixelSpacing]
                        )

                    # 4. Sovelletaan maski doseen slice-reversoinnilla
                    if hasattr(ds_mask, "InstanceNumber") and arr_dose_down is not None:
                        idx = arr_dose_down.shape[0] - ds_mask.InstanceNumber
                        mask_min = np.min(mask_down)
                        arr_dose_down[idx, :, :] *= np.isin(
                            mask_down, mask_min, invert=True
                        )

                    # 5. Tallennetaan maski maskids-kansioon
                    ds_mask.save_as(os.path.join(folders["mask"], os.path.basename(f)))
                except Exception as e:
                    warnings.warn(
                        f"{patient_name}: Mask-tiedoston {os.path.basename(f)} käsittely epäonnistui: {e}"
                    )

            # 5. Tallennetaan muokattu dose doseds-kansioon
            if ds_dose is not None and arr_dose_down is not None:
                try:
                    ds_dose.PixelData = arr_dose_down.tobytes()
                    ds_dose.save_as(
                        os.path.join(folders["doseds"], os.path.basename(dose_files[0]))
                    )
                except Exception as e:
                    warnings.warn(
                        f"{patient_name}: RD-tiedoston tallennus doseds-kansioon epäonnistui: {e}"
                    )

            # Copy RP and RS unchanged
            for f in rp_files:
                try:
                    ds = dcmread(f, stop_before_pixels=True)
                    ds.save_as(os.path.join(folders["plan"], os.path.basename(f)))
                except Exception as e:
                    warnings.warn(
                        f"{patient_name}: RP-tiedoston kopiointi epäonnistui: {e}"
                    )
            for f in rs_files:
                try:
                    ds = dcmread(f, stop_before_pixels=True)
                    ds.save_as(os.path.join(folders["struct"], os.path.basename(f)))
                except Exception as e:
                    warnings.warn(
                        f"{patient_name}: RS-tiedoston kopiointi epäonnistui: {e}"
                    )

    print("DONE")
