# -*- coding: utf-8 -*-
"""
Downsample DICOM data where source folders are Patient* and contain CT slices, RD (RTDOSE), RP (RTPLAN), RS (RTSTRUCT).
- CT and RD are downsampled by factor 2 in-plane (Rows/Columns halved; PixelSpacing doubled).
- RP and RS are copied unchanged.
"""

import os
import glob
import warnings
from typing import List

from pydicom import dcmread
from pydicom.dataset import FileDataset
from pydicom.multival import MultiValue
from scipy.ndimage import zoom
import re


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _find_patient_folders(source_root: str) -> List[str]:
    # Hae kaikki kansiot, jotka alkavat "Patient"
    candidates = [p for p in glob.glob(os.path.join(source_root, "*")) if os.path.isdir(p)]
    patients = [p for p in candidates if os.path.basename(p).lower().startswith("patient")]

    # Lajittele numeron mukaan, ei pelkän merkkijonon
    def patient_sort_key(path):
        name = os.path.basename(path)
        m = re.search(r'(\d+)', name)
        return int(m.group(1)) if m else float('inf')

    patients_sorted = sorted(patients, key=patient_sort_key)
    return patients_sorted


def _find_dicom_by_modality(folder: str, modality: str) -> List[str]:
    # Etsi .dcm tiedostoja (tai kaikkia tiedostoja) ja filtteröi DICOM Modaliteetillä
    files = [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    dicoms = []
    for f in files:
        try:
            ds = dcmread(f, stop_before_pixels=True)
            if getattr(ds, "Modality", None) == modality:
                dicoms.append(f)
        except Exception:
            # Skippaa ei-DICOM tiedostot sekä tiedostot joita ei voi lukea
            continue
    return dicoms


def _downsample_ct_slice(ds: FileDataset, scale_xy: float = 0.5) -> FileDataset:
    arr = ds.pixel_array  # 2D
    # Lineaarinen interpolaatio CT:lle
    down = zoom(arr, zoom=(scale_xy, scale_xy), order=1)

    ds.PixelData = down.tobytes()
    ds.Rows, ds.Columns = down.shape

    # Päivitä PixelSpacing (Decimal String, typically two values: row, column spacing)
    if hasattr(ds, "PixelSpacing"):
        new_spacing = [float(x) / scale_xy for x in list(ds.PixelSpacing)]  # doubling when scale=0.5
        ds.PixelSpacing = MultiValue(float, new_spacing)

    return ds


def _downsample_dose(ds: FileDataset, scale_xy: float = 0.5) -> FileDataset:
    arr = ds.pixel_array  # Oletetaan 3D: (z, y, x)
    if arr.ndim == 3:
        # Downsamplätään ainoastaan y,x-tasossa. Pidä z muuttumatomana
        down = zoom(arr, zoom=(1.0, scale_xy, scale_xy), order=1)
        ds.Rows, ds.Columns = down.shape[1], down.shape[2]
        ds.PixelData = down.tobytes()

        # Päivitä PixelSpacing: for dose, PixelSpacing is typically [row_spacing, col_spacing]
        if hasattr(ds, "PixelSpacing"):
            new_spacing = [float(x) / scale_xy for x in list(ds.PixelSpacing)]
            ds.PixelSpacing = MultiValue(float, new_spacing)
    else:
        warnings.warn("RTDOSE has unexpected pixel array shape; copying unchanged.")
    return ds


def downsample_dicom_folder(dataset: str) -> None:
    # Mapataan aineiston lippu lähteen/määränpään juureen
    if dataset == 'L':
        SOURCE_PATH = r"C:\Users\User01\GRADU\Aineisto\VN0"
        DESTINATION_PATH = r"C:\Users\User01\GRADU\Aineisto\VN0ds"
    elif dataset == 'R':
        SOURCE_PATH = r"C:\Users\User01\GRADU\Aineisto\ON0"
        DESTINATION_PATH = r"C:\Users\User01\GRADU\Aineisto\ON0ds"
    elif dataset == 'LAX':
        SOURCE_PATH = r"C:\Users\User01\GRADU\Aineisto\VN+"
        DESTINATION_PATH = r"C:\Users\User01\GRADU\Aineisto\VN+ds"
    elif dataset == 'RAX':
        SOURCE_PATH = r"C:\Users\User01\GRADU\Aineisto\ON+"
        DESTINATION_PATH = r"C:\Users\User01\GRADU\Aineisto\ON+ds"
    else:
        raise ValueError("Tuntematon dataset-parametri. Käytä: 'L', 'R', 'LAX', 'RAX'.")

    _ensure_dir(DESTINATION_PATH)

    patients = _find_patient_folders(SOURCE_PATH)
    if not patients:
        print("Ei löytynyt potilaskansioita (Patient*).")
        return

    for p in patients:
        patient_name = os.path.basename(p)
        print(f"Processing {patient_name}...")

        # Etsi modaliteettitiedostot
        ct_files = _find_dicom_by_modality(p, "CT")
        rd_files = _find_dicom_by_modality(p, "RTDOSE")
        rp_files = _find_dicom_by_modality(p, "RTPLAN")
        rs_files = _find_dicom_by_modality(p, "RTSTRUCT")

        if not ct_files:
            warnings.warn(f"{patient_name}: CT-kuvia ei löytynyt.")
        if not rd_files:
            warnings.warn(f"{patient_name}: RTDOSE (RD) ei löytynyt.")
        if not rp_files:
            warnings.warn(f"{patient_name}: RTPLAN (RP) ei löytynyt.")
        if not rs_files:
            warnings.warn(f"{patient_name}: RTSTRUCT (RS) ei löytynyt.")

        # Valmistele alakansiot
        ct_out = os.path.join(DESTINATION_PATH, patient_name, "ct")
        dose_out = os.path.join(DESTINATION_PATH, patient_name, "dose")
        plan_out = os.path.join(DESTINATION_PATH, patient_name, "plan")
        struct_out = os.path.join(DESTINATION_PATH, patient_name, "struct")
        for d in [ct_out, dose_out, plan_out, struct_out]:
            _ensure_dir(d)

        # Downsamplää CT kuvat
        for f in ct_files:
            try:
                ds = dcmread(f)
                ds_ds = _downsample_ct_slice(ds, scale_xy=0.5)
                out_path = os.path.join(ct_out, os.path.basename(f))
                ds_ds.save_as(out_path)
            except Exception as e:
                warnings.warn(f"{patient_name}: CT-tiedoston {os.path.basename(f)} käsittely epäonnistui: {e}")

        # Downsamplää RTDOSE 
        if rd_files:
            rd_path = rd_files[0]
            try:
                ds = dcmread(rd_path)
                ds_ds = _downsample_dose(ds, scale_xy=0.5)
                out_path = os.path.join(dose_out, os.path.basename(rd_path))
                ds_ds.save_as(out_path)
            except Exception as e:
                warnings.warn(f"{patient_name}: RD-tiedoston käsittely epäonnistui: {e}")

        # Kopioi RP ja RS 
        for f in rp_files:
            try:
                ds = dcmread(f, stop_before_pixels=True)
                ds.save_as(os.path.join(plan_out, os.path.basename(f)))
            except Exception as e:
                warnings.warn(f"{patient_name}: RP-tiedoston {os.path.basename(f)} kopiointi epäonnistui: {e}")

        for f in rs_files:
            try:
                ds = dcmread(f, stop_before_pixels=True)
                ds.save_as(os.path.join(struct_out, os.path.basename(f)))
            except Exception as e:
                warnings.warn(f"{patient_name}: RS-tiedoston {os.path.basename(f)} kopiointi epäonnistui: {e}")

    print("DONE")


if __name__ == "__main__":
    print("PROCESSING...")
    downsample_dicom_folder('RAX')
    print("DONE")
