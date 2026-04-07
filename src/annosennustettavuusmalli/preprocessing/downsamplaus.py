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

from annosennustettavuusmalli.preprocessing.luokat import BASE_DIR


def ensure_dir(path: str | Path) -> None:
    """
    Luodaan kansio, jos sitä ei vielä ole
    """
    os.makedirs(path, exist_ok=True)


def find_patient_folders(source_root: str | Path) -> List[str]:
    """
    Etsitään kaikki potilaskansiot, jotka alkaa 'Patient', ja järjestetään ne numerojärjestykseen
    """
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

    return sorted(patients, key=patient_sort_key)


def find_dicom_by_modality(folder: str, modality: str) -> List[str]:
    """
    Etsitään kansiosta kaikki tietyn modaliteetin DICOM tiedostot
    """
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
    from argparse import ArgumentParser

    argparser = ArgumentParser(
        description="Downsample DICOM files for ANNOSENNUSTETTAVUUSMALLI"
    )
    argparser.add_argument(
        "-d",
        "--dataset",
        type=str,
        required=True,
        choices=["L", "R", "LAX", "RAX"],
        help="Valitse datasetti: 'L', 'R', 'LAX', 'RAX'",
    )
    args = argparser.parse_args()

    print("PROCESSING...")

    # Valitse datasetti: "L", "R", "LAX", "RAX"
    dataset = args.dataset.upper()

    # Mapataan datasetti lähde- ja kohdekansiohin
    if dataset == "L":
        SOURCE_PATH = BASE_DIR / "VN0"
        DESTINATION_PATH = BASE_DIR / "VN0ds"
    elif dataset == "R":
        SOURCE_PATH = BASE_DIR / "ON0"
        DESTINATION_PATH = BASE_DIR / "ON0ds"
    elif dataset == "LAX":
        SOURCE_PATH = BASE_DIR / "VN+"
        DESTINATION_PATH = BASE_DIR / "VN+ds"
    elif dataset == "RAX":
        SOURCE_PATH = BASE_DIR / "ON+"
        DESTINATION_PATH = BASE_DIR / "ON+ds"
    else:
        raise ValueError("Tuntematon dataset-parametri. Käytä: 'L', 'R', 'LAX', 'RAX'.")

    ensure_dir(DESTINATION_PATH)
    patients = find_patient_folders(SOURCE_PATH)
    if not patients:
        print("Ei löytynyt potilaskansioita (Patient*).")
    else:
        for p in patients:
            patient_name = os.path.basename(p)
            print(f"Processing {patient_name}...")

            # Etsitään DICOM:it modaliteetin perusteella
            ct_files = find_dicom_by_modality(p, "CT")
            rd_files = find_dicom_by_modality(p, "RTDOSE")
            rp_files = find_dicom_by_modality(p, "RTPLAN")
            rs_files = find_dicom_by_modality(p, "RTSTRUCT")

            # Luodaan kohdekansiot
            folders = {
                "ct": os.path.join(DESTINATION_PATH, patient_name, "ct"),
                "dose": os.path.join(DESTINATION_PATH, patient_name, "dose"),
                "doseds": os.path.join(DESTINATION_PATH, patient_name, "doseds"),
                "plan": os.path.join(DESTINATION_PATH, patient_name, "plan"),
                "struct": os.path.join(DESTINATION_PATH, patient_name, "struct"),
                "mask": os.path.join(DESTINATION_PATH, patient_name, "maskids"),
            }
            for d in folders.values():
                ensure_dir(d)

            # Downsamplataan CT-kuvat
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

            # Downsamplataan RTDOSE
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
                    # 2. Downsamplataan Y/X (Z pysyy samana)
                    arr_dose_down = zoom(arr_dose, zoom=(1, 0.5, 0.5), order=1)
                    ds_dose.Rows, ds_dose.Columns = (
                        arr_dose_down.shape[1],
                        arr_dose_down.shape[2],
                    )
                    dose_origin_z = ds_dose.ImagePositionPatient[2]
                    dose_z_positions = dose_origin_z + np.array(
                        ds_dose.GridFrameOffsetVector
                    )
                    if hasattr(ds_dose, "PixelSpacing"):
                        ds_dose.PixelSpacing = MultiValue(
                            float, [float(x) * 2 for x in ds_dose.PixelSpacing]
                        )
                except Exception as e:
                    warnings.warn(
                        f"{patient_name}: RD-tiedoston käsittely epäonnistui: {e}"
                    )

            # 3. Ladataan maskit ja downsamplataan samaan kokoon
            mask_src_folder = os.path.join(DESTINATION_PATH, patient_name, "maski")
            mask_files = [
                os.path.join(mask_src_folder, f)
                for f in os.listdir(mask_src_folder)
                if os.path.isfile(os.path.join(mask_src_folder, f))
            ]
            arr_dose_down_flipped = arr_dose_down[::-1, :, :]

            for i, f in enumerate(mask_files):
                ds_mask = dcmread(f)
                mask_orig = ds_mask.pixel_array

                # Downsamplataan X/Y
                zoom_y = arr_dose_down_flipped.shape[1] / mask_orig.shape[0]
                zoom_x = arr_dose_down_flipped.shape[2] / mask_orig.shape[1]
                mask_down = zoom(mask_orig, zoom=(zoom_y, zoom_x), order=0)

                # Slice-reversointi: käännä index Z-akselilla
                idx = len(mask_files) - 1 - i
                # Sovitetaan dose tähän sliceen
                # HUOM! Ei tehdä maskin mukaan multiplicaatiota, vaan indeksi vain järjestää
                dose_slice = arr_dose_down_flipped[idx, :, :]

                # Tallenna maski kuten ennen
                ds_mask.PixelData = mask_down.astype(np.int32).tobytes()
                ds_mask.Rows, ds_mask.Columns = mask_down.shape
                if hasattr(ds_mask, "PixelSpacing"):
                    ds_mask.PixelSpacing = MultiValue(
                        float, [float(x) * 2 for x in ds_mask.PixelSpacing]
                    )
                ds_mask.save_as(os.path.join(folders["mask"], os.path.basename(f)))

            # Lopuksi tallenna flipattu dose
            ds_dose.PixelData = arr_dose_down_flipped.tobytes()
            ds_dose.Rows, ds_dose.Columns = (
                arr_dose_down_flipped.shape[1],
                arr_dose_down_flipped.shape[2],
            )
            ds_dose.save_as(
                os.path.join(folders["doseds"], os.path.basename(dose_files[0]))
            )

            # 6. Tallennetaan muokattu dose doseds-kansioon
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

            # Kopioidaan RS ja RP muuttumattomina
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
