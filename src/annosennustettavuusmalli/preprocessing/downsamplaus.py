# -*- coding: utf-8 -*-
"""
Luotu Ti 27.1.2026
Tekijä: Akseli Leino
Muokkaaja: Sanni Sinisalo

Tiedostojen koon pienentämiseen käytetty koodi
"""

import re
import warnings
from pathlib import Path
from typing import List

import numpy as np
from pydicom import dcmread
from pydicom.multival import MultiValue
from scipy.ndimage import zoom  # type: ignore

from annosennustettavuusmalli.preprocessing.data import (  # type :ignore
    BASE_DIR,
    AllPatients,
)


def ensure_dir(path: str | Path) -> None:
    """
    Luodaan kansio, jos sitä ei vielä ole
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)


def find_patient_folders(source_root: str | Path) -> List[Path]:
    """
    Etsitään kaikki potilaskansiot, jotka alkaa 'Patient', ja järjestetään ne numerojärjestykseen
    """

    source_root = Path(source_root)

    patients = [p for p in source_root.glob("Patient*") if p.is_dir()]

    def patient_sort_key(path: Path):
        name = path.name
        m = re.search(r"(\d+)", name)
        return int(m.group(1)) if m else float("inf")

    return sorted(patients, key=patient_sort_key)


def find_dicom_by_modality(folder: str | Path, modality: str) -> List[Path]:
    """
    Etsitään kansiosta kaikki tietyn modaliteetin DICOM tiedostot
    """
    folder = Path(folder)

    files = [
        folder / f
        for f in folder.iterdir()
        if (folder / f).is_file() and f.suffix == ".dcm"
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


def get_data_paths(dataset: str, base_dir: Path = BASE_DIR) -> tuple[Path, Path]:
    """
    Mapataan datasetti lähde- ja kohdekansiohin
    """
    dataset = dataset.upper()
    if dataset == "L":
        return base_dir / "VN0", base_dir / "VN0ds"
    elif dataset == "R":
        return base_dir / "ON0", base_dir / "ON0ds"
    elif dataset == "LAX":
        return base_dir / "VN+", base_dir / "VN+ds"
    elif dataset == "RAX":
        return base_dir / "ON+", base_dir / "ON+ds"
    else:
        raise ValueError("Tuntematon dataset-parametri. Käytä: 'L', 'R', 'LAX', 'RAX'.")


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

    SOURCE_PATH, DESTINATION_PATH = get_data_paths(dataset)

    patients = AllPatients(
        original_root=SOURCE_PATH, processed_root=DESTINATION_PATH
    ).patients

    for p in patients:
        patient_name = p.patient_folder.name
        folder = p.patient_folder
        print(f"Processing {patient_name}...")

        # Downsamplataan CT-kuvat
        for f in p.ct_files:
            try:
                ds = dcmread(f)
                arr_down = zoom(ds.pixel_array, zoom=(0.5, 0.5), order=1)
                ds.PixelData = arr_down.tobytes() if hasattr(arr_down, "tobytes")
                ds.Rows, ds.Columns = arr_down.shape
                if hasattr(ds, "PixelSpacing"):
                    ds.PixelSpacing = MultiValue(
                        float, [float(x) * 2 for x in ds.PixelSpacing]
                    )
                ds.save_as(p.ds_ct_dir / f.name)
            except Exception as e:
                warnings.warn(
                    f"{patient_name}: CT-tiedoston {f.name} käsittely epäonnistui: {e}"
                )

        # Downsamplataan RTDOSE
        # 1. Ladataan alkuperäinen dose
        dose_src_folder = p.ds_dose_dir
        dose_files = [
            f for f in dose_src_folder.iterdir() if f.is_file() and f.suffix == ".dcm"
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
        mask_src_folder = p.ds_mask_dir
        mask_files = [
            f for f in mask_src_folder.iterdir() if f.is_file() and f.suffix == ".dcm"
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
            ds_mask.save_as(p.ds_mask_dir / f.name)

        # Lopuksi tallenna flipattu dose
        ds_dose.PixelData = arr_dose_down_flipped.tobytes()
        ds_dose.Rows, ds_dose.Columns = (
            arr_dose_down_flipped.shape[1],
            arr_dose_down_flipped.shape[2],
        )
        ds_dose.save_as(p.ds_doseds_dir / dose_files[0].name)

        # 6. Tallennetaan muokattu dose doseds-kansioon
        if ds_dose is not None and arr_dose_down is not None:
            try:
                ds_dose.PixelData = arr_dose_down.tobytes()
                ds_dose.save_as(
                    p.ds_doseds_dir / dose_files[0].name
                )
            except Exception as e:
                warnings.warn(
                    f"{patient_name}: RD-tiedoston tallennus doseds-kansioon epäonnistui: {e}"
                )

        # Kopioidaan RS ja RP muuttumattomina
        for f in p.rp_files:
            try:
                ds = dcmread(f, stop_before_pixels=True)
                ds.save_as(p.ds_plan_dir / f.name)
            except Exception as e:
                warnings.warn(
                    f"{patient_name}: RP-tiedoston kopiointi epäonnistui: {e}"
                )
        for f in p.rs_files:
            try:
                ds = dcmread(f, stop_before_pixels=True)
                ds.save_as(p.ds_struct_dir / f.name)
            except Exception as e:
                warnings.warn(
                    f"{patient_name}: RS-tiedoston kopiointi epäonnistui: {e}"
                )

    print("DONE")
