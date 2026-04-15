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
from typing import List, Optional

import numpy as np
import typer
from loguru import logger
from pydicom import FileDataset, dcmread
from pydicom.multival import MultiValue
from scipy.ndimage import zoom  # type: ignore

from annosennustettavuusmalli.preprocessing.luokat import (
    BASE_DIR,
    AllPatients,
    Patient,
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


def downsample_ct_file(ct_file: Path) -> Optional[FileDataset]:
    """
    Downsamplaa CT-tiedoston puoleen alkuperäisestä resoluutiosta X/Y-suunnassa.
    Palauttaa downsamplatun DICOM datasetin, tai None jos käsittely epäonnistuu.
    """

    try:
        ds = dcmread(ct_file)

        arr_down = np.array(zoom(ds.pixel_array, zoom=(0.5, 0.5), order=1))

        ds.PixelData = arr_down.tobytes()

        ds.Rows, ds.Columns = arr_down.shape

        if hasattr(ds, "PixelSpacing"):
            ds.PixelSpacing = MultiValue(float, [float(x) * 2 for x in ds.PixelSpacing])

        return ds
    except Exception as e:
        warnings.warn(f"CT-tiedoston {ct_file.name} käsittely epäonnistui: {e}")
        return None


def downsample_dose_file(
    dose_file: Path, zoom_factor: float = 0.5
) -> Optional[FileDataset]:
    """
    Downsamplaa RTDOSE-tiedoston puoleen alkuperäisestä resoluutiosta X/Y-suunnassa (Z pysyy samana).

    Palauttaa downsamplatun DICOM datasetin, tai None jos käsittely epäonnistuu.
    """
    arr_dose_down = None
    ds_dose = None

    try:
        ds_dose = dcmread(dose_file)

        arr_dose_down = np.array(
            zoom(ds_dose.pixel_array, zoom=(1, zoom_factor, zoom_factor), order=1)
        )

        # Flipataan Z-akselilla, jotta slice-indeksi vastaa CT:n kanssa (slice 0 on alhaalla, ei päällä)
        arr_dose_down_flipped = arr_dose_down[::-1, :, :]

        ds_dose.PixelData = arr_dose_down_flipped.tobytes()

        ds_dose.Rows, ds_dose.Columns = (
            arr_dose_down_flipped.shape[1],
            arr_dose_down_flipped.shape[2],
        )

        # Tarvitaanko näitä?
        # dose_origin_z = ds_dose.ImagePositionPatient[2]
        # dose_z_positions = dose_origin_z + np.array(ds_dose.GridFrameOffsetVector)

        if hasattr(ds_dose, "PixelSpacing"):
            # Tuplataan PixelSpacing-arvot, koska resoluutio puolitetaan
            ds_dose.PixelSpacing = MultiValue(
                float, [float(x) * 1 / zoom_factor for x in ds_dose.PixelSpacing]
            )

        return ds_dose
    except Exception as e:
        warnings.warn(f"RTDOSE-tiedoston {dose_file.name} käsittely epäonnistui: {e}")
        return None


def downsample_mask_file(
    mask_file: Path, target_shape: tuple[int, int]
) -> Optional[FileDataset]:
    """
    Downsamplaa maskitiedoston puoleen alkuperäisestä resoluutiosta X/Y-suunnassa, ja slice-reversoi Z-akselilla.

    Palauttaa downsamplatun DICOM datasetin, tai None jos käsittely epäonnistuu.
    """
    try:
        ds_mask = dcmread(mask_file)

        # Tämä on 2D slice (Y, X)
        mask_orig = ds_mask.pixel_array

        # Downsample X/Y
        zoom_y = target_shape[0] / mask_orig.shape[0]
        zoom_x = target_shape[1] / mask_orig.shape[1]

        mask_down = np.array(
            zoom(
                mask_orig,
                zoom=(zoom_y, zoom_x),
                order=0,  # tärkeä maskille
            )
        )

        ds_mask.PixelData = mask_down.astype(np.int32).tobytes()

        ds_mask.Rows = mask_down.shape[0]
        ds_mask.Columns = mask_down.shape[1]

        # TÄRKEÄ: sama kuin alkuperäinen
        if hasattr(ds_mask, "PixelSpacing"):
            ds_mask.PixelSpacing = MultiValue(
                float, [float(x) * 2 for x in ds_mask.PixelSpacing]
            )

        return ds_mask

    except Exception as e:
        warnings.warn(f"Maskitiedoston {mask_file.name} käsittely epäonnistui: {e}")
        return None


app = typer.Typer()


@app.command()
def main(
    dataset: str = typer.Option(
        "L", "-d", "--dataset", help="Valitse datasetti: 'L', 'R', 'LAX', 'RAX'"
    ),
    patient: Optional[str] = typer.Option(
        None,
        "-p",
        "--patient",
        help="Valitse yksittäinen potilas (esim. 'Patient1_VN0') tai jätä tyhjäksi käsitelläksesi kaikki",
    ),
    base_dir: Path = typer.Option(
        BASE_DIR, "-b", "--base-dir", help="Peruskansio, jossa datasetit sijaitsevat"
    ),
) -> None:

    print(f"PROCESSING DATASET: {dataset}")
    print(f"BASE DIRECTORY: {base_dir}")

    if not base_dir.exists():
        logger.error(f"Base directory {base_dir} does not exist. Exiting.")
        return

    SOURCE_PATH, DESTINATION_PATH = get_data_paths(dataset, base_dir)

    if patient:
        patient_folder = SOURCE_PATH / patient
        if not patient_folder.exists():
            logger.error(f"Patient folder {patient_folder} does not exist. Exiting.")
            return
        patients = [
            Patient(
                original_root=Path(SOURCE_PATH.name),
                processed_root=Path(DESTINATION_PATH.name),
                patient_folder=Path(patient_folder.name),
                basedir=base_dir,
            )
        ]
    else:
        patients = AllPatients(
            original_root=SOURCE_PATH, processed_root=DESTINATION_PATH
        ).patients

    for p in patients:
        patient_name = p.patient_folder.name
        print(f"Processing {patient_name}...")

        # Downsamplataan CT-kuvat
        for f in p.ct_files:
            ds_ct = downsample_ct_file(f)
            if ds_ct is not None:
                ds_ct.save_as(p.ds_ct_dir / f.name)

        # Downsamplataan RTDOSE
        # 1. Ladataan alkuperäinen dose
        ds_dose = (
            downsample_dose_file(p.rd_file, zoom_factor=0.5) if p.rd_file else None
        )
        if ds_dose is not None and p.rd_file is not None:
            ds_dose.save_as(p.ds_dose_dir / p.rd_file.name)

        # 3. Ladataan maskit ja downsamplataan samaan kokoon
        if ds_dose is None:
            warnings.warn(
                f"{patient_name}: Dose-tiedosto puuttuu, maskien downsamplaus ohitetaan."
            )
            continue

        y_shape = ds_dose.pixel_array.shape[1]
        x_shape = ds_dose.pixel_array.shape[2]

        for i, f in enumerate(p.maski_files):
            ds_mask = downsample_mask_file(f, target_shape=(y_shape, x_shape))
            if ds_mask is not None:
                ds_mask.save_as(p.ds_maski_dir / f.name)

            # # TARVITAANKO TÄTÄ? --- IGNORE ---
            # # Slice-reversointi: käännä index Z-akselilla
            # idx = len(p.maski_files) - 1 - i
            # # Sovitetaan dose tähän sliceen
            # # HUOM! Ei tehdä maskin mukaan multiplicaatiota, vaan indeksi vain järjestää
            # dose_slice = ds_dose.pixel_array[idx, :, :]

        # Kopioidaan RS ja RP muuttumattomina
        try:
            ds = dcmread(p.rp_file, stop_before_pixels=True) if p.rp_file else None
            if ds is not None and p.rp_file is not None:
                ds.save_as(p.ds_plan_dir / p.rp_file.name)
        except Exception as e:
            warnings.warn(f"{patient_name}: RP-tiedoston kopiointi epäonnistui: {e}")
        try:
            ds = dcmread(p.rs_file, stop_before_pixels=True) if p.rs_file else None
            if ds is not None and p.rs_file is not None:
                ds.save_as(p.ds_struct_dir / p.rs_file.name)
        except Exception as e:
            warnings.warn(f"{patient_name}: RS-tiedoston kopiointi epäonnistui: {e}")

    print("DONE")


if __name__ == "__main__":
    app()
