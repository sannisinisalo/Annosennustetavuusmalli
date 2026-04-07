# -*- coding: utf-8 -*-
"""
Luotu Ti 16.12.2025 klo 9:24:17
Tekijä: Sanni Sinisalo

Toinen koodiyritys maskipakan luomiseen.
Ensimmäisestä versiosta poistettu
- intensitettimaskin luominen, koska se hävitti tiedon päällekkäisistä ROI:sta
- koodi muutettu lukemaan monta potilasta kerralla yhdestä kansiosta

Koodissa määritetään jokaiselle ROI:lle (Region Of Intrest) numero, jolla
annosennustettavuusmalli tunnistaa ne sekä koostetaan CT pakka, johon on
lisätty kaikki ROI:t.
"""

from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger  # type: ignore
from pydicom import FileDataset, dcmread
from pydicom.uid import generate_uid
from rt_utils import RTStructBuilder  # type: ignore

from annosennustettavuusmalli.preprocessing.luokat import (
    AllPatients,
)


def load_CT(path: Path | str) -> list[FileDataset]:
    """
    Loading the CT-images and arranging them by the InstanceNumber metadata.

    InstanceNumber on DICOM-metatieto, joka kertoo kuvan järjestysnumeron CT-sarjassa.
    Järjestetään kuvat ensin nousevaan järjestykseen InstanceNumberin mukaan, jonka
    jälkeen listan järjestys käännetään päinvastaiseksi, jotta se vastaa odotettua orientaatiota.

    Parameters
    ----------
    path : str or Path
        The path to the directory containing the CT DICOM series.

    Returns
    -------
    slices : list[FileDataset]
        A list of the loaded CT images in DICOM form, sorted by InstanceNumber.

    Raises
    ------
    FileNotFoundError
        If no CT slices are found in the specified directory.

    """

    path = Path(path)

    logger.debug(f"Loading CT images from: {path}")

    # Tuotetaan lista CT-kuvista. Varmistetaan, että kyseessä on CT modality ja että kaikki kuvat kuuluvat samaan sarjaan SeriesInstanceUID:n avulla.
    slices = [
        dcmread(f)
        for f in path.glob("*.dcm")
        if dcmread(f, stop_before_pixels=True).Modality == "CT"
    ]

    if not slices:
        logger.warning(f"No CT slices found in directory: {path}")
        raise FileNotFoundError(f"No CT slices found in directory: {path}")

    # Otetaan ensimmäisen kuvan UID ja jätetään listaan ne tiedostot, joiden UID matchaa ensimmäisen kanssa
    series_uid = slices[0].SeriesInstanceUID
    logger.debug(f"Found SeriesInstanceUID: {series_uid} in first CT slice.")
    slices = [s for s in slices if s.SeriesInstanceUID == series_uid]
    logger.debug(f"Number of CT slices with matching SeriesInstanceUID: {len(slices)}")

    # Järjestetään listan tiedostot InstanceNumberin mukaan ensin nousevaan järjestykseen, jonka jälkeen
    # listan järjestys käännetään päinvastaiseksi
    slices.sort(key=lambda x: int(x.InstanceNumber))
    logger.debug("CT slices sorted by InstanceNumber.")
    slices = slices[::-1]
    logger.debug("CT slices order reversed to match expected orientation.")

    return slices


# Normalisoidaan ROI-maskin akselit muotoon (Z, Y, X), jotta ne ovat samassa muodossa CT kuvien kanssa
def normalize_axes(
    mask: np.ndarray, ct_slices: list[FileDataset]
) -> Optional[tuple[np.ndarray, str]]:
    """
    Normalizes the axes of the ROI mask array tho match the CT-images axes (Z, Y, X).

    Parameters
    ----------
    mask : numpy.ndarray
        The 3D array representing the mask.
    ct_slices : [FileDataset]
        A list of the loaded CT images in DICOM form

    Returns
    -------
    mask_normalized : numpy.ndarray
        The mask array with axes reordered to (Z, Y, X).

    message : str
        A text message describing whether the mask was already correctly aligned
        or how the axes were transposed.

    """
    num_slices = len(ct_slices)  # Siivujen lukumäärä Z
    rows = int(ct_slices[0].Rows)  # Rivien määrä eli kuvan korkeus eli Y
    cols = int(ct_slices[0].Columns)  # Sarakkeiden määrä eli kuvan leveys eli X

    # Maskin alkuperäiset akselit
    shape = mask.shape

    # Jos maskin akselit ovat suoraan oikeat eikä korjausta tarvita, tulostetaan teksti txt
    txt = "Maskin akselit: oletetaan (Z,Y,X)"
    if shape == (num_slices, rows, cols):
        return mask, txt

    # Jos akselit vaativat korjausta, etsitään permutaatio, joka tuottaa (num_slices, rows, cols) ja
    # käännetään akselit sen mukaan haluttuun järjestykseen.
    for perm in [
        (0, 1, 2),
        (0, 2, 1),
        (1, 0, 2),
        (1, 2, 0),
        (2, 0, 1),
        (2, 1, 0),
    ]:
        if len(shape) == 3:
            trial = np.transpose(mask, axes=perm)

            # Tarkistetaan vielä tuottiko muutos halutun lopputuloksen
            if trial.shape == (num_slices, rows, cols):
                return trial, f"Maskin akselit korjattu transpoosilla{perm} -> (Z,Y,X)"


# ROI nimien määritys ja numeroiden määrääminen
def map_roi_name_to_label(roi_name: str) -> Optional[int]:
    """
    Maps a ROI name to a predefined number that are powers of 2.

    Parameters
    ----------
    roi_name : str
        Name of the ROI.

    Returns
    -------
    int
        A number representing the ROI class if a match is found.
    None
        If the ROI is not recognized or should be excluded.

    """
    # Muuttaa ROI:n nimen pieniksi kirjaimiksi sekä poistaa turhat välilyönnit nimen edestä ja lopusta
    roi = roi_name.lower().strip()

    # Määritellään jokainen mallin haluama ROI ja sen nimet sekä sitä vastaavan lukuarvon
    if "sydän vasen" in roi or "sydan vasen" in roi:
        return None

    if "#ptv" in roi:
        return None

    if "keuhko sin-ptv 40gy" in roi:
        return None

    if "body" in roi:
        return 0

    if "ptv iho" in roi or "ptv-iho" in roi:
        return 1

    if "heart" in roi or "sydän" in roi or "sydan" in roi:
        return 2

    if "keuhko dex" in roi or "lung_l" in roi:
        return 4

    if "keuhko sin" in roi or "lung_r" in roi:
        return 8

    if "rinta dex" in roi or "breast_r" in roi:
        return 16

    if "lad" in roi or "a_lad" in roi:
        return 32

    if "humerus head_l" in roi or "olkanivel sin" in roi:
        return 64

    if "plexus" in roi or "brachial plexus" in roi or "brachial_plexus" in roi:
        return 128

    if "esophagus" in roi or "ruokatorvi" in roi:
        return 256

    if "trachea" in roi or "tracea" in roi:
        return 512

    if "thyroid" in roi or "kilpirauhanen" in roi:
        return 1024

    # Jos ROI ei vastaa mitään mainittua, ROI:lle ei anneta numeroa, vaan arvo None
    return None


# Asetetaan ROI:t CT-kuvien päälle
def overlay_ROI(rt_path: str | Path, ct_path: str | Path):
    """
    overlays the ROIs on top of the CT-images.

    Parameters
    ----------
    rt_path : str
        Path to the RT Structure Set (RTSTRUCT) DICOM file.
    ct_path : str
        Path to the directory containing the CT DICOM series.

    Returns
    -------
    sum_mask : numpy.ndarray
        A 3D integer array of shape (Z, Y, X) where each voxel contains:
        - a power-of-two label value representing the ROIs covering it
        - `-1` if no ROI covers that voxel
    ct_slices : list
        A list of the loaded CT images in DICOM form

    """

    # Ladataan CT-kuvat käyttäen aikaisemmin määriteltyä Load_CT funktiota
    ct_slices = load_CT(ct_path)
    num_slices = len(ct_slices)
    rows = int(ct_slices[0].Rows)
    cols = int(ct_slices[0].Columns)

    # Luodaan RTStructBuilder-objekti, joka osaa lukea RS:n ja resamplata ROI:t CT:n koordinaatistoon
    rtstruct = RTStructBuilder.create_from(
        dicom_series_path=str(ct_path), rt_struct_path=str(rt_path)
    )

    # Luodaan ensin tyhjä summamaski
    # Alustetaan tausta arvoksi ensin 0, tämä muutetaan myöhemmin arvoon -1
    sum_mask = np.zeros((num_slices, rows, cols), dtype=np.int32)

    # Luodaan bool-taulukko, joka tosi, kun pikselissä vähintään yksi ROI
    any_mask = np.zeros((num_slices, rows, cols), dtype=bool)

    # Listataan kaikki saatavilla olevat ROI:t
    roi_list = rtstruct.get_roi_names()

    # Määritetään ROI listan halutuille ROI:lle map_roi_name_to_label funktiossa määritetyt luvut (2:n potenssi)
    for roi_name in roi_list:
        roi_value = map_roi_name_to_label(roi_name)

        if roi_value is None:
            continue

        try:
            mask = rtstruct.get_roi_mask_by_name(roi_name)
        except AttributeError:
            print(f"ROI '{roi_name}' ohitettu (ei ContourSequenceä)")
            continue

        normalised_axes = normalize_axes(mask, ct_slices)
        if normalised_axes is None:
            print(f"ROI '{roi_name}' ohitettu (ei onnistuttu normalisoimaan)")
            continue
        else:
            mask, _ = normalised_axes
        # print(f"{roi_name}: {txt}")

        any_mask |= mask.astype(bool)

        sum_mask |= mask.astype(np.int32) * roi_value

    # Muutetaan pikselit, joita mikään ROI ei peittänyt, arvolle -1
    sum_mask[~any_mask] = -1

    # Palautetaan summamaski ja CT-lista
    return sum_mask, ct_slices


# Tallennetaan maski DICOM-pakkana
def save_mask_as_dicom_series(
    mask: np.ndarray, ct_slices: list[FileDataset], output_folder: str | Path
) -> None:
    """
    Saves the mask as a DICOM series using CT slice metadata.

    Parameters
    ----------
    mask : numpy.ndarray
        A 3D array (Z, Y, X) containing the mask data to be saved.
    ct_slices : list[FileDataset]
        A list of the loaded CT images in DICOM form.
    output_folder : str | Path
        Path to the folder where the DICOM mask series will be saved.

    Returns
    -------
    None.

    """
    output_folder = Path(output_folder)

    # Luodaan output-kansio, jos sitä ei vielä ole
    output_folder.mkdir(parents=True, exist_ok=True)

    # Generoidaan uusi SeriesInstanceUID maskisarjalle
    series_uid = generate_uid()

    # Käydään läpi kaikki slicet
    for idx, (slice_img, ct) in enumerate(zip(mask, ct_slices)):
        new_ds = ct.copy()

        # Pixel data
        new_ds.PixelData = slice_img.astype(np.int32).tobytes()
        new_ds.Rows, new_ds.Columns = slice_img.shape

        # Metadata maskille
        new_ds.SeriesDescription = "ROI MASK"
        new_ds.SeriesInstanceUID = series_uid
        new_ds.SOPInstanceUID = generate_uid()
        new_ds.InstanceNumber = idx + 1

        # Päivitetään ImagePositionPatient Z-koordinaatti
        new_ds.ImagePositionPatient = list(ct.ImagePositionPatient)
        new_ds.ImagePositionPatient[2] = (
            ct.ImagePositionPatient[2] + idx * ct.SliceThickness
        )

        new_ds.PixelSpacing = list(ct.PixelSpacing)
        new_ds.SliceThickness = ct.SliceThickness

        # Skaalausasetukset
        new_ds.RescaleIntercept = 0
        new_ds.RescaleSlope = 1

        # DICOM-tyyppiasetukset
        new_ds.PixelRepresentation = 1
        new_ds.BitsAllocated = 32
        new_ds.BitsStored = 32
        new_ds.HighBit = 31

        # Tallennus
        out_path = output_folder / f"mask_{idx:04d}.dcm"
        new_ds.save_as(out_path)


# PÄÄOHJELMA

if __name__ == "__main__":
    # Luodaan AllPatients-objekti
    all_patients = AllPatients(processed_root=Path("VN0ds"), original_root=Path("VN0"))

    # Käydään kaikki potilaat läpi numerojärjestyksessä
    for patient in all_patients.sorted_by_number():
        print(f"Käsitellään {patient.patient_folder}...")

        # Polut luokkien kautta
        ct_path = patient.ds_org_ct_dir
        out_path = patient.ds_maski_dir

        # Etsitään RS-tiedosto
        rs_file = patient.rs_files[0] if patient.rs_files else None
        if rs_file is None:
            print(f"RS-tiedostoa ei löytynyt potilaalta {patient.patient_folder}")
            continue

        # Luodaan maski
        mask, ct_slices = overlay_ROI(rs_file, ct_path)

        # Tulostetaan maskin tyyppi ja muoto
        print(type(mask), mask.shape)

        # Tallennetaan maski DICOM-sarjana
        save_mask_as_dicom_series(mask, ct_slices, out_path)

        print("Maski tallennettu")
