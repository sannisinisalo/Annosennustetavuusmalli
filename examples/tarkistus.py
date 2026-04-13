# -*- coding: utf-8 -*-
"""
Luotu Ti 11.11.2025
Tekijä: Sanni Sinisalo

Koodi, joka piirtää CT-kuvien päälle yhden kohdassa roi_name="" määritetyn ROI:n. 
"""

import numpy as np
import matplotlib.pyplot as plt
from rt_utils import RTStructBuilder
from pathlib import Path
from typing import Optional
from loguru import logger  
from pydicom import FileDataset, dcmread

def load_CT(path: Path | str) -> list[FileDataset]:
    path = Path(path)

    logger.debug(f"Loading CT images from: {path}")

    slices = [
        dcmread(f)
        for f in path.glob("*.dcm")
        if dcmread(f, stop_before_pixels=True).Modality == "CT"
    ]

    if not slices:
        logger.warning(f"No CT slices found in directory: {path}")
        raise FileNotFoundError(f"No CT slices found in directory: {path}")

    series_uid = slices[0].SeriesInstanceUID
    logger.debug(f"Found SeriesInstanceUID: {series_uid} in first CT slice.")
    slices = [s for s in slices if s.SeriesInstanceUID == series_uid]
    logger.debug(f"Number of CT slices with matching SeriesInstanceUID: {len(slices)}")

    slices.sort(key=lambda x: int(x.InstanceNumber))
    logger.debug("CT slices sorted by InstanceNumber.")
    slices = slices[::-1]
    logger.debug("CT slices order reversed to match expected orientation.")

    return slices

def normalize_axes(
    mask: np.ndarray, ct_slices: list[FileDataset]
) -> Optional[tuple[np.ndarray, str]]:
    num_slices = len(ct_slices)  
    rows = int(ct_slices[0].Rows)  
    cols = int(ct_slices[0].Columns)  

    shape = mask.shape

    if len(shape) != 3:
        logger.warning(
            f"Maskin muoto {shape} ei ole 3-ulotteinen, ei onnistuttu normalisoimaan"
        )
        return None

    txt = "Maskin akselit: oletetaan (Z,Y,X)"
    if shape == (num_slices, rows, cols):
        return mask, txt

    for perm in [
        (0, 1, 2),
        (0, 2, 1),
        (1, 0, 2),
        (1, 2, 0),
        (2, 0, 1),
        (2, 1, 0),
    ]:
        trial = np.transpose(mask, axes=perm)

        if trial.shape == (num_slices, rows, cols):
            return trial, f"Maskin akselit korjattu transpoosilla{perm} -> (Z,Y,X)"
        else:
            logger.debug(
                f"Maskin akselien permutaatio {perm} tuotti muodon {trial.shape}, ei haluttu (Z,Y,X)"
            )
            continue
    logger.warning(
        f"Maskin muoto {shape} ei saatu normalisoitua haluttuun (Z,Y,X) muotoon, kaikki permutaatiot testattu"
    )
    return None


def overlay_roi_on_ct(rt_path, ct_path, roi_name, flip_ud=False, flip_lr=False): # Lataa CT-sarjan ja RS:n, normalisoi maskin akselit, piirtää CT-leikkeen ROI:n kanssa päällekkäin
    
    ct_slices = load_CT(ct_path) # Lataa CT-sarjan aikaisemmin määritellyn funktion avulla
    rows = int(ct_slices[0].Rows) # CT-sarjan y-suunta
    cols = int(ct_slices[0].Columns) # CT-sarjan x-suunta
    num_slices = len(ct_slices) # CT-sarjan z-suunta

    # 3D Ct-volyymin rakentamien 
    ct_vol = np.stack([s.pixel_array for s in ct_slices], axis=0) # Muodostetaan numpy-taulukko, jokainen siivu (2D) kootaan z-akselille
    assert ct_vol.shape == (num_slices, rows, cols), f"CT-volyymi shape {ct_vol.shape} ei täsmää." # Tarkistetaan, että datassa ei virheitä 

    # Ladataan RTstruct ja haetaan ROI:t
    rtstruct = RTStructBuilder.create_from(dicom_series_path=ct_path, rt_struct_path=rt_path) # Luodaan RTStructBuilder-objekti, joka osaa lukea RS:n ja resampolata ROI:t CT:n koordinaatistoon
    try: # Haetaan nimiin perustuva ROI
        mask = rtstruct.get_roi_mask_by_name(roi_name)
    except TypeError: # try-except rakenne kiertää yleisen TypeError bugin 
        mask = rtstruct.get_roi_mask_by_name(roi_name)

    # Normalisoidaan maskin akselit (Z,Y,X)
    mask_fixed, note = normalize_axes(mask, ct_slices) # Varmistaa maskin orientaation vastaavan CT:n orientaatiota
    print(note)
    print(f"CT volume shape: {ct_vol.shape}, ROI mask shape: {mask_fixed.shape}") # Tulostetaan miten maski ja CT asettuvat

    # Varmistetaan, että arvot ovat binäärisiä
    mask_fixed = (mask_fixed > 0).astype(np.uint8)

    # Valinnaiset käännöt ylös-alas tai vasen-oikea
    if flip_ud:
        ct_vol = np.flip(ct_vol, axis=1)   # Kääntää y-suunnassa CT:n ja maskin 
        mask_fixed = np.flip(mask_fixed, axis=1)
    if flip_lr:
        ct_vol = np.flip(ct_vol, axis=2)   # Kääntää x-suunnassa 
        mask_fixed = np.flip(mask_fixed, axis=2)

    # CT-kuvien ja ROI:den piirto 
    for i in range(num_slices): 
        ct_image = ct_vol[i] # Otetaan aina yksi CT
        roi_slice = mask_fixed[i] # ja sitä vastaava ROI-maskileike

        # Peitetään alueet, joissa maski on 0 eli esim. tausta
        overlay = np.ma.masked_where(roi_slice == 0, roi_slice)

        plt.imshow(np.flipud(ct_image), cmap="gray", origin="lower") # Piirtää CT:n harmaana, np.flipud kääntää kuvan ylöalaisin, origin varmistaa oikean koordinaatiston suunnan
        plt.imshow(np.flipud(overlay), cmap="autumn", alpha=0.5, origin="lower") # Piirtää ROI:n värillä autumn, peittävyys 0.5
        plt.title(f"Slice {i} - ROI: {roi_name}")
        plt.axis("off")
        plt.show()


# Pääohjelma, tämä ajetaan vain, jos tämä ohjelma ajetaan sellaisenaan. Jos kooodi importoitu toiseen ohjelmaan, tätä ei ajeta 
if __name__ == "__main__": 
    overlay_roi_on_ct(
        rt_path=r"C:\Users\User01\GRADU\Aineisto\VN0ds\Patient47_VN0\struct\RS.1.2.246.352.221.5454258401965172541.996177027274946221.dcm",
        ct_path=r"C:\Users\User01\GRADU\Aineisto\VN0ds\Patient47_VN0\vanha ct",
        roi_name="PTV-iho",
        flip_ud=False,  # Jos CT tai maski ylösalaisin, vaihda True
        flip_lr=False   # Jos CT tai maski pelikuvana, vaihda True
    )
    

