# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 13:59:14 2025

@author: User01
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from rt_utils import RTStructBuilder
import pydicom


def load_ct_series(path): #Funktio ottaa yhden parametrin, jossa CT-kuvat sijaitsevat
    slices = [
        pydicom.dcmread(os.path.join(path, f)) # Lukee DICOM tiedoston, parametreina path ja silmukan muuttuja f (tiedostonimi)
        for f in os.listdir(path) # Käydään läpi jokainen tiedosto kerrallaan. Jokainen tiedosto annetaan muuttujalle f 
        if f.endswith(".dcm") 
    ] # slices on lista kaikista DICOM kuvista hakemistossa
    if not slices: # Virhe, jos hakemistossa ei DICOM tiedostoja
        raise ValueError("CT-hakemistossa ei ole .dcm-tiedostoja.")

    # SeriesInstanceUID tunnistaa yhden sarjan tutkimuksen sisällä 
    series_uid = slices[0].SeriesInstanceUID # Ensimmäisen kuvan UID valitaan oletuksena 
    slices = [s for s in slices if s.SeriesInstanceUID == series_uid] # Käydään läpi jokainen listan kuva ja tarkistetaan kuuluuko se samaan sarjaan kuin ensimmäinen kuva
    if not slices: # Virhe, jos slices tyhjä  
        raise ValueError("Yhtään CT-slicea ei vastaa valittua SeriesInstanceUID:ia.")

    # InstanceNumber on DICOM-metatieto, joka kertoo kuvan järjestysnumeron CT-sarjassa
    slices.sort(key=lambda x: int(x.InstanceNumber)) # Järjestää kuvat InstanceNumberin mukaan nousevaan järjestykseen
    slices = slices[::-1] # kääntää listan järjestyksen päinvastaiseksi
    return slices # palauttaa listan DICOM-kuvadatan



def normalize_mask_axes(mask, ct_slices): # selvittäää onko maski oikeassa muodossa ja tarvittaessa korjaa sen (Z,Y,X)-muotoon
    
    num_slices = len(ct_slices) # Siivujen lukumäärä (Z)
    rows = int(ct_slices[0].Rows) # Kuvan korkeus (Y)
    cols = int(ct_slices[0].Columns) # Kuvan leveys (X)

    msg = "Mask axes: assumed (Z,Y,X)" # msg palautetaan, jos mitään korjausta ei tarvitse tehdä 
    shape = mask.shape # Kertoo maskin nykyisen akselijärjestyksen 

    # 1) Maski on jo oikean muotoinen
    if shape == (num_slices, rows, cols):
        return mask, msg

    # 2) Maksi on muodossa (Y, X, Z) -> siirretään Z eteen
    if shape == (rows, cols, num_slices):
        mask_fixed = np.moveaxis(mask, 2, 0)  # Siirtää paikalla 2 olevan Z-akslein 1. paikalle eli kohtaan 0
        return mask_fixed, "Mask axes corrected: (Y,X,Z) -> (Z,Y,X) via moveaxis(2->0)"

    # 3) Maski muodossa (X, Y, Z) -> ensin vaihda X/Y, sitten siirrä Z
    if shape == (cols, rows, num_slices):
        mask_swapped = np.swapaxes(mask, 0, 1)  # Vaihdetaan maskin 1. ja 2. paikalla olevien akselien paikkaa keskenään
        mask_fixed = np.moveaxis(mask_swapped, 2, 0) # Siirtää Z ensimmäiseksi 
        return mask_fixed, "Mask axes corrected: (X,Y,Z) -> swapaxes(0,1) + moveaxis(2->0)"

    # 4) Maski muodossa (Z, X, Y) -> vaihda X/Y
    if shape == (num_slices, cols, rows):
        mask_fixed = np.swapaxes(mask, 1, 2)
        return mask_fixed, "Mask axes corrected: (Z,X,Y) -> swapaxes(1,2)"

    # 5) Yleisvarmistus: etsi permutaatio, joka tuottaa (num_slices, rows, cols). Tämä riittäisi yksinäänkin, mutta muut tapaukset selvyyden vuoksi hyviä 
    for perm in [
        (0, 1, 2),
        (0, 2, 1),
        (1, 0, 2),
        (1, 2, 0),
        (2, 0, 1),
        (2, 1, 0),
    ]:
        if len(shape) == 3:
            trial = np.transpose(mask, axes=perm) # järjestää maskin akselit uudelleen annettujen indeksien mukaan
            if trial.shape == (num_slices, rows, cols): # Tarkistaa tuottaako aksleipermutaatio halutun lopputulon
                return trial, f"Mask axes corrected by transpose{perm} -> (Z,Y,X)"

    raise ValueError(
        f"Mask shape {shape} ei ole yhteensopiva CT:n (Z,Y,X)=({num_slices},{rows},{cols}) kanssa."
    ) # Virhe, jos maskin muoto ei sovi CT:n kanssa


def overlay_roi_on_ct(rt_path, ct_path, roi_name, flip_ud=False, flip_lr=False): # Lataa CT-sarjan ja RS:n, normalisoi maskin akselit, piirtää CT-leikkeen ROI:n kanssa päällekkäin
    
    ct_slices = load_ct_series(ct_path) # Lataa CT-sarjan aikaisemmin määritellyn funktion avulla
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
    mask_fixed, note = normalize_mask_axes(mask, ct_slices) # Varmistaa maskin orientaation vastaavan CT:n orientaatiota
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
        rt_path=r"C:\Users\User01\GRADU\Aineisto\VN0ds\Patient2_VN0\struct\RS.1.2.246.352.221.5081513604484729159.13153969492833577387.dcm",
        ct_path=r"C:\Users\User01\GRADU\Aineisto\VN0ds\Patient2_VN0\vanha ct",
        roi_name="Keuhko sin",
        flip_ud=False,  # Jos CT tai maski ylösalaisin, vaihda True
        flip_lr=False   # Jos CT tai maski pelikuvana, vaihda True
    )
    

