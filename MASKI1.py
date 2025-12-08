# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 09:46:21 2025

@author: User01
"""

# -*- coding: utf-8 -*-
"""
Luotu Ma 24.11.2025 klo 10:17:17

Ensimmäinen koodiyritys maskipakan luomiseen. 
- aloitetaan yhdellä potilaalla
- tämän jälkeen voidaan siirtyä käsittelemään kaikkia potilaita

Koodissa määritetään jokaiselle ROI:lle numero, jolla annosennustettavuusmalli 
tunnistaa ne sekä koostetaan CT pakka, johon on lisätty kaikki ROI:t.
"""

import os
import glob
import numpy as np
import cv2
import pydicom
from rt_utils import RTStructBuilder
import re
import matplotlib.pyplot as plt
from pathlib import Path

# Kansio, jossa downsamplatut CT-kuvat
file = r"C:\Users\User01\GRADU\Aineisto\VN0ds\Patient1_VN0\ct"

#Kansio, jossa muokattu RS tiedosto
file2 = r"C:\Users\User01\GRADU\Aineisto\VN0ds\Patient1_VN0\struct\RS_SKAALATTU.dcm"



# CT-sarjan lataaminen ja järjestäminen InstanceUID metatiedon mukaan. InstanceNumber on 
# DICOM-metatieto, joka kertoo kuvan järjestysnumeron CT-sarjassa
def Load_CT(path): 
    # Tuotetaan lista CT-kuvista, f silmukkamuuttuja, johon .dcm tiedostot tallentuvat 
    slices = [
        pydicom.dcmread(os.path.join(path, f)) 
        for f in os.listdir(path) 
        if f.endswith(".dcm") 
    ] 
    
    # Otetaan ensimmäisen kuvan UID ja jätetään listaan ne tiedostot, joiden UID matchaa esnimmäisen kanssa 
    series_uid = slices[0].SeriesInstanceUID 
    slices = [s for s in slices if s.SeriesInstanceUID == series_uid] 
    
    # Järjestetään listan tiedostot InstanceNumberin mukaan ensin nousevaan järjestykseen, jonka jälkeen
    # listan järjestys käännetään päinvastaiseksi
    slices.sort(key=lambda x: int(x.InstanceNumber)) 
    slices = slices[::-1] 
    
    return slices 



# Normalisoidaan ROI-maskin akselit muotoon (Z, Y, X), jotta ne ovat samassa muodossa CT kuvien kanssa
def Normalize_axes(mask, ct_slices): 
    
    num_slices = len(ct_slices) # Siivujen lukumäärä Z
    rows = int(ct_slices[0].Rows) # Rivien määrä eli kuvan korkeus eli Y
    cols = int(ct_slices[0].Columns) # Sarakkeiden määrä eli kuvan leveys eli X

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
def ROI_names(roi_name):
    # Muuttaa ROI:n nimen pieniksi kirjaimiksi sekä poistaa turhat välilyönnit nimen edestä ja lopusta 
    roi = roi_name.lower().strip()
    
    # Määritellään jokainen mallin haluama ROI ja sen nimet sekä sitä vastaavan lukuarvon 
    if("sydän vasen" in roi or
       "sydan vasen" in roi):
        return None
    
    if("#ptv" in roi):
        return None
    
    if ("keuhko sin-ptv 40gy" in roi):
        return None
    
    if ("body" in roi):
        return 0

    if ("ptv iho" in roi or
        "ptv-iho" in roi):
        return 1
    
    if ("heart" in roi or
        "sydän" in roi or
        "sydan" in roi):
        return 2
    
    if ("keuhko dex" in roi or 
        "lung_l" in roi):
        return 4 
    
    if ("keuhko sin" in roi or 
        "lung_r" in roi):
        return 8  
    
    if ("rinta dex" in roi or
        "breast_r" in roi):
        return 16
    
    if ("lad" in roi or 
        "a_lad" in roi):
        return 32
    
    if ("humerus head_l" in roi or
        "olkanivel sin" in roi):
        return 64
    
    if ("plexus" in roi or
        "brachial plexus" in roi or
        "brachial_plexus" in roi):
        return 128
    
    if("esophagus" in roi or
       "ruokatorvi" in roi):
        return 256
    
    if ("trachea" in roi or
        "tracea" in roi):
        return 512
    
    if ("thyroid" in roi or
        "kilpirauhanen" in roi):
        return 1024
    
    # Jos ROI ei vastaa mitään mainittua, ROI:lle ei anneta numeroa, vaan arvo None
    return None



# Asetetaan ROI:t CT-kuvien päälle
def Overlay_ROI(rt_path, ct_path):
    
    # Ladataan CT-kuvat käyttäen aikaisemmin määriteltyä Load_CT funktiota 
    ct_slices = Load_CT(ct_path) 
    num_slices = len(ct_slices) 
    rows = int(ct_slices[0].Rows) 
    cols = int(ct_slices[0].Columns) 
    
    # Muodostetaan numpy CT-kuvista pakka 
    # ct_vol = np.stack([s.pixel_array for s in ct_slices], axis=0) 

    # Luodaan RTStructBuilder-objekti, joka osaa lukea RS:n ja resampolata ROI:t CT:n koordinaatistoon
    rtstruct = RTStructBuilder.create_from(
        dicom_series_path=ct_path,
        rt_struct_path=rt_path
    )
    
    # Luodaan ensin tyhjä summamaski 
    # Alustetaan tausta arvoksi ensin 0, tämä muutetaan myöhemmin arvoon -1
    sum_mask = np.zeros((num_slices, rows, cols), dtype=np.int32)
    
    # Luodaan bool-taulukko, joka tosi, kun pikselissä vähintään yksi ROI
    any_mask = np.zeros((num_slices, rows, cols), dtype=bool)
    
    # Listataan kaikki saatavilla olevat ROI:t
    roi_list = rtstruct.get_roi_names()
    
    
    # Määritetään ROI listan halutuille ROI:lle ROI_names funktiossa määritetyt luvut (2:n potenssi)
    for roi_name in roi_list:
        roi_value = ROI_names(roi_name)
        if roi_value is None:
            continue
        
        mask = rtstruct.get_roi_mask_by_name(roi_name)
        
        mask, txt = Normalize_axes(mask, ct_slices)
        # print(f"{roi_name}: {txt}")
        
        any_mask |= mask.astype(bool)
        
        sum_mask |= (mask.astype(np.int32) * roi_value)
        
    #Muutetaan pikselit, joita mikään ROI ei peittänyt, arvolle -1
    sum_mask[~any_mask] = -1

    # Palautetaan summamaski (, CT-volyymi, ja CT-lista)
    return sum_mask, ct_slices



# Luodaan haluttuja ROI:ta vastaavalle arvolle intensiteetti, jolla väritys määräytyy
ROI_INTENSITY_MAP = {
    0: 800,      
    1: 3000,    
    2: 2800,    
    4: 2600,    
    8: 2400,    
    16: 2200,   
    32: 2000,   
    64: 1800,  
    128: 1600, 
    256: 1400, 
    512: 1200, 
    1024: 1000  
}



# Funktio, jolla annetaan ROI:lle intensiteettien painokertoimet
# Taustan arvoksi asetetaan 0, jolloin tausta näkyy mustana
def apply_intensity_weights(bitmask, roi_map, background_value=0):
    # Kopioidaan maski sellaisenaan NumPy-taulukoksi
    mask = np.array(bitmask, copy=True)

    # Määritellään taustan arvoksi -1 
    background_mask = (mask == -1)

    # Määritellään body:n arvoksi 0, mikä toteutuu silloin, kun background_mask=false
    body_mask = (mask == 0) & (~background_mask)  

    # Alustetaan output luomalla tyhjä taulukko samassa muodossa kuin maksi
    output = np.zeros(mask.shape, dtype=np.int16)

    # Käsitellään erikseen body:n arvo 0 ja asetetaan alueelle sen intensiteetti
    if 0 in roi_map:
        output[body_mask] = roi_map[0]

    # Käsitellään loput ROI:t ns. bittilogiikalla ohittaen arvon 0, joka käsiteltiin jo yllä 
    for roi_bit, intensity in roi_map.items():
        if roi_bit == 0:
            continue
        hits = (mask & roi_bit) != 0
        output[hits] += intensity

    # Asetetaan kaikille taustan pikseleille ennalta määritetty arvo, eli 0
    output[background_mask] = background_value

    # Palauttaa uuden tualukon, jossa kullekin pikselille on laskettu kokonaisintensiteetti
    return output



# Tallennetaan maski DICOM-pakkana
def save_mask_as_dicom_series(mask, ct_slices, output_folder):
    # Luodaan output-kansio, jos sitä ei vielä ole
    os.makedirs(output_folder, exist_ok=True)
    
    # Käydään läpi jokainen maskin ja CT-kuvien leike ja yhdistetään ne
    for idx, (slice_img, ct) in enumerate(zip(mask, ct_slices)):
        new_ds = ct.copy()

        # Päivitetään pikselidata
        new_ds.PixelData = slice_img.astype(np.int16).tobytes()
        new_ds.Rows, new_ds.Columns = slice_img.shape

        # Päivitä metadata
        new_ds.SeriesDescription = "ROI MASK"
        new_ds.SeriesInstanceUID = pydicom.uid.generate_uid()
        new_ds.SOPInstanceUID = pydicom.uid.generate_uid()

        # Päivitetään skaalausasetukset
        new_ds.RescaleIntercept = 0
        new_ds.RescaleSlope = 1

        out_path = os.path.join(output_folder, f"mask_{idx:04d}.dcm")
        new_ds.save_as(out_path)



# PÄÄOHJELMA
# Luodaan maski
mask, ct_slices = Overlay_ROI(file2, file)

# Tulostetaan maskin tyyppi ja muoto
print(type(mask))
print(mask.shape)   

# Muunnetaan maski intensiteettikuvaksi
gray = apply_intensity_weights(mask, ROI_INTENSITY_MAP)

# Kansio, johon maskin kuvat tallennetaan 
out = r"C:\Users\User01\GRADU\Aineisto\VN0ds\Patient1_VN0\maski"

# Tallentaa intensiteettikuvat uuteen DICOM-sarjaan
save_mask_as_dicom_series(gray, ct_slices, out)










