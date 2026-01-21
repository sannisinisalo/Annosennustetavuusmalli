# -*- coding: utf-8 -*-
"""
Luotu Ti 16.12.2025 klo 9:24:17

Toinen koodiyritys maskipakan luomiseen. 
Ensimmäisestä versiosta poistettu 
- intensitettimaskin luominen, koska se hävitti tiedon päällekkäisistä ROI:sta
- koodi muutettu lukemaan monta potilasta kerralla yhdestä kansiosta

Koodissa määritetään jokaiselle ROI:lle (Region Of Intrest) numero, jolla 
annosennustettavuusmalli tunnistaa ne sekä koostetaan CT pakka, johon on 
lisätty kaikki ROI:t.
"""


import os
import numpy as np
import pydicom
from rt_utils import RTStructBuilder
import re




# CT-sarjan lataaminen ja järjestäminen InstanceUID metatiedon mukaan. InstanceNumber on 
# DICOM-metatieto, joka kertoo kuvan järjestysnumeron CT-sarjassa
def Load_CT(path): 
    """
    Loading the CT-images and arranging them by the InstanceUID metadata.

    Parameters
    ----------
    path : str
        The path of the original CT-images.

    Returns
    -------
    slices : list 
        Arranged CT-images.

    """
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
    """
    Normalizes the axes of the ROI mask array tho match the CT-images axes (Z, Y, X).

    Parameters
    ----------
    mask : numpy.ndarray
        The 3D array representing the mask.
    ct_slices : list 
        A list of the loaded CT images in DICOM form

    Returns
    -------
    mask_normalized : numpy.ndarray
        The mask array with axes reordered to (Z, Y, X).

    message : str
        A text message describing whether the mask was already correctly aligned
        or how the axes were transposed.

    """
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
    ct_slices = Load_CT(ct_path) 
    num_slices = len(ct_slices) 
    rows = int(ct_slices[0].Rows) 
    cols = int(ct_slices[0].Columns) 
    
    # Luodaan RTStructBuilder-objekti, joka osaa lukea RS:n ja resampolata ROI:t CT:n koordinaatistoon
    rtstruct = RTStructBuilder.create_from(
        dicom_series_path=ct_path,
        rt_struct_path=rt_path)
        
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
        
        try:
            mask = rtstruct.get_roi_mask_by_name(roi_name)
        except AttributeError:
            print(f"ROI '{roi_name}' ohitettu (ei ContourSequenceä)")
            continue
            
        mask, txt = Normalize_axes(mask, ct_slices)
        # print(f"{roi_name}: {txt}")
            
        any_mask |= mask.astype(bool)
            
        sum_mask |= (mask.astype(np.int32) * roi_value)
            
    #Muutetaan pikselit, joita mikään ROI ei peittänyt, arvolle -1
    sum_mask[~any_mask] = -1
    
    # Palautetaan summamaski ja CT-lista
    return sum_mask, ct_slices
    


# Tallennetaan maski DICOM-pakkana
def save_mask_as_dicom_series(mask, ct_slices, output_folder):
    """
    Saves the mask as a DICOM series using CT slice metadata.

    Parameters
    ----------
    mask : numpy.ndarray
        A 3D array (Z, Y, X) containing the mask data to be saved.
    ct_slices : list
        A list of the loaded CT images in DICOM form.
    output_folder : str
        Path to the folder where the DICOM mask series will be saved.

    Returns
    -------
    None.

    """
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
        new_ds.PixelRepresentation = 1
        new_ds.BitsAllocated = 16
        new_ds.BitsStored = 16
        new_ds.HighBit = 15

        new_ds.save_as(out_path)



# PÄÄOHJELMA

# Kansio, jossa jokaisen potilaan kansio
patient_dir = r"C:\Users\User01\GRADU\Aineisto\VN0ds"

# Etsitään kaikki potilaskansiot, jotka alkavat "Patient"
patients = [d for d in os.listdir(patient_dir) if d.startswith("Patient")]

# Järjestetään kansiot numerojärjestykseen, muuten tulisi aakkosjärjestyksessä
def Patient_sort(name):
    """
    Function that sorts patients by number, not by letter 

    Parameters
    ----------
    name : str
        The name of the patient folder.

    Returns
    -------
    int
        The numeric value extracted from the folder name, or 0 if none found.
        
    """
    # Eristetään numero nimestä
    m = re.search(r'(\d+)', name)
    return int(m.group(1)) if m else 0

# Luodaan maski
patients = sorted(patients, key=Patient_sort)

# Käydään kaikki potilaat läpi
for patient in patients:
    print(f"Käsitellään {patient}...")

    ct_path = os.path.join(patient_dir, patient, "vanha ct")
    out_path = os.path.join(patient_dir, patient, "maski")

    # Etsitään RS-tiedosto potilaan struct-kansiosta
    struct_dir = os.path.join(patient_dir, patient, "struct")
    rs_files = [f for f in os.listdir(struct_dir) if f.startswith("RS.")]
    
    if not rs_files:
        print(f"RS-tiedostoa ei löytynyt potilaalta {patient}")
        continue  # hypätään tämän potilaan yli
    
    # Oletetaan, että halutaan ensimmäinen RS-tiedosto, jos niitä on useampi
    rs_path = os.path.join(struct_dir, rs_files[0])


    # Luodaan maski
    mask, ct_slices = Overlay_ROI(rs_path, ct_path)

    # Tulostetaan maskin tyyppi ja muoto
    print(type(mask), mask.shape)

    # Tallennetaan maski DICOM-sarjana
    save_mask_as_dicom_series(mask, ct_slices, out_path)
    
    print(f"Maski tallennettu")
