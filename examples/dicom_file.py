# -*- coding: utf-8 -*-
"""
Luotu Ti 18.11.2025
Tekijä: Sanni Sinisalo

Koodi, jolla DICOM-muotoisia kuvia voidaan tarkastella.
Koodissa myös vaihtoehtona tallentaa kuvat PNG-muodossa haluttuun kansioon.
"""

import os
import pydicom
import matplotlib.pyplot as plt
import numpy as np

kansio = r"C:\Users\User01\GRADU\Aineisto\VN0ds\Patient1_VN0\vanha ct"
# tallennusk = r"C:\Users\User01\GRADU\tiedostot\MASKI1_tuloksia"

dicom_lista = []

# Luetaan kaikki DICOM-tiedostot ja talletetaan sekä dataset että järjestysarvo
for tiedosto in os.listdir(kansio):
    if tiedosto.lower().endswith(".dcm"):
        polku = os.path.join(kansio, tiedosto)
        ds = pydicom.dcmread(polku)

        # Otetaan slice order mieluiten ImagePositionPatient -> Z-koordinaatti
        if "ImagePositionPatient" in ds:
            z = float(ds.ImagePositionPatient[2])
        elif "SliceLocation" in ds:
            z = float(ds.SliceLocation)
        elif "InstanceNumber" in ds:
            z = int(ds.InstanceNumber)
        else:
            z = 0

        dicom_lista.append((z, tiedosto, ds))

# Sortataan Z-koordinaatin perusteella
dicom_lista.sort(key=lambda x: x[0])

# Näytetään kuvat oikeassa järjestyksessä
for z, tiedosto, ds in dicom_lista:
    plt.figure(figsize=(6, 6))
    plt.imshow(ds.pixel_array, cmap="gray")
    plt.title(f"{tiedosto} (z={z})")
    plt.axis("off")
    
    # Tallenna kuva PNG-muodossa
    # tallennus_polku = os.path.join(tallennusk, f"{tiedosto}.png")
    # plt.savefig(tallennus_polku, bbox_inches='tight', pad_inches=0)
    
    plt.show()