# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 10:27:14 2026

@author: User01
"""

import pydicom
import matplotlib.pyplot as plt
import numpy as np

# RT Dose -tiedoston polku
dose_polku = r"C:\Users\User01\GRADU\Aineisto\VN0ds\Patient2_VN0\dose\RD.1.2.246.352.221.5243245925728214844.2197592380591182783.dcm"

ds = pydicom.dcmread(dose_polku)

# Luetaan annosdata
dose_raw = ds.pixel_array  # muoto: (z, y, x)
scaling = float(ds.DoseGridScaling)

dose = dose_raw * scaling  # Gy

print("Dose-matriisin muoto:", dose.shape)
print("Annoksen min/max (Gy):", dose.min(), dose.max())

# Näytetään viipaleet
for i in range(dose.shape[0]):
    plt.figure(figsize=(6, 6))
    plt.imshow(dose[i, :, :], cmap="inferno")
    plt.colorbar(label="Dose (Gy)")
    plt.title(f"RT Dose slice {i}")
    plt.axis("off")
    plt.show()
