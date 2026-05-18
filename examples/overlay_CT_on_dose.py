# -*- coding: utf-8 -*-
"""
Luotu Ma 02.03.2026
Tekijä: Sanni Sinisalo

Koodi jolla voidaan sijoittaa visualisointitarkoituksessa RTDose CT-kuvien tai maskin kuvien päälle
"""

import pydicom
import matplotlib.pyplot as plt
import SimpleITK as sitk
import os


# Lataa CT-sarja oikeassa järjestyksessä
ct_folder = r"C:\Users\User01\GRADU\Aineisto\VN0ds\Patient1_VN0\vanha ct"
ct_files_unsorted = [os.path.join(ct_folder, f) for f in os.listdir(ct_folder) if f.endswith(".dcm")]

# Lue jokaisen tiedoston ImagePositionPatient (z-koordinaatti)
positions = []
for f in ct_files_unsorted:
    ds = pydicom.dcmread(f, stop_before_pixels=True)
    pos = ds.ImagePositionPatient  
    positions.append((f, pos[2]))  

# Järjestä viipaleet vatsasta päähän (pienin z ensin, suurin z viimeiseksi)
ct_files_sorted = [f for f, z in sorted(positions, key=lambda x: x[1])]

# Lataa kuvat SimpleITK:lla
reader = sitk.ImageSeriesReader()
reader.SetFileNames(ct_files_sorted)
ct_img = reader.Execute()
ct_arr = sitk.GetArrayFromImage(ct_img)  

print("CT shape:", ct_arr.shape)


# Lataa annos ja resamplaa CT:n ruudukkoon
dose_path = r"C:\Users\User01\GRADU\Aineisto\VN0\Patient1_VN0\RD.1.2.246.352.221.4972727230104878982.15806810084685865633.dcm"
ds_dose = pydicom.dcmread(dose_path)

dose_img = sitk.ReadImage(dose_path, sitk.sitkFloat32)
dose_img = dose_img * float(ds_dose.DoseGridScaling)

# Resamplaus CT:n koordinaatistoon
resampler = sitk.ResampleImageFilter()
resampler.SetReferenceImage(ct_img)
resampler.SetInterpolator(sitk.sitkLinear)
resampler.SetDefaultPixelValue(0.0)

dose_resampled = resampler.Execute(dose_img)
dose_arr = sitk.GetArrayFromImage(dose_resampled)  

print("Dose shape:", dose_arr.shape)


# Visualisointi overlaynä kaikille viipaleille
num_slices = ct_arr.shape[0]

for slice_index in range(num_slices):
    ct_slice = ct_arr[slice_index]
    dose_slice = dose_arr[slice_index]

    if dose_slice.max() > 0:
        dose_norm = dose_slice / dose_slice.max()
    else:
        dose_norm = dose_slice 

    plt.figure(figsize=(8, 8))
    plt.imshow(ct_slice, cmap="gray", interpolation="none")
    plt.imshow(dose_norm, cmap="inferno", alpha=0.4, interpolation="none")
    plt.title(f"CT + Dose overlay (slice {slice_index})")
    plt.axis("off")
    plt.show()
