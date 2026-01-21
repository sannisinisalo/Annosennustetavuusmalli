# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 09:20:58 2025

@author: User01
"""

import os
import pydicom
import matplotlib.pyplot as plt
from pathlib import Path

polku = Path(r"C:\Users\User01\GRADU\Aineisto\VN0ds\Patient2_VN0\struct\RS.1.2.246.352.221.5081513604484729159.13153969492833577387.dcm")
ds = pydicom.dcmread(polku)
print("Onko ROIContourSequence:", hasattr(ds, "ROIContourSequence"))

ds = pydicom.dcmread(polku, stop_before_pixels=True)
print(ds.Modality)

rt_ds = pydicom.dcmread(polku)

for i, roi_contour in enumerate(rt_ds.ROIContourSequence):
    name = rt_ds.StructureSetROISequence[i].ROIName
    has_contours = hasattr(roi_contour, "ContourSequence")
    print(f"ROI: {name} | ContourSequence: {has_contours}")
