# -*- coding: utf-8 -*-
"""
Luotu Ma 24.11.2025
Tekijä: Sanni Sinisalo

Koodi, joka listaa kaikki kyseisen potilaan ROI-alueet.
"""

from pathlib import Path

import pydicom

polku = Path(
    r"C:\Users\User01\GRADU\Aineisto\VN0ds\Patient47_VN0\struct\RS.1.2.246.352.221.5454258401965172541.996177027274946221.dcm"
)
ds = pydicom.dcmread(polku)
print("Onko ROIContourSequence:", hasattr(ds, "ROIContourSequence"))

ds = pydicom.dcmread(polku, stop_before_pixels=True)
print(ds.Modality)

rt_ds = pydicom.dcmread(polku)

for i, roi_contour in enumerate(rt_ds.ROIContourSequence):
    name = rt_ds.StructureSetROISequence[i].ROIName
    has_contours = hasattr(roi_contour, "ContourSequence")
    print(f"ROI: {name} | ContourSequence: {has_contours}")
