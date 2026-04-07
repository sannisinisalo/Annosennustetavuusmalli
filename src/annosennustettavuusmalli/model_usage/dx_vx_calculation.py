# -*- coding: utf-8 -*-
"""
Luotu Ti 07.04.2026 
Tekijä: Sanni Sinisalo

Koodi Dx ja Vx tietojen laskemiseen testipotilaille.
"""

import torch
import csv
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from annosennustettavuusmalli.utils.calculate_dx import calculate_dx
from annosennustettavuusmalli.utils.calculate_vx import calculate_vx

# Polut
base_dir = Path("predicted_doses")
csv_file = base_dir / "dose_metrics.csv"

# Organit ja tunnukset
organ_config = {
    "PTV": 1,
    "Heart": 2,
    "Contralateral lung": 3,
    "Ipsilateral lung": 4,
    "Contralateral breast": 5,
}

# Dx- ja Vx-parametrit
dx_percentages = [98.5, 95, 90, 75, 50, 25, 10, 2]  
vx_thresholds = [35, 16, 8, 4]                      
vx_organs = ["Heart", "Ipsilateral lung", "Contralateral lung", "Contralateral breast"]

with open(csv_file, mode="w", newline="") as f:
    writer = csv.writer(f)

    # Luo otsikot
    header = ["Patient"]
    for d in dx_percentages:
        header.append(f"PTV_D{int(d*10)/10}")
    for organ in vx_organs:
        for v in vx_thresholds:
            header.append(f"{organ}_V{v}")

    header_pred = [f"{name}_pred" for name in header if name != "Patient"]
    header_clin = [col.replace("_pred","_clin") for col in header_pred]
    writer.writerow(["Patient"] + header_pred + header_clin)

    # Loop potilaille
    for patient_dir in base_dir.iterdir():
        if not patient_dir.is_dir():
            continue

        pred_file = patient_dir / "pred.pt"
        clin_file = patient_dir / "clin.pt"
        mask_file = patient_dir / "mask.pt"

        if not (pred_file.exists() and clin_file.exists() and mask_file.exists()):
            print(f"Skipping {patient_dir.name}: required files not found.")
            continue

        patient_name = patient_dir.name
        print("Processing:", patient_name)

        # Lataa data
        pred = torch.load(pred_file).squeeze()
        clin = torch.load(clin_file).squeeze()
        mask = torch.load(mask_file).squeeze()

        row_pred = []
        row_clin = []

        # Dx PTV
        for d in dx_percentages:
            row_pred.append(calculate_dx(pred, mask, {"PTV":1}, d).get("PTV", float("nan")))
            row_clin.append(calculate_dx(clin, mask, {"PTV":1}, d).get("PTV", float("nan")))

        # Vx muille organille
        for organ in vx_organs:
            label = organ_config[organ]
            for v in vx_thresholds:
                row_pred.append(calculate_vx(pred, mask, {organ: label}, v).get(organ, float("nan")))
                row_clin.append(calculate_vx(clin, mask, {organ: label}, v).get(organ, float("nan")))

        # Kirjoita CSV:hen
        writer.writerow([patient_name] + row_pred + row_clin)

print("CSV saved with Dx and Vx metrics.")