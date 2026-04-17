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

sys.path.append(str(Path(__file__).resolve().parent.parent))

from luokat2 import DoseMetricsConfig, BASE_DIR
from src.annosennustettavuusmalli.utils.calculate_dx import calculate_dx
from src.annosennustettavuusmalli.utils.calculate_vx import calculate_vx



# Konfiguraatio

config = DoseMetricsConfig()

organs = [o for o in config.vx_organs if o != "PTV"]

# Polut
base_dir = BASE_DIR / "predicted_doses"
csv_file = base_dir / "dose_metrics.csv"

  
# CSV kirjoitus
with open(csv_file, mode="w", newline="") as f:

    writer = csv.writer(f)

    # Otsikot
    header = ["Patient"]
    

    # Dx otsikot kaikille elimille
    for organ in organs:
        for d in config.dx_percentages:
            header.append(f"{organ}_D{int(d)}")

    # Vx otsikot kaikille elimille
    for organ in organs:
        for v in config.vx_thresholds:
            header.append(f"{organ}_V{v}")

    header_pred = [f"{name}_pred" for name in header if name != "Patient"]
    header_clin = [col.replace("_pred", "_clin") for col in header_pred]

    writer.writerow(["Patient"] + header_pred + header_clin)

    # Loop potilaille
    for patient_dir in base_dir.iterdir():

        if not patient_dir.is_dir():
            continue

        pred_file = patient_dir / "pred.pt"
        clin_file = patient_dir / "clin_original.pt"
        mask_file = patient_dir / "mask_original.pt"

        if not (pred_file.exists() and clin_file.exists() and mask_file.exists()):
            print(f"Skipping {patient_dir.name}: required files not found.")
            continue

        patient_name = patient_dir.name
        if patient_name == "Patient47_VN0":
            print(f"Skipping {patient_name} (right breast patient)")
            continue

        print("Processing:", patient_name)

        # Lataa data
        pred = torch.load(pred_file).squeeze()
        clin = torch.load(clin_file).squeeze()
        mask = torch.load(mask_file).squeeze()
        
        row_pred = []
        row_clin = []

        # Laske Dx ja Vx kaikille elimille
        for organ in organs:
            label = config.organ_config[organ]
            
            # Dx
            for d in config.dx_percentages:

                dx_pred = calculate_dx(
                    pred,
                    mask,
                    {organ: label},
                    d
                ).get(organ, float("nan"))

                dx_clin = calculate_dx(
                    clin,
                    mask,
                    {organ: label},
                    d
                ).get(organ, float("nan"))

                row_pred.append(dx_pred)
                row_clin.append(dx_clin)

            # Vx
            for v in config.vx_thresholds:

                vx_pred = calculate_vx(
                    pred,
                    mask,
                    {organ: label},
                    v
                ).get(organ, float("nan"))

                vx_clin = calculate_vx(
                    clin,
                    mask,
                    {organ: label},
                    v
                ).get(organ, float("nan"))

                row_pred.append(vx_pred)
                row_clin.append(vx_clin)
        # Kirjoita CSV
        writer.writerow([patient_name] + row_pred + row_clin)

print("CSV saved with Dx and Vx metrics.")



