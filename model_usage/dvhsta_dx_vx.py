# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 08:29:45 2026

@author: User01
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from luokat2 import DoseMetricsConfig, BASE_DIR
from src.annosennustettavuusmalli.utils.calculate_dvhs import calculate_dvhs


# DVH METRIKAT
def compute_dvh_metrics(dvh, dose_axis, config, organ):
    dvh = np.asarray(dvh)
    dose_axis = np.asarray(dose_axis)

    metrics = {}

    # Dx arvot
    for dx in config.dx_percentages:
        idx = np.where(dvh <= dx)[0]

        if len(idx) > 0:
            dose_value = dose_axis[idx[0]]
        else:
            dose_value = dose_axis[-1]

        metrics[f"D{dx}"] = dose_value


    return metrics


# CONFIG + PATHS
config = DoseMetricsConfig()

base_dir = BASE_DIR /"predicted_doses"
output_dir = base_dir 
output_dir.mkdir(exist_ok=True)

rows = []


# MAIN LOOP
for patient_dir in base_dir.iterdir():

    if not patient_dir.is_dir():
        continue

    pred_file = patient_dir / "pred.pt"
    clin_file = patient_dir / "clin.pt"
    mask_file = patient_dir / "mask.pt"

    if not (pred_file.exists() and clin_file.exists() and mask_file.exists()):
        print(f"Skipping {patient_dir.name}: missing files")
        continue

    patient_name = patient_dir.name
    print(f"Processing {patient_name}")

    pred = torch.load(pred_file).squeeze()
    clin = torch.load(clin_file).squeeze()
    mask = torch.load(mask_file).squeeze()

    dvh_pred, dose_axis = calculate_dvhs(
        pred,
        mask,
        config.organ_config
    )

    dvh_clin, _ = calculate_dvhs(
        clin,
        mask,
        config.organ_config
    )

    dose_axis = dose_axis.numpy()

    # yksi rivi per potilas
    row = {
        "patient": patient_name
    }

    # LOOP ORGANS
    for organ in dvh_pred.keys():

        pred_metrics = compute_dvh_metrics(
            dvh_pred[organ],
            dose_axis,
            config,
            organ
        )

        clin_metrics = compute_dvh_metrics(
            dvh_clin[organ],
            dose_axis,
            config,
            organ
        )


        for metric_name, value in pred_metrics.items():

            col_name = f"{organ}_{metric_name}_pred"
            row[col_name] = float(value)

        for metric_name, value in clin_metrics.items():

            col_name = f"{organ}_{metric_name}_clin"
            row[col_name] = float(value)

    rows.append(row)


# SAVE CSV
df = pd.DataFrame(rows)

ordered_cols = ["patient"]

for organ in config.organ_config.keys():

    # Dx kaikille elimille
    for dx in config.dx_percentages:

        ordered_cols.append(f"{organ}_D{dx}_pred")
        ordered_cols.append(f"{organ}_D{dx}_clin")


df = df[ordered_cols]

out_file = output_dir / "dvh_metrics.csv"

df.to_csv(out_file, index=False)

print(f"\nDONE. Saved to: {out_file}")