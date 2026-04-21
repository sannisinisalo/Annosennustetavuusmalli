# -*- coding: utf-8 -*-
"""
Luotu Ti 21.4.2026
Tekijä: Sanni Sinisalo

Dx ja Vx metriikoiden laskeminen DVH:sta ilman Akselin koodeja 
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from luokat2 import BASE_DIR, DoseMetricsConfig
import re


def is_patient_folder(folder_name):
    """Tunnistaa potilaskansiot."""
    return re.match(r"^Patient\d+_VN0$", folder_name) is not None


def compute_dx(dose_values, percent):
    """
    Laskee Dx-arvon: annos, jonka vähintään x% rakenteen tilavuudesta saa.
    """
    if len(dose_values) == 0:
        return np.nan

    sorted_vals = np.sort(dose_values)[::-1]  # suurimmasta pienimpään
    idx = int(len(sorted_vals) * (percent / 100.0))

    idx = min(max(idx, 0), len(sorted_vals) - 1)
    return sorted_vals[idx]


def compute_vx(dose_values, threshold):
    """
    Laskee Vx-arvon: prosenttiosuus rakenteen tilavuudesta,
    joka saa vähintään 'threshold' Gy annoksen.
    """
    if len(dose_values) == 0:
        return np.nan

    return np.sum(dose_values >= threshold) / len(dose_values) * 100.0


def main():

    cfg = DoseMetricsConfig()
    organ_map = cfg.organ_config
    dx_list = cfg.dx_percentages
    vx_list = cfg.vx_thresholds
    vx_organs = cfg.vx_organs

    pred_dir = BASE_DIR / "predicted_doses"
    out_dir = BASE_DIR / "DoseMetrics"
    out_dir.mkdir(exist_ok=True)

    patients = [
        p for p in pred_dir.iterdir()
        if p.is_dir() and is_patient_folder(p.name)
    ]

    print(f"Lasketaan metriikat {len(patients)} potilaalle.")

    for patient_dir in patients:
        patient_name = patient_dir.name
        print(f"\nPotilas: {patient_name}")

        pred = torch.load(patient_dir / "pred.pt").squeeze().numpy()
        clin = torch.load(patient_dir / "clin.pt").squeeze().numpy()
        mask = torch.load(patient_dir / "mask.pt").squeeze().numpy()

        rows = []

        for organ_name, organ_idx in organ_map.items():

            organ_mask = (mask == organ_idx)
            pred_vals = pred[organ_mask]
            clin_vals = clin[organ_mask]

            if len(pred_vals) == 0:
                print(f"  - Ei pikseleitä rakenteelle {organ_name}, ohitetaan.")
                continue

            # Laske Dx-arvot
            for dx in dx_list:
                rows.append({
                    "Organ": organ_name,
                    "Metric": f"D{dx}",
                    "Pred": compute_dx(pred_vals, dx),
                    "Clin": compute_dx(clin_vals, dx),
                })

            # Laske Vx-arvot vain määritetyille elimille
            if organ_name in vx_organs:
                for vx in vx_list:
                    rows.append({
                        "Organ": organ_name,
                        "Metric": f"V{vx}",
                        "Pred": compute_vx(pred_vals, vx),
                        "Clin": compute_vx(clin_vals, vx),
                    })

        # Tallenna CSV
        df = pd.DataFrame(rows)
        df.to_csv(out_dir / f"{patient_name}_metrics.csv", sep=";", index=False)

        print(f"  -> Tallennettu metriikat: {patient_name}_metrics.csv")

    print("\nMetriikoiden laskenta valmis.")


if __name__ == "__main__":
    main()
