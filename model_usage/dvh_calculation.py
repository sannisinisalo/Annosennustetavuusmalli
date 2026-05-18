# -*- coding: utf-8 -*-
"""
Luotu Ti 07.04.2026
Tekijä: Sanni Sinisalo

Koodi DVH:n laskemiseen testipotilaille.
"""

import torch
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from luokat2 import DoseMetricsConfig, BASE_DIR
from src.annosennustettavuusmalli.utils.calculate_dvhs import calculate_dvhs
from src.annosennustettavuusmalli.utils.plot_dvh_dict import plot_dvhs

# Konfiguraatio
config = DoseMetricsConfig()

# Polut
base_dir = BASE_DIR  / "predicted_doses" 
plot_dir = base_dir / "dvh_plots"
plot_dir.mkdir(exist_ok=True)

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
    if patient_name == "Patient47_VN0":
        print(f"Skipping {patient_name} (right breast patient)")
        continue

    patient_name = patient_dir.name
    print("Processing:", patient_name)

    # Lataa data
    pred = torch.load(pred_file).squeeze()
    clin = torch.load(clin_file).squeeze()
    mask = torch.load(mask_file).squeeze()
    print("clin:", clin.shape)
    print("mask:", mask.shape)
    print("pred:", pred.shape)


    dvh_pred, dose_axis = calculate_dvhs(pred, mask, config.organ_config)
    dvh_clin, _ = calculate_dvhs(clin, mask, config.organ_config)

    fig, ax = plt.subplots(figsize=(8,6))
    plot_dvhs(ax, dvh_clin, linestyle='-')
    plot_dvhs(ax, dvh_pred, linestyle='--')
    ax.legend()
    plt.title(f"{patient_name}, kliininen vs. ennustettu DVH")
    plt.savefig(plot_dir / f"{patient_name}_DVH.png")
    plt.close()

print("DVH plots saved.")