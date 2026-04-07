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

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from annosennustettavuusmalli.utils.calculate_dvhs import calculate_dvhs
from annosennustettavuusmalli.utils.plot_dvh_dict import plot_dvhs

# Polut
base_dir = Path("predicted_doses")
plot_dir = base_dir / "dvh_plots"
plot_dir.mkdir(exist_ok=True)

# Organit ja tunnukset
organ_config = {
    "PTV": 1,
    "Heart": 2,
    "Contralateral lung": 3,
    "Ipsilateral lung": 4,
    "Contralateral breast": 5,
}

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

    # DVH
    dvh_pred = calculate_dvhs(pred, mask, organ_config)
    dvh_clin = calculate_dvhs(clin, mask, organ_config)

    fig, ax = plt.subplots(figsize=(8,6))
    plot_dvhs(ax, dvh_clin, linestyle='-')
    plot_dvhs(ax, dvh_pred, linestyle='--')
    ax.legend()
    plt.title("Kliininen vs. ennustettu DVH")
    plt.savefig(plot_dir / f"{patient_name}_DVH.png")
    plt.close()

print("DVH plots saved.")