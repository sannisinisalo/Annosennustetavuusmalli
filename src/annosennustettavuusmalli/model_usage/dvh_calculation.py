# -*- coding: utf-8 -*-
"""
Luotu Ti 07.04.2026 
Tekijä: Sanni Sinisalo

Koodi DVH:n laskemiseen testipotilaille.
"""
import torch
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from annosennustettavuusmalli.utils.calculate_dvhs import calculate_dvhs
from annosennustettavuusmalli.utils.plot_dvh_dict import plot_dvhs

patient_dir = "predicted_doses/Patient3_VN0"

pred = torch.load(f"{patient_dir}/pred.pt").squeeze()
clin = torch.load(f"{patient_dir}/clin.pt").squeeze()
mask = torch.load(f"{patient_dir}/mask.pt").squeeze()

organ_config = {
    "PTV": 1,
    "Heart": 2,
    "Ipsilateral lung": 3,
}

# DVH:t
dvh_pred = calculate_dvhs(pred, mask, organ_config)
dvh_clin = calculate_dvhs(clin, mask, organ_config)

# Plot
fig, ax = plt.subplots(figsize=(8,6))

plot_dvhs(ax, dvh_clin)
plot_dvhs(ax, dvh_pred)

ax.legend()

plt.title("Clinical vs Predicted DVH")

plt.show()

