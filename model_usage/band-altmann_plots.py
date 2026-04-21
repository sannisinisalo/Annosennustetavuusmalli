# -*- coding: utf-8 -*-
"""
Luotu 16.04.2026
Tekijä: Sanni Sinisalo

Koodi Band-Altmann-plottien tekemiseen
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from luokat2 import DoseMetricsConfig, BASE_DIR

def compute_patient_means(pred, clin, mask, organ_config):
    means_clin = {}
    means_pred = {}
    for organ, organ_id in organ_config.items():
        vox = (mask == organ_id)
        means_clin[organ] = clin[vox].mean().item()
        means_pred[organ] = pred[vox].mean().item()
    return means_clin, means_pred


def bland_altman_single_plot(clin_list, pred_list, organ_name, save_path):
    clin = np.array(clin_list)
    pred = np.array(pred_list)

    mean_vals = (clin + pred) / 2
    diff = pred - clin
    md = diff.mean()
    sd = diff.std()
    upper = md + 1.96 * sd
    lower = md - 1.96 * sd

    plt.figure(figsize=(6,4))
    plt.scatter(mean_vals, diff, alpha=0.6)
    plt.axhline(md, color='orange')
    plt.axhline(upper, color='orange', linestyle='--')
    plt.axhline(lower, color='orange', linestyle='--')

    margin = 0.4 * (upper - lower)   
    plt.ylim(lower - margin, upper + margin)
    
    xmax = np.max(mean_vals)
    plt.xlim(0, xmax * 1.40) 

    plt.title(f"{organ_name} Bland–Altman")
    plt.xlabel("Mean dose (Gy)")
    plt.ylabel("Difference (Gy)")
    plt.grid(alpha=0.3)

    plt.savefig(save_path)
    plt.close()
   
    
if __name__ == "__main__":  
    cfg = DoseMetricsConfig()
    organ_config = cfg.organ_config
    
    clinical_means_all = { organ: [] for organ in organ_config.keys() }
    predicted_means_all = { organ: [] for organ in organ_config.keys() }
    
    base_dir = BASE_DIR / "predicted_doses"
    out_dir = base_dir / "bland_altman_plots"
    out_dir.mkdir(exist_ok=True)   
     
    for patient_dir in base_dir.iterdir():
        if not patient_dir.is_dir():
            continue
    
        pred_file = patient_dir / "pred.pt"
        clin_file = patient_dir / "clin.pt"
        mask_file = patient_dir / "mask.pt"
    
        if not (pred_file.exists() and clin_file.exists() and mask_file.exists()):
            continue
    
        pred = torch.load(pred_file).squeeze()
        clin = torch.load(clin_file).squeeze()
        mask = torch.load(mask_file).squeeze()
    
        means_clin, means_pred = compute_patient_means(pred, clin, mask, organ_config)
    
        for organ in organ_config.keys():
            clinical_means_all[organ].append(means_clin[organ])
            predicted_means_all[organ].append(means_pred[organ])
    
    for organ in organ_config.keys():
        if organ == "PTV":
            continue
    
        save_path = out_dir / f"{organ.replace(' ', '_')}_bland_altman.png"
        bland_altman_single_plot(
            clinical_means_all[organ],
            predicted_means_all[organ],
            organ,
            save_path
        )
    
    print("Bland–Altman plots created.")
