# -*- coding: utf-8 -*-
"""
Luotu Ti 21.4.2026
Tekijä: Sanni Sinisalo

DVH-käyrien tekeminen ilman Akselin funktioita
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from luokat2 import BASE_DIR, DoseMetricsConfig
import re


def compute_dvh(dose_array, mask_array, structure_index, num_bins=2000):
    """Laskee DVH:n annetulle rakenteelle."""
    structure_voxels = dose_array[mask_array == structure_index]

    if len(structure_voxels) == 0:
        return None, None

    max_dose = float(structure_voxels.max())
    bins = np.linspace(0, max_dose, num_bins)

    hist, bin_edges = np.histogram(structure_voxels, bins=bins)

    cum = np.cumsum(hist[::-1])[::-1]
    volume_fraction = cum / cum[0]

    return bin_edges[:-1], volume_fraction


def is_patient_folder(folder_name):
    """Tunnistaa potilaskansiot."""
    return re.match(r"^Patient\d+_VN0$", folder_name) is not None


def main():

    cfg = DoseMetricsConfig()
    organ_map = cfg.organ_config

    pred_dir = BASE_DIR / "predicted_doses"
    dvh_out = pred_dir / "DVH_results"
    dvh_out.mkdir(exist_ok=True)

    patients = [
        p for p in pred_dir.iterdir()
        if p.is_dir() and is_patient_folder(p.name)
    ]

    print(f"Lasketaan DVH:t {len(patients)} potilaalle.")

    for patient_dir in patients:
        patient_name = patient_dir.name
        print(f"\nPotilas: {patient_name}")

        pred = torch.load(patient_dir / "pred.pt").squeeze().numpy()
        clin = torch.load(patient_dir / "clin.pt").squeeze().numpy()
        mask = torch.load(patient_dir / "mask.pt").squeeze().numpy()

        patient_out = dvh_out 
        patient_out.mkdir(exist_ok=True)

        
        
        plt.figure(figsize=(8, 6))
        
        # Käyttäjän määrittelemät värit
        colors = {
            "PTV": "red",
            "Heart": "blue",
            "Ipsilateral lung": "green",
            "Contralateral lung": "orange",
            "Contralateral breast": "purple",
        }
        
        for organ_name, organ_idx in organ_map.items():
        
            dose_bins_pred, vol_pred = compute_dvh(pred, mask, organ_idx)
            dose_bins_clin, vol_clin = compute_dvh(clin, mask, organ_idx)
        
            if dose_bins_pred is None:
                print(f"  - Ei pikseleitä rakenteelle {organ_name}, ohitetaan.")
                continue
            
            pred_vals = pred[mask == organ_idx]
            print(organ_name, pred_vals.min(), pred_vals.max(), pred_vals.mean())
            
            color = colors.get(organ_name, "black")  # fallback mustalle
        
            # Kliininen DVH: yhtenäinen viiva
            plt.plot(
                dose_bins_clin,
                vol_clin,
                label=f"{organ_name} – Clin",
                color=color,
                linewidth=2,
            )
        
            # Ennustettu DVH: katkoviiva
            plt.plot(
                dose_bins_pred,
                vol_pred,
                label=f"{organ_name} – Pred",
                color=color,
                linestyle="--",
                linewidth=2,
            )
        
        
        # Tallenna kuva
        plt.title(f"DVH – {patient_name}")
        plt.xlabel("Dose (Gy)")
        plt.ylabel("Volume fraction")
        plt.grid(True)
        plt.legend(fontsize=8, ncol=2)
        plt.tight_layout()
        plt.savefig(patient_out / f"{patient_name}_DVH.png")
        plt.close()



    print("\nDVH-laskenta valmis.")


if __name__ == "__main__":
    main()
