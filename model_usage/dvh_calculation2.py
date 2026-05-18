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

        pred = torch.load(patient_dir / "pred.pt").squeeze().numpy()
        clin = torch.load(patient_dir / "clin.pt").squeeze().numpy()
        mask = torch.load(patient_dir / "mask.pt").squeeze().numpy()

        patient_out = dvh_out 
        patient_out.mkdir(exist_ok=True)

        
        
        plt.figure(figsize=(7, 6))
        
        # Käyttäjän määrittelemät värit
        colors = {
            "PTV": "red",
            "Heart": "blue",
            "Ipsilateral lung": "green",
            "Contralateral lung": "orange",
            "Contralateral breast": "purple",
        }
        
        legend_handles = []
        legend_labels = []
        
        for organ_name, organ_idx in organ_map.items():
        
            dose_bins_pred, vol_pred = compute_dvh(pred, mask, organ_idx)
            dose_bins_clin, vol_clin = compute_dvh(clin, mask, organ_idx)
        
            if dose_bins_pred is None:
                print(f"  - Ei pikseleitä rakenteelle {organ_name}, ohitetaan.")
                continue
            
            #pred_vals = pred[mask == organ_idx]
            #print(organ_name, pred_vals.min(), pred_vals.max(), pred_vals.mean())
            
            color = colors.get(organ_name, "black")  # fallback mustalle
        
            # Kliininen DVH: yhtenäinen viiva
            line1, = plt.plot(
                dose_bins_clin,
                vol_clin,
                color=color,
                linewidth=2,
            )
            
            # Ennustettu DVH
            line2, = plt.plot(
                dose_bins_pred,
                vol_pred,
                color=color,
                linestyle="--",
                linewidth=2,
            )
            
            # Lisää legendaan (vain kerran per organ)
            legend_handles.extend([line1, line2])
            legend_labels.extend([f"{organ_name} – Clinical", f"{organ_name} – Predicted"])
        
        
        # Tallenna kuva
        plt.xlabel("Dose (Gy)", fontsize=16)
        plt.ylabel("Volume fraction", fontsize=16)
        plt.title(f"DVH – {patient_name}", fontsize=18)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.grid(True)
        #plt.legend(fontsize=8, ncol=2)
        plt.tight_layout()
        plt.savefig(patient_out / f"{patient_name}_DVH.png")
        plt.close()
    
    # Luo erillinen legendakuva
    fig_legend = plt.figure(figsize=(8, 4))
    
    # Jaa kolmeen sarakkeeseen
    handles_col1 = legend_handles[0:4]
    labels_col1  = legend_labels[0:4]
    
    handles_col2 = legend_handles[4:8]
    labels_col2  = legend_labels[4:8]
    
    handles_col3 = legend_handles[8:10]
    labels_col3  = legend_labels[8:10]
    
    # Sarake 1
    fig_legend.legend(
        handles_col1,
        labels_col1,
        loc="center left",
        bbox_to_anchor=(0.0, 0.5),
        fontsize=10,
        frameon=False,
    )
    
    # Sarake 2
    fig_legend.legend(
        handles_col2,
        labels_col2,
        loc="center",
        bbox_to_anchor=(0.4, 0.5),
        fontsize=10,
        frameon=False,
    )
    
    # Sarake 3
    fig_legend.legend(
        handles_col3,
        labels_col3,
        loc="center right",
        bbox_to_anchor=(0.935, 0.55),
        fontsize=10,
        frameon=False,
    )
    
    # Poista akselit
    plt.axis("off")
    
    # Tallenna
    fig_legend.savefig(dvh_out / "DVH_legend2.png", bbox_inches="tight")
    plt.close(fig_legend)


    print("\nDVH-laskenta valmis.")


if __name__ == "__main__":
    main()
