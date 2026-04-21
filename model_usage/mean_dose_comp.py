# -*- coding: utf-8 -*-
"""
Luotu Ti 21.4.2026
Tekijä: Sanni Sinisalo

"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import re

from luokat2 import BASE_DIR, DoseMetricsConfig


def is_patient_folder(folder_name):
    return re.match(r"^Patient\d+_VN0$", folder_name) is not None


def compute_mean_dose(dose_array, mask_array, structure_index):
    voxels = dose_array[mask_array == structure_index]

    if len(voxels) == 0:
        return None

    return float(voxels.mean())


def create_plot():

    cfg = DoseMetricsConfig()

    organ_map = cfg.organ_config

    pred_dir = BASE_DIR / "predicted_doses"

    patients = [
        p for p in pred_dir.iterdir()
        if p.is_dir() and is_patient_folder(p.name)
    ]

    print(f"Löydetty {len(patients)} potilasta.")

    # Collect data
    clinical_means = {
        organ: []
        for organ in organ_map
        if organ != "PTV"
    }

    predicted_means = {
        organ: []
        for organ in organ_map
        if organ != "PTV"
    }

    for patient_dir in patients:

        pred = torch.load(
            patient_dir / "pred.pt"
        ).squeeze().numpy()

        clin = torch.load(
            patient_dir / "clin.pt"
        ).squeeze().numpy()

        mask = torch.load(
            patient_dir / "mask.pt"
        ).squeeze().numpy()

        for organ_name, organ_idx in organ_map.items():

            if organ_name == "PTV":
                continue

            clin_mean = compute_mean_dose(
                clin,
                mask,
                organ_idx
            )

            pred_mean = compute_mean_dose(
                pred,
                mask,
                organ_idx
            )

            if clin_mean is None:
                continue

            clinical_means[organ_name].append(
                clin_mean
            )

            predicted_means[organ_name].append(
                pred_mean
            )

    # Plotting
    organs = list(clinical_means.keys())

    n_organs = len(organs)

    fig, axes = plt.subplots(
        1,
        n_organs,
        figsize=(4 * n_organs, 6)
    )

    if n_organs == 1:
        axes = [axes]

    # Example dose objectives (modify if needed)
    organ_objectives = {
        "Heart": 2.0,
        "Ipsilateral lung": 7.5,
        "Contralateral lung": 2.0,
        "Contralateral breast": 2.0,
    }

    for i, organ in enumerate(organs):

        ax = axes[i]

        clin_vals = np.array(
            clinical_means[organ]
        )

        pred_vals = np.array(
            predicted_means[organ]
        )

        n_patients = len(clin_vals)

        # Individual lines
        for p in range(n_patients):

            c = clin_vals[p]
            pr = pred_vals[p]

            if c > pr:
                color = "orange"
            else:
                color = "blue"

            ax.plot(
                [0, 1],
                [c, pr],
                color=color,
                alpha=0.7,
                linewidth=1.5
            )

        # Objective line
        if organ in organ_objectives:
            ax.axhline(
                organ_objectives[organ],
                linestyle="--",
                color="black",
                linewidth=2
            )

        # Axis formatting
        ax.set_xticks([0, 1])
        ax.set_xticklabels(
            ["Clinical", "Prediction"]
        )
        ax.set_title(organ)
        if i == 0:
            ax.set_ylabel("Dose (Gy)")

    plt.tight_layout()

    save_path = (
        pred_dir / "OAR_mean_doses.png"
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print(f"Kuva tallennettu:\n{save_path}")


# RUN
if __name__ == "__main__":

    create_plot()