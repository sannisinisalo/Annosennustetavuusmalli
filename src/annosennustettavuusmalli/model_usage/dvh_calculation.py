# -*- coding: utf-8 -*-
"""
Luotu Ti 07.04.2026 
Tekijä: Sanni Sinisalo

Koodi DVH:n laskemiseen testipotilaille.
"""

import torch
import matplotlib.pyplot as plt
from pathlib import Path
import csv
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from annosennustettavuusmalli.utils.calculate_dvhs import calculate_dvhs
from annosennustettavuusmalli.utils.plot_dvh_dict import plot_dvhs
from annosennustettavuusmalli.utils.calculate_dx import calculate_dx
from annosennustettavuusmalli.utils.calculate_vx import calculate_vx

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

# Dx- ja Vx-parametrit
dx_percentages = [98.5, 95, 90, 75, 50, 25, 10, 2]  
vx_thresholds = [35, 16, 8, 4]                      

# Vx kohdeorganit (vain tietyille)
vx_organs = ["Heart", "Ipsilateral lung", "Contralateral lung", "Contralateral breast"]


# CSV-tiedosto
csv_file = base_dir / "dose_metrics.csv"

with open(csv_file, mode="w", newline="") as f:
    writer = csv.writer(f)

    # Luo otsikot dynaamisesti
    header = ["Patient"]

    # Dx sarakkeet
    for organ in organ_config.keys():
        if organ == "PTV":  # Dx vain PTV:lle
            for d in dx_percentages:
                header.append(f"{organ}_D{int(d*10)/10}")  # esim. D98.5
    # Vx sarakkeet
    for organ in vx_organs:
        for v in vx_thresholds:
            header.append(f"{organ}_V{v}")

    # Sama kliininen ja ennuste
    header = [f"{name}_pred" if i != 0 else "Patient" for i, name in enumerate(header)]
    writer.writerow(header + [col.replace("_pred", "_clin") for col in header if col != "Patient"])

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

        # DVH (valinnainen tallennus)
        dvh_pred = calculate_dvhs(pred, mask, organ_config)
        dvh_clin = calculate_dvhs(clin, mask, organ_config)

        fig, ax = plt.subplots(figsize=(8,6))
        plot_dvhs(ax, dvh_clin, linestyle='-')  
        plot_dvhs(ax, dvh_pred, linestyle='--') 
        ax.legend()
        plt.title("Kliinnen vs. ennustettu DVH")
        plt.savefig(plot_dir / f"{patient_name}_DVH.png")
        plt.close()

        # Laske Dx ja Vx
        row_pred = []
        row_clin = []

        # Dx PTV
        for d in dx_percentages:
            row_pred.append(calculate_dx(pred, mask, {"PTV":1}, d).get("PTV", float("nan")))
            row_clin.append(calculate_dx(clin, mask, {"PTV":1}, d).get("PTV", float("nan")))

        # Vx muille organille
        for organ in vx_organs:
            label = organ_config[organ]
            for v in vx_thresholds:
                row_pred.append(calculate_vx(pred, mask, {organ: label}, v).get(organ, float("nan")))
                row_clin.append(calculate_vx(clin, mask, {organ: label}, v).get(organ, float("nan")))

        # Kirjoita CSV:hen
        writer.writerow([patient_name] + row_pred + row_clin)

print("All patients processed. CSV and DVH plots are saved.")

