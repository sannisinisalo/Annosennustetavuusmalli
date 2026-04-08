# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 08:16:14 2026

@author: User01
"""

import torch
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


PREDICTED_DIR = Path("predicted_doses")
SLICE_INDEX = 80  


def check_patient(patient_dir):

    print("\n==============================")
    print("Patient:", patient_dir.name)

    pred_file = patient_dir / "pred.pt"
    clin_file = patient_dir / "clin.pt"
    mask_file = patient_dir / "mask.pt"

    # Tiedostojen tarkistus 

    if not pred_file.exists():
        print("pred.pt puuttuu")
        return

    if not clin_file.exists():
        print("clin.pt puuttuu")
        return

    if not mask_file.exists():
        print("mask.pt puuttuu")
        return

    print("Kaikki tiedostot löytyvät")

    # Lataa tensorit 

    pred = torch.load(pred_file)
    clin = torch.load(clin_file)
    mask = torch.load(mask_file)

    # Jos pred sisältää kanavan (1, Z, Y, X)
    if pred.ndim == 4:
        pred = pred.squeeze(0)

    # Shape tarkistus

    print("\nShapes:")

    print("Pred:", pred.shape)
    print("Clin:", clin.shape)
    print("Mask:", mask.shape)

    if pred.shape != clin.shape:
        print("ERROR: Pred ja Clin eri kokoiset!")

    if mask.shape != clin.shape:
        print("ERROR: Mask ja Clin eri kokoiset!")

    else:
        print("Shapes OK")

    # Arvojen tarkistus

    print("\nValue ranges:")

    print("Pred min:", float(pred.min()))
    print("Pred max:", float(pred.max()))

    print("Clin min:", float(clin.min()))
    print("Clin max:", float(clin.max()))

    print("Mask min:", float(mask.min()))
    print("Mask max:", float(mask.max()))

    if float(pred.max()) == 0:
        print("WARNING: Pred näyttää olevan pelkkää nollaa!")

    # MAE laskenta 

    mae = torch.mean(torch.abs(pred - clin))

    print("\nMAE:", float(mae))

    if mae > 30:
        print("WARNING: MAE on suuri")

    else:
        print("MAE näyttää järkevältä")

    # Slice visualisointi 

    slice_idx = min(SLICE_INDEX, pred.shape[2] - 1)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(
        np.fliplr(
            np.rot90(clin[:, :, slice_idx], k=-1)
        )
    )
    plt.title("Clinical dose")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(
        np.fliplr(
            np.rot90(pred[:, :, slice_idx], k=-1)
        )
    )
    plt.title("Predicted dose")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(
        np.fliplr(
            np.rot90(mask[:, :, slice_idx], k=-1)
        )
    )
    plt.title("Mask")
    plt.axis('off')

    plt.suptitle(patient_dir.name)
    plt.show()



def main():

    if not PREDICTED_DIR.exists():
        print("predicted_doses kansiota ei löydy")
        return

    patient_dirs = [
        p for p in PREDICTED_DIR.iterdir()
        if p.is_dir()
    ]

    if len(patient_dirs) == 0:
        print("Ei potilaskansioita löytynyt")
        return

    print("Potilaita löytyi:", len(patient_dirs))

    # Käydään kaikki potilaat läpi
    for patient_dir in patient_dirs:

        check_patient(patient_dir)

    print("\n Tarkistus valmis.")


if __name__ == "__main__":
    main()