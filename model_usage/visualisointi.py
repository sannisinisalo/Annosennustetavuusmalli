# -*- coding: utf-8 -*-
"""
Luotu Ke 22.04.2026
Tekijä: Sanni Sinisalo

Koodi, jolla visualisoidaan kliinistä ja ennustettua annosjakaumaa CT-leikkeen
päällä
"""

from luokat2 import BASE_DIR
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import pydicom


def load_dicom_series(dicom_dir):
    dicom_files = list(Path(dicom_dir).glob("*.dcm"))

    slices = []

    for f in dicom_files:
        ds = pydicom.dcmread(f)
        slices.append(ds)

    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

    images = []

    for s in slices:
        img = s.pixel_array.astype(np.float32)

        slope = float(s.RescaleSlope)
        intercept = float(s.RescaleIntercept)

        img = img * slope + intercept
        images.append(img)

    volume = np.stack(images, axis=-1)

    return volume


def normalize_ct(ct):
    ct = np.clip(ct, -1000, 1000)
    ct = (ct + 1000) / 2000
    return ct


def load_dose_tensors(patient_dir):
    pred = torch.load(patient_dir / "pred.pt")
    clin = torch.load(patient_dir / "clin.pt")
    
    pred = pred.squeeze().numpy()
    clin = clin.squeeze().numpy()

    return pred, clin


def fix_dose_orientation(slice_2d):
    return np.fliplr(
        np.rot90(slice_2d, k=-1)
    )


def plot_dose_comparison(
    ct,
    clin,
    pred,
    slice_idx=None,
    save_path="dose_comparison.png",
):

    if slice_idx is None:
        slice_idx = ct.shape[2] // 2
        
    ct_slice = ct[:, :, slice_idx]
    clin_slice = fix_dose_orientation(clin[:, :, slice_idx])
    pred_slice = fix_dose_orientation(pred[:, :, slice_idx])

    fig, axes = plt.subplots(1, 2, figsize=(18, 6), layout="compressed")

    vmax = max(clin_slice.max(), pred_slice.max())

    # Clinical
    axes[0].imshow(ct_slice, cmap="gray")  

    im1 = axes[0].imshow(
        clin_slice,
        cmap="inferno",
        alpha=0.6,
        vmin=0,
        vmax=vmax,
    )

    axes[0].set_title("Kliininen annosjakauma")
    axes[0].axis("off")

    # Predicted
    axes[1].imshow(ct_slice, cmap="gray")

    axes[1].imshow(
        pred_slice,
        cmap="inferno",
        alpha=0.6,
        vmin=0,
        vmax=vmax,
    )

    axes[1].set_title("Ennustettu annosjakauma")
    axes[1].axis("off")

    fig.colorbar(
        im1, ax=axes, location="right", fraction=0.03, pad=0.02, label="Annos (Gy)"
    )

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def main():
    patient_name = "Patient8_VN0"
    
    base_dir = BASE_DIR / "predicted_doses" / patient_name

    pred, clin = load_dose_tensors(base_dir)

    dicom_dir = BASE_DIR / "VN0ds" / patient_name / "ct"

    ct = load_dicom_series(dicom_dir)

    ct = normalize_ct(ct)

    plot_dose_comparison(
        ct,
        clin,
        pred,
        slice_idx=None,
        save_path=base_dir / "dose_comparison.png",
    )


if __name__ == "__main__":
    main()