# -*- coding: utf-8 -*-
"""
Luotu Ke 01.04.2026
Tekijä: Sanni Sinisalo

Koodi, jossa käytetään mallia ennustamaan annosjakauma testipotilaille
"""

import sys
from pathlib import Path

import torch
import torchio as tio

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from annosennustettavuusmalli.config.config import load_config
from annosennustettavuusmalli.models.unet3plus_3d import UNet3plus_3d
from annosennustettavuusmalli.utils.generate_datasets import generate_datasets


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    BASE_DIR = Path(__file__).parent
    save_dir = BASE_DIR / "predicted_doses"
    save_dir.mkdir(exist_ok=True)

    config = load_config()

    DATA = "VN0_data"

    # Dataset
    _, _, test_set = generate_datasets(config["data_paths"][DATA], reduce_samples=1)

    print("Test set size:", len(test_set))

    # Mallin hyperparametrit
    hp_config_iter = config["manual_search"][0]

    model = UNet3plus_3d(
        in_channels=hp_config_iter["in_channels"],
        out_channels=hp_config_iter["out_channels"],
        filters=hp_config_iter["filters"],
        conv_layers=hp_config_iter["conv_layers"],
        kernel_size=hp_config_iter["kernel_size"],
        skip_filters=hp_config_iter["skip_filters"],
        pool_size=hp_config_iter["pool_size"],
        act_func=hp_config_iter["act_func"],
        patch_size=hp_config_iter["patch_size"],
    ).to(device)

    # Lataa malli
    model_path = (
        r"C:\Users\User01\GRADU\trained_models\gregarious-chimp-691_epoch_16.pth"
    )

    model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()

    patch_size = (256, 256, 1)

    with torch.no_grad():
        for subject in test_set:
            patient_name = subject["name"]

            print(f"\nProcessing {patient_name}")

            patient_dir = save_dir / patient_name
            patient_dir.mkdir(exist_ok=True)

            # Luo patch-sampler
            sampler = tio.GridSampler(subject, patch_size, patch_overlap=(64, 64, 0))

            patch_loader = torch.utils.data.DataLoader(sampler, batch_size=1)

            aggregator = tio.GridAggregator(sampler, overlap_mode="hann")

            for patches_batch in patch_loader:
                input_patch = torch.cat(
                    [
                        patches_batch["ct"][tio.DATA].float(),
                        patches_batch["mask"][tio.DATA].float(),
                        patches_batch["distance_to_PTV"][tio.DATA].float(),
                    ],
                    dim=1,
                ).to(device)

                pred_patch = model(input_patch).main_output

                # Lisää reconstructioniin
                aggregator.add_batch(pred_patch.cpu(), patches_batch[tio.LOCATION])

            # Tässä muodostuu koko annos
            pred_full = aggregator.get_output_tensor()

            # Tallennus
            torch.save(pred_full, patient_dir / "pred.pt")

            torch.save(
                subject["dose"][tio.DATA].squeeze().float(), patient_dir / "clin.pt"
            )

            torch.save(
                subject["mask"][tio.DATA].squeeze().int(), patient_dir / "mask.pt"
            )

            print(f"Saved {patient_name}")

    print("\nAll predictions saved.")


if __name__ == "__main__":
    main()
