# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 13:01:31 2026

@author: User01
"""

import torch
import yaml
from pathlib import Path
import torchio as tio
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from annosennustettavuusmalli.models.unet3plus_3d import UNet3plus_3d
from annosennustettavuusmalli.utils.generate_datasets import generate_datasets

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BASE_DIR = Path(__file__).parent
    save_dir = BASE_DIR / "predicted_doses"
    save_dir.mkdir(exist_ok=True)

    config_file = BASE_DIR.parent / "config.yaml"
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    DATA = "VN0_data"

    # Luo datasetit
    _, _, test_set = generate_datasets(
        config["data_paths"][DATA],
        reduce_samples=1
    )
    print("Test set size:", len(test_set))

    # Hyperparametrit mallille
    hp_config_iter = config["manual_search"][0]

    # Luo malli
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

    # Lataa koulutettu malli
    model_path = r"C:\Users\User01\GRADU\trained_models\gregarious-chimp-691_epoch_16.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Patchien koko (muutettavissa, pienempi = vähemmän muistia)
    patch_size = (64, 64, 32)

    with torch.no_grad():
        for subject in test_set:
            patient_name = subject["name"]
            patient_dir = save_dir / patient_name
            patient_dir.mkdir(exist_ok=True)

            # Grid-sampler jakaa potilaan patchiin
            sampler = tio.sampler.GridSampler(subject, patch_size)
            patch_loader = torch.utils.data.DataLoader(sampler, batch_size=1)

            print(f"Predicting patches for {patient_name}... Total patches: {len(patch_loader)}")

            for j, patch in enumerate(patch_loader):
                
                # Lataa vain patch float32 ja GPU:lle
                input_patch = torch.cat([
                    patch['ct'][tio.DATA].float(),
                    patch['mask'][tio.DATA].float(),
                    patch['distance_to_PTV'][tio.DATA].float()
                ], dim=1).to(device)

                # Ennusta patch
                pred_patch = model(input_patch).main_output.squeeze().cpu()
                
                bb = patch['ct'][tio.LOCATION]
                torch.save(
                    {
                        "pred": pred_patch,
                        "location": bb
                    },
                    patient_dir / f"patch_{j}.pt"
                )

                # Tallenna patch suoraan levyyn
                torch.save(pred_patch, patient_dir / f"patch_{j}.pt")

            # Tallennetaan alkuperäinen annos ja maski tarvittaessa
            torch.save(subject["dose"][tio.DATA].squeeze().float(), patient_dir / "clin.pt")
            torch.save(subject["mask"][tio.DATA].squeeze().int(), patient_dir / "mask.pt")

            print(f"Saved all patches for {patient_name}")

    print("All predictions saved.")

if __name__ == "__main__":
    main()