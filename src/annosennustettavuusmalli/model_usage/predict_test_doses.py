# -*- coding: utf-8 -*-
"""
Luotu Ke 01.04.2026
Tekijä: Sanni Sinisalo

Testisetin tallentamiseen 
"""

import torch
import numpy as np
import yaml
from pathlib import Path
import torchio as tio

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
    train_set, val_set, test_set = generate_datasets(
        config["data_paths"][DATA],
        reduce_samples=1
    )

    print("Test set size:", len(test_set))
    
    # Tulosta testipotilaiden nimet
    #print("\nTest patients:")
    #test_names = []
    #for subject in test_set:
    #    name = subject["name"]
    #    print(name)
    #    test_names.append(name)
    # Tallenna nimet tiedostoon
    #with open(save_dir / "test_patient_names.txt", "w") as f:
    #    for name in test_names:
    #        f.write(name + "\n")
    #print("\nTest patient names saved.")

    # Sama hyperparametrirakenne kuin trainingissa
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
    model_path = r"C:\Users\User01\GRADU\GitHub-koodit\Annosennustetavuusmalli\src\annosennustettavuusmalli\training\trained_models\gregarious-chimp-691_epoch_16.pth"

    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )

    model.eval()

    with torch.no_grad():

        for i, subject in enumerate(test_set):

            input_ct = subject["ct"][tio.DATA].float().unsqueeze(0)
            input_mask = subject["mask"][tio.DATA].float().unsqueeze(0)
            input_dist = subject["distance_to_PTV"][tio.DATA].float().unsqueeze(0)

            input_ = torch.cat(
                (input_ct, input_mask, input_dist),
                dim=1
            ).to(device)

            pred = model(input_)

            predicted_dose = (
                pred.main_output
                .squeeze()
                .cpu()
                .numpy()
            )
            
            patient_name = subject["name"]
            patient_dir = save_dir / patient_name
            patient_dir.mkdir(exist_ok=True)

            clinical_dose = (
                subject["dose"][tio.DATA]
                .squeeze()
                .numpy()
            )

            mask = (
                subject["mask"][tio.DATA]
                .squeeze()
                .numpy()
            )

            np.save(
                patient_dir / "pred.npy",
                predicted_dose
            )
            
            np.save(
                patient_dir / "clin.npy",
                clinical_dose
            )
            
            np.save(
                patient_dir / "mask.npy",
                mask
            )

            print(f"Saved patient {i}")

    print("All predictions saved.")

if __name__ == "__main__":
    main()