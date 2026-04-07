# -*- coding: utf-8 -*-
"""
Luotu Ma 26.1.2026
Tekijä: Akseli Leino
Muokkaaja: Sanni Sinisalo

Koodiin tehdyt muokkaukset:
    - Jupyter notebookille ominaiset osat muutettu Python-toimiviksi Spyderissä
    - Tiedostopolut muutettu toimiviksi
    - Lisätty if __name__ == "__main__" rakenne
    - Lopun visualisointi kommentoiti pois käytöstä toistaiseksi
"""

import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import mlflow
import torch
import torch.nn as nn
import torchio as tio

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from annosennustettavuusmalli.config.config import load_default_config
from annosennustettavuusmalli.models.unet3plus_3d import UNet3plus_3d
from annosennustettavuusmalli.utils.evaluate_dataset import evaluate_dataset
from annosennustettavuusmalli.utils.evaluate_dose_metrics import evaluate_dose_metrics
from annosennustettavuusmalli.utils.flatten_dict import flatten_dict
from annosennustettavuusmalli.utils.generate_datasets import generate_datasets
from annosennustettavuusmalli.utils.init_weights import init_weights_kaiming
from annosennustettavuusmalli.utils.random_sample_hyperparameters import (
    random_sample_hyperparameters,
)


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs("trained_models", exist_ok=True)

    HYPERPARAMETERS = "manual_search"  # default, manual_search or random_search
    DATA = "VN0_data"

    BASE_DIR = Path(__file__).parent

    config = load_default_config()

    if HYPERPARAMETERS == "default":
        hp_config = [config["default"]]
    elif HYPERPARAMETERS == "random_search":
        hp_config = [
            random_sample_hyperparameters(config["random_search"])
            for _ in range(config["random_search"]["random_samples"])
        ]
    elif HYPERPARAMETERS == "manual_search":
        hp_config = config["manual_search"]
    else:
        raise ValueError("Invalid HYPERPARAMETERS value in config.yaml")

    mlflow.set_experiment(hp_config[0]["experiment_name"])

    # Looping through all defined hyperparamter combinations
    for hp_config_iter in hp_config:
        dose_metrics_val = {}
        dose_metrics_test = {}

        # !!!NOTE!!! reduce_samples means that epoch samples is reduced by a factor of defined number. This means that full epoch is really
        # epochs*n. This can be set to 1 so epoch = full epoch. This can be used to increase number of checkpoints. Note also that .yaml
        # configurations are reduced epochs. So if reduce_epochs = 4, training for 10 full epochs is 40 epochs in yaml.
        train_set, val_set, test_set = generate_datasets(
            config["data_paths"][DATA], reduce_samples=1
        )

        # Probability map probabilities are defined in custom_transforms -> ProbabilityMapTransform

        training_sampler = tio.sampler.WeightedSampler(
            hp_config_iter["patch_size"], probability_map="probability_map"
        )
        train_queue = tio.Queue(
            subjects_dataset=train_set,
            max_length=config["train_loader_config"]["max_length"],
            samples_per_volume=config["train_loader_config"]["samples_per_volume"],
            sampler=training_sampler,
            num_workers=0,
            verbose=True,
        )
        train_loader = torch.utils.data.DataLoader(
            train_queue, batch_size=hp_config_iter["batch_size"], num_workers=0
        )  # num_workers must be 0. (due to TorchIO queue implementation(?))

        print("Train set length:", len(train_set))
        print("Validation set length:", len(val_set))
        print("Test set length:", len(test_set))

        random.seed()  # Seed was set when splitting sets. Without seed reset, the mlflow naming always starts from the same name.

        # Initialize model
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

        model.apply(init_weights_kaiming)

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        mlflow.log_metrics({"total_params": total_params})
        flattened_dict = flatten_dict(hp_config_iter)
        mlflow.log_params(
            {
                key: value
                for key, value in flattened_dict.items()
                if isinstance(value, (str, tuple, list, bool))
            }
        )
        # Log all numbers as metrics to enable visualizations. Bool is handled also here, because bool inherits int, which causes isinstance(True, int) be True.
        mlflow.log_metrics(
            {
                key: value
                for key, value in flattened_dict.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }
        )

        # Initialize optimizer
        dynamic_optimizer = getattr(torch.optim, hp_config_iter["optimizer"]["type"])
        optimizer = dynamic_optimizer(
            model.parameters(), **hp_config_iter["optimizer"]["params"]
        )

        # Initialize primary loss and secondary loss (metric) functions
        primary_loss = getattr(nn, hp_config_iter["primary_loss"])()
        secondary_loss = getattr(nn, hp_config_iter["secondary_loss"])()

        # Initialize CosineAnnealing with warm restarts and warmup -scheduler
        # warmup_period = int(len(train_loader)*hp_config_iter["warmup_period"]/hp_config_iter["gradient_accumulation"])
        # restart_period = int(len(train_loader)*hp_config_iter["warm_restart_every"]/hp_config_iter["gradient_accumulation"])

        # warmup_period = int(len(train_loader)*hp_config_iter["warmup_period"]/hp_config_iter["gradient_accumulation"])
        # restart_period = int(len(train_loader)*hp_config_iter["warm_restart_every"]/hp_config_iter["gradient_accumulation"])

        num_optimizer_steps = (
            len(train_loader) / hp_config_iter["gradient_accumulation"]
        )
        warmup_period = int(num_optimizer_steps * hp_config_iter["warmup_period"])
        restart_period = int(num_optimizer_steps * hp_config_iter["warm_restart_every"])
        lrs_warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.000000001, end_factor=1, total_iters=warmup_period
        )
        lrs_cos = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, restart_period, eta_min=0.00000000001
        )
        seq_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[lrs_warmup, lrs_cos], milestones=[warmup_period]
        )

        best_val_primary = 100000000000  # Arbitrarily large number that loss is (hopefully) never going to be exceed
        last_improved = 0

        """
        Training loop
        """

        start_epoch = 1

        checkpoint_path = (
            BASE_DIR / "trained_models" / "gregarious-chimp-691_epoch_16.pth"
        )

        if os.path.exists(checkpoint_path):
            print("Loading checkpoint:", checkpoint_path)
            model.load_state_dict(torch.load(checkpoint_path))
            start_epoch = 17

        for epoch in range(start_epoch, hp_config_iter["EPOCHS"] + 1):
            # This is for cosine annealing decay over time
            if (
                epoch > hp_config_iter["warmup_period"]
                and (epoch - hp_config_iter["warmup_period"] + 1)
                % hp_config_iter["warm_restart_every"]
                == 0
            ):
                for group in optimizer.param_groups:
                    group["lr"] *= 0.8
                    lrs_cos.base_lrs = [x * 0.8 for x in lrs_cos.base_lrs]

            epoch_losses = defaultdict(int)
            running_losses = defaultdict(int)
            last_loss = 0
            epoch_size = len(train_loader)
            optimizer.zero_grad()

            """
            Loop for one training epoch
            """
            for i, batch in enumerate(train_loader):
                batch_losses = defaultdict(int)

                # TorchIO uses double by default, so these must be cast to float.
                input_ct = batch["ct"][tio.DATA].float()
                input_mask = batch["mask"][tio.DATA].float()
                input_distancetoPTV = batch["distance_to_PTV"][tio.DATA].float()
                input_ = torch.cat((input_ct, input_mask, input_distancetoPTV), dim=1)
                true_train_dose = batch["dose"][tio.DATA].float()
                input_ = input_.to(device)
                true_train_dose = true_train_dose.to(device)
                model.train()
                pred_train_dose = model(input_)

                for j in range(len(pred_train_dose)):
                    loss_multiplier = (
                        1 if j == 0 else 0.25
                    )  # Weights for different deep supervision outputs
                    batch_loss = primary_loss(pred_train_dose[j], true_train_dose)
                    batch_losses["deepsup_loss"] += batch_loss * loss_multiplier

                batch_losses["primary_metric"] = primary_loss(
                    pred_train_dose.main_output, true_train_dose
                )
                batch_losses["secondary_metric"] = secondary_loss(
                    pred_train_dose.main_output, true_train_dose
                )

                accumulated_loss = (
                    batch_losses["deepsup_loss"]
                    / hp_config_iter["gradient_accumulation"]
                )
                accumulated_loss.backward()

                if (i + 1) % hp_config_iter["gradient_accumulation"] == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    seq_scheduler.step()

                running_losses["deepsup_loss"] += batch_losses["deepsup_loss"].item()
                running_losses["primary_metric"] += batch_losses[
                    "primary_metric"
                ].item()
                running_losses["secondary_metric"] += batch_losses[
                    "secondary_metric"
                ].item()

                # Print running loss every 10 batches (10*batch_size images)
                if i % 10 == 9 or i == epoch_size - 1:
                    print(
                        f"EPOCH {epoch} | "
                        f"sample {(i + 1) * hp_config_iter['batch_size']}/{
                            epoch_size * hp_config_iter['batch_size']
                        } | "
                        f"batch {i + 1}/{epoch_size} | "
                        f"deepsup_loss: {running_losses['deepsup_loss'] / i:.3f} | "
                        f"primary_metric {running_losses['primary_metric'] / i:.3f} | "
                        f"secondary_metric: {running_losses['secondary_metric'] / i:.3f} | "
                        f"LR: {seq_scheduler.get_last_lr()[0]:.6f}",
                        end="\r",
                    )

                """
                This is the end of one epoch
                """

            epoch_losses["val_primary_metric"], epoch_losses["val_secondary_metric"] = (
                evaluate_dataset(model, val_set, hp_config_iter, device)
            )

            # Saving best model
            if epoch_losses["val_primary_metric"] <= best_val_primary:
                best_val_primary = epoch_losses["val_primary_metric"]
                best_val_secondary = epoch_losses["val_secondary_metric"]
                best_epoch = epoch
                last_improved = 0
            else:
                last_improved += 1

            torch.save(
                model.state_dict(),
                f"trained_models/{mlflow.active_run().info.run_name}_epoch_{epoch}.pth",
            )

            # Logging training metrics. To sort models by val_mae, go to mlflow, select all runs in experiment -> compare -> select val_mae -> sort
            epoch_losses["train_deepsup_loss"] = running_losses["deepsup_loss"] / len(
                train_loader
            )
            epoch_losses["train_primary_metric"] = running_losses[
                "primary_metric"
            ] / len(train_loader)
            epoch_losses["train_secondary_metric"] = running_losses[
                "secondary_metric"
            ] / len(train_loader)

            mlflow.log_metrics(epoch_losses, step=epoch)
            dose_metrics_val["val"] = evaluate_dose_metrics(
                model,
                val_set,
                hp_config_iter["patch_size"],
                config["structures"],
                device,
            )
            mlflow.log_metrics(flatten_dict(dose_metrics_val), step=epoch)

            print(
                f"EPOCH {epoch} | "
                f"tr_deepsup: {epoch_losses['train_deepsup_loss']:.3f} | "
                f"tr_primary: {epoch_losses['train_primary_metric']:.3f} | "
                f"tr_sec: {epoch_losses['train_secondary_metric']:.3f} | "
                f"val_pri: {epoch_losses['val_primary_metric']:.3f} | "
                f"val_sec: {epoch_losses['val_secondary_metric']:.3f} | "
                f"best_val_pri: {best_val_primary:.3f} | "
                f"last LR: {seq_scheduler.get_last_lr()[0]:.6f}"
            )

            """
            This is the end of training loop
            """

        model.load_state_dict(
            torch.load(
                f"trained_models/{mlflow.active_run().info.run_name}_epoch_{best_epoch}.pth"
            )
        )
        test_primary_metric, test_secondary_metric = evaluate_dataset(
            model, test_set, hp_config_iter, device
        )

        dose_metrics_test["test"] = evaluate_dose_metrics(
            model, test_set, hp_config_iter["patch_size"], config["structures"], device
        )
        # We'll use dictionary, so that after dict flattening MLflow metrics are in form val.PTV.mean
        mlflow.log_metrics(
            {
                "test_primary_metric": test_primary_metric,
                "test_secondary_metric": test_secondary_metric,
            }
        )
        mlflow.log_metrics(flatten_dict(dose_metrics_val))
        mlflow.log_metrics(flatten_dict(dose_metrics_test))
        mlflow.log_metric(key="best_val_secondary", value=best_val_secondary)
        mlflow.log_metric(key="best_val_primary", value=best_val_primary)
        mlflow.end_run()

    """
    This is for training set checking. Visualizes 3x8 figure that includes ct, mask and dose for one 8-sized batch. Figure updates every 2 seconds.
    """

    # plt.rcParams["figure.figsize"] = (10, 15)
    # plt.figure()

    # for image in train_loader:

    #    fig, axs = plt.subplots(8, 3)

    #    for j, (ax1, ax2, ax3) in enumerate(axs):
    #        ax1.imshow(image['mask'][tio.DATA].detach().numpy()[j, 0, :, :, 0])
    #        ax2.imshow(image['ct'][tio.DATA].detach().numpy()[j, 0, :, :, 0])
    #        ax2.set_title(image['name'][j])
    #        ax3.imshow(image['dose'][tio.DATA].detach().numpy()[j, 0, :, :, 0])

    #    plt.show()
    #    time.sleep(2)
    #    plt.close()


if __name__ == "__main__":
    main()
