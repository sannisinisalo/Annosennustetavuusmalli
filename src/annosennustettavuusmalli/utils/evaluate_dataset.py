import torch
import torch.nn as nn
import torchio as tio
from torchio.constants import DATA

"""
Tekijä: Akseli Leino
"""


def evaluate_dataset(model, dataset, config, device):
    """Uses given model and dataset to evaluate metrics defined in config.

    Args:
        model (torch model): Your PyTorch model
        dataset (tio dataloader): Your dataset as TorchIO dataloader
        config (dict): config dict that has specified primary and secondary loss
        device (torch.device): Device you want to use.

    Returns:
        dataset_primary (float): Primary loss for the datasets
        dataset_secondary (float): Secondary loss for the dataset
    """
    loss_primary = getattr(nn, config["primary_loss"])()
    loss_secondary = getattr(nn, config["secondary_loss"])()

    with torch.no_grad():
        model.eval()
        dataset_primary = 0
        dataset_secondary = 0

        for subject in dataset:
            inference_sampler = tio.data.GridSampler(subject, config["patch_size"])
            dataloader = torch.utils.data.DataLoader(
                inference_sampler, batch_size=4, num_workers=4
            )
            subject_primary = 0
            subject_secondary = 0

            for batch in dataloader:
                input_ct = batch["ct"][DATA].float()
                input_mask = batch["mask"][DATA].float()
                PTV_dist = batch["distance_to_PTV"][DATA].float()
                input_ = torch.cat((input_ct, input_mask, PTV_dist), dim=1)
                true_dose = batch["dose"][DATA].float()

                input_ = input_.to(device)
                true_dose = true_dose.to(device)

                pred_dose = model(input_)
                batch_primary = loss_primary(pred_dose.main_output, true_dose)
                batch_secondary = loss_secondary(pred_dose.main_output, true_dose)

                subject_primary += batch_primary.item()
                subject_secondary += batch_secondary.item()

            dataset_primary += subject_primary / len(dataloader)
            dataset_secondary += subject_secondary / len(dataloader)

        dataset_primary = dataset_primary / len(dataset)
        dataset_secondary = dataset_secondary / len(dataset)

        return dataset_primary, dataset_secondary
