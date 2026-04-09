import numpy as np
import torch

"""
Tekijä: Akseli Leino
"""


def calculate_dvhs(dose, mask, organ_config, num_bins=601, dose_max=60):
    """Calculate cumulative dose-volume histograms (DVHs) for each organ.

    This function computes the DVHs for each specified organ based on the input
    dose distribution and a mask that delineates different organs. The DVHs are
    calculated by binning the dose values and normalizing the histogram to show
    the volume as a percentage of the total organ volume.

    Args:
        dose (torch.Tensor): Pytorch tensor containing the patient dose distribution.
        mask (torch.Tensor): Pytorch tensor containing the delineated structures.
        organ_config (dict): A dictionary mapping organ names (str) to their corresponding int values.
        num_bins (int, optional): The number of bins between 0 and dose_max.
        dose_max (int, optional): The maximum dose and limit for x-axis.

    Returns:
        dict: A dictionary where each key is an organ name (str) and the value is
            a  PyTorch tensor representing the cumulative DVH for that organ.
    """

    dvhs = {}
    dose_min = 0
    # bin_edges = torch.linspace(dose_min, dose_max, steps = num_bins)

    for organ_name, organ_value in organ_config.items():
        organ_doses = dose[mask == organ_value]

        hist = torch.histc(organ_doses, bins=num_bins, min=dose_min, max=dose_max)

        cumulative_dvh = (
            torch.flip(torch.cumsum(torch.flip(hist, dims=[0]), dim=0), dims=[0])
            / organ_doses.numel()
            * 100
        )

        dvhs[organ_name] = cumulative_dvh

    return dvhs


def calculate_dvhs_numpy(dose, mask, organ_config, num_bins=601, max_dose=60):
    """Calculate cumulative dose-volume histograms (DVHs) for each organ.

    This function computes the DVHs for each specified organ based on the input
    dose distribution and a mask that delineates different organs. The DVHs are
    calculated by binning the dose values and normalizing the histogram to show
    the volume as a percentage of the total organ volume.

    Args:
        dose (np.ndarray): Numpy array containing the patient dose distribution.
        mask (np.ndarray): Numpy array containing the delineated structures.
        organ_config (dict): A dictionary mapping organ names (str) to their corresponding int values.
        num_bins (int, optional): The number of bins between 0 and dose_max.
        dose_max (int, optional): The maximum dose and limit for x-axis.

    Returns:
        dict: A dictionary where each key is an organ name (str) and the value is
            a  PyTorch tensor representing the cumulative DVH for that organ.
    """

    dvhs = {}
    bin_edges = np.arange(0, max_dose + 0.2, 0.1)

    for organ_name, organ_index in organ_config.items():
        organ_dose = dose[mask[:, :, :, :, organ_index] == 1]

        diff_dvh_voxels, _ = np.histogram(organ_dose, bins=bin_edges)
        if np.any(diff_dvh_voxels):
            diff_dvh = diff_dvh_voxels / np.sum(diff_dvh_voxels) * 100
        else:
            diff_dvh = diff_dvh_voxels
            print(f"- Missing {organ_name}")
        dvh = 100 - np.cumsum(diff_dvh)
        dvhs[organ_name] = dvh

    return dvhs
