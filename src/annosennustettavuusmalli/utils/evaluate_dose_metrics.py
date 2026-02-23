import torch
import torchio as tio
from collections import defaultdict
import numpy as np
from .integer_mask_to_binary import integer_mask_to_binary

"""
Tekijä: Akseli Leino
"""

def evaluate_dose_metrics(model, dataset, patch_size, structure_config, device, average=True, verbose=False, subject_ids=False):
    model.eval()
    metrics = {structure: [] for structure in structure_config}
    subject_ids_list = []
    model = model.to(device)
    with torch.no_grad():
        for subject in dataset:
            if verbose:
                print(subject.name)

            subject_ids_list.append(subject.name)

            inference_sampler = tio.data.GridSampler(subject, patch_size)
            loader = torch.utils.data.DataLoader(inference_sampler, batch_size = 1, num_workers = 4) # Here num_workers can be > 0 as tio.Queue is not used
            aggregator = tio.inference.GridAggregator(inference_sampler)

            for patches_batch in loader:
                ct = patches_batch['ct'][tio.DATA].to(dtype=torch.float, device=device)
                mask = patches_batch['mask'][tio.DATA].to(dtype=torch.float, device=device)
                PTV_dist = patches_batch['distance_to_PTV'][tio.DATA].to(dtype=torch.float, device=device)
                locations = patches_batch[tio.LOCATION]
                input_ = torch.cat((ct, mask, PTV_dist), dim = 1)
                output = model(input_).main_output
                aggregator.add_batch(output, locations)

            pred_dose = aggregator.get_output_tensor()
            dose = subject['dose'][tio.DATA]
            binary_mask = integer_mask_to_binary(subject['original_mask'][tio.DATA], 11)
            
            for structure, mask_number in structure_config.items():
                structure_mask = binary_mask[:, :, :, :, mask_number]
                structure_metrics = calculate_structure_metrics(dose, pred_dose, structure_mask)
                metrics[structure].append(structure_metrics)
        averaged_metrics = average_metrics(metrics)

    if average and not subject_ids:
        return averaged_metrics
    elif average and subject_ids:
        return averaged_metrics, subject_ids_list
    elif not average and subject_ids:
        return metrics, subject_ids_list
    else:
        return metrics

def calculate_structure_metrics(dose, pred_dose, structure_mask):
    structure_pred_dose = pred_dose[structure_mask==1]
    structure_true_dose = dose[structure_mask==1]
    mean_pred_dose = structure_pred_dose.nanmean()
    mean_true_dose = structure_true_dose.nanmean()
    mean_abs_error = torch.abs(torch.sub(mean_pred_dose, mean_true_dose))

    # You can implement more metrics here and append them to the return dict
    return {'mean_pred_dose': mean_pred_dose.item(),
            'mean_true_dose': mean_true_dose.item(),
            'mean_abs_error': mean_abs_error.item()}

def average_metrics(metrics):
    average_metrics = {}
    for structure, metric_dicts in metrics.items():
        aggregated_metrics = defaultdict(list)
        for metric_dict in metric_dicts:
            for metric_name, metric_value in metric_dict.items():
                aggregated_metrics[metric_name].append(metric_value)
    
        averaged_structure_metrics = {metric_name: np.nanmean(metric_values)
                                     for metric_name, metric_values in aggregated_metrics.items()}
        average_metrics[structure] = averaged_structure_metrics
    return average_metrics