import random
import copy

def random_sample_hyperparameters(config):
    sampled_config = {}
    for key, values in copy.deepcopy(config).items():
        if isinstance(values, list):
            sampled_config[key] = random.choice(values)
        else:
            sampled_config[key] = values
    
    for key, values in copy.deepcopy(sampled_config).items():
        if isinstance(values, dict):
            try:
                sampled_config[key] = random.randint(values['min'], values['max'])
            # Only optimizer should be here, but any other is ok as long as it has only type and params keys
            except KeyError:
                sampled_config[key]['type'] = values['type']
                for optim_param, optim_param_values in values['params'].items():
                    sampled_config[key]['params'][optim_param] = random.uniform(optim_param_values['min'], optim_param_values['max']) 
    return sampled_config

def grid_sample_hyperparameters():
    raise NotImplementedError("Implement yourself or use manual search")