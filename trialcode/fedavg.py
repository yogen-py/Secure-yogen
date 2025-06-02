import torch

def fed_avg(state_dicts):
    avg_state_dict = {}
    for key in state_dicts[0].keys():
        avg_state_dict[key] = torch.stack([sd[key] for sd in state_dicts], dim=0).mean(dim=0)
    return avg_state_dict

