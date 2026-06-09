import torch
from scipy.optimize import linear_sum_assignment


def minibatch_ot_pair(x0, x1):
    cost = torch.cdist(x0.detach(), x1.detach()).pow(2).cpu().numpy()
    _, col = linear_sum_assignment(cost)
    order = torch.as_tensor(col, device=x1.device)
    return x0, x1[order]


def apply_pairing(x0, x1, config):
    pairing = config.get("pairing", "independent")
    if pairing == "independent":
        return x0, x1
    if pairing == "minibatch_ot":
        return minibatch_ot_pair(x0, x1)
    raise ValueError(f"Unknown pairing: {pairing}")
