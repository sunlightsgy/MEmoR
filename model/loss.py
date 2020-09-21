import torch.nn.functional as F


def nll_loss(output, target):
    return F.nll_loss(output, target)

def cross_entropy_loss(output, target):
    return F.cross_entropy(output, target)

def cross_entropy_loss_with_weight(output, target, weight):
    return F.cross_entropy(output, target, weight=weight)

def mse_loss(output, target):
    return F.mse_loss(output, target)