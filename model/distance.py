# https://arxiv.org/abs/2410.01201v1

import torch
import torch.nn as nn


# Displacement
# TODO: distance, slope, elevation diff
class InverseDistance(nn.Module):
    def __init__(self, in_dim, n_layers, n_dim):
        super().__init__()

    def forward(self, xs, x, lrain, nrain, valid):
        """
        xs: Feature vectors of neighboring stations - batch x stations x channels
        x: Time series of neighboring stations - batch x stations x time series (168)
        lrain: Time series of rainfall of the target station  - batch x time series (168)
        nrain: Time series of rainfall of the neighboring stations  - batch x stations x time series (168)
        valid: Valid mask - batch x stations x time series (168)
        """
        xs[:, :, :1][xs[:, :, :1]<0] = 9999
        weights = 1 / (torch.abs(xs[:, :, :1]) )
        alpha = weights / (torch.sum(weights, dim=1, keepdim=True) + 1e-8)
        return torch.sum(x * alpha, dim=1, keepdim=True)

    def freeze(self):
        pass

    def unfreeze(self):
        pass


class InverseDistance2(nn.Module):
    def __init__(self, in_dim, n_layers, n_dim):
        super().__init__()

    def forward(self, xs, x, lrain, nrain, valid):
        """
        xs: Feature vectors of neighboring stations - batch x stations x channels
        x: Time series of neighboring stations - batch x stations x time series (168)
        lrain: Time series of rainfall of the target station  - batch x time series (168)
        nrain: Time series of rainfall of the neighboring stations  - batch x stations x time series (168)
        valid: Valid mask - batch x stations x time series (168)
        """
        weights = 1 / torch.sqrt((xs[:, :, 1:2])**2 + (xs[:, :, 2:3])**2)
        alpha = weights / (torch.sum(weights, dim=1, keepdim=True) + 1e-8)
        return torch.sum(x * alpha, dim=1, keepdim=True)

    def freeze(self):
        pass

    def unfreeze(self):
        pass