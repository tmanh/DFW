# https://arxiv.org/abs/2410.01201v1

import numpy as np
from pykrige import OrdinaryKriging
from pykrige.uk import UniversalKriging

from joblib import Parallel, delayed

import torch
import torch.nn as nn


# Displacement
# TODO: distance, slope, elevation diff
class OrdinaryKrigingInterpolation(nn.Module):
    def __init__(self, in_dim, n_layers, n_dim):
        super().__init__()

    def forward(self, xs, x, lrain, nrain, valid):
        try:
            return fast_ok_kriging_time_series(xs, x, valid)
        except:
            weights = (1 / xs[:, :, 0:1]) * valid
            alpha = weights / (torch.sum(weights, dim=1, keepdim=True) + 1e-8)
            return torch.sum(x * alpha, dim=1, keepdim=True)

    def freeze(self):
        pass

    def unfreeze(self):
        pass


def fast_uk_kriging_time_series(xs, x):
    """
    xs: (1, 3, 3)  – 1 batch, 3 sensors, columns: ID, x, y
    x:  (1, 3, T)  – 1 batch, 3 sensors, T time steps
    """
    assert xs.shape[0] == 1, "This version only supports batch size 1 for now"
    lx = xs[0, :, 1].detach().cpu().numpy()
    ly = xs[0, :, 2].detach().cpu().numpy()
    elevation = xs[0, :, 9].detach().cpu().numpy()  # shape: (N,)

    lv_all = x[0].detach().cpu().numpy()  # shape: (3, T)

    results = []

    for t in range(lv_all.shape[1]):
        # Build UK with elevation as external drift
        uk = UniversalKriging(
            lx, ly, lv_all[:, t],
            variogram_model='linear',
            drift_terms=['external_Z'],
            external_drift=elevation,
            external_drift_coordinates=np.column_stack((lx, ly))
        )

        z, _ = uk.execute('points', [0.0], [0.0])
        results.append(z[0])

    return torch.tensor(results, device=x.device).unsqueeze(0)  # (1, T)


def fast_ok_kriging_time_series(xs, x, valid):
    """
    xs: (1, 3, 3)  – 1 batch, 3 sensors, columns: ID, x, y
    x:  (1, 3, T)  – 1 batch, 3 sensors, T time steps
    """
    assert xs.shape[0] == 1, "This version only supports batch size 1 for now"
    lx = xs[0, :, 1].detach().cpu().numpy()
    ly = xs[0, :, 2].detach().cpu().numpy()
    lv_all = x[0].detach().cpu().numpy()  # shape: (3, T)

    uk = OrdinaryKriging(
        lx, ly, lv_all[:, 0],
        variogram_model='linear',
    )

    # Run kriging in parallel for all time steps
    results = []
    for t in range(lv_all.shape[1]):
        uk.Z = np.atleast_1d(np.squeeze(np.array(lv_all[:, t], copy=True, dtype=np.float64)))
        z, _ = uk.execute('points', [0.0], [0.0])
        results.append(z[0])

    return torch.tensor(results, device=x.device).unsqueeze(0)  # (1, T)


class UniversalKrigingInterpolation(nn.Module):
    def __init__(self, in_dim, n_layers, n_dim):
        super().__init__()

    def forward(self, xs, x, lrain, nrain, valid):
        try:
            return fast_uk_kriging_time_series(xs, x)
        except:
            weights = 1 / xs[:, :, 0:1]
            alpha = weights / torch.sum(weights, dim=1, keepdim=True)
            return torch.sum(x * alpha, dim=1, keepdim=True)

    def freeze(self):
        pass

    def unfreeze(self):
        pass
