# https://arxiv.org/abs/2410.01201v1

import torch
import torch.nn as nn

from scipy.interpolate import Rbf


# Displacement
# TODO: distance, slope, elevation diff
class RBF(nn.Module):
    def __init__(self, in_dim, n_layers, n_dim, fmts):
        super().__init__()

    def forward(self, xs, x, inputs, train, stage):
        out = torch.zeros((x.shape[0], x.shape[2]), device=x.device)
        for b in range(xs.shape[0]):
            for i in range(x.shape[2]):
                lx = xs[b, :, 1].detach().cpu().numpy()
                ly = xs[b, :, 2].detach().cpu().numpy()
                lv = x[b, :, i].detach().cpu().numpy()

                if lv.shape[0] > 1:
                    rbf = Rbf(
                        lx, ly, lv,
                        function='multiquadric'
                    )
                
                    # Interpolation at the target point
                    z_interp = rbf([0.0], [0.0])
                else:
                    z_interp = lv
                
                out[b, i] = torch.tensor(z_interp, device=x.device)

        return out

    def freeze(self):
        pass

    def unfreeze(self):
        pass
