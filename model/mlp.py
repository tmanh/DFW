import torch
import torch.nn as nn
import torch.nn.functional as F

from model.common import Mlp
from model.gru import minGRU


class MLP(nn.Module):
    def __init__(self, in_dim, n_layers, n_dim):
        super().__init__()
        self.in_dim = in_dim
        self.n_dim = n_dim
        self.out_dim = 1

        modules = []
        for i in range(n_layers):
            modules.extend(
                [
                    Mlp(
                        in_features=self.in_dim if i == 0 else n_dim,
                        out_features=n_dim,
                        add_noise=False,
                    ),
                    nn.Dropout(p=0.1),
                ]
            )

        self.dist = nn.Sequential(
            *modules,
        )

        self.gru = minGRU(dim=n_dim)
        self.weight = nn.Linear(in_features=n_dim, out_features=1)
        self.out = nn.Linear(in_features=n_dim, out_features=1)

    def forward(self, xs, x, lrain, nrain, valid):
        feats = self.dist(xs)
        feats = torch.cat([feats.unsqueeze(-2).repeat(1, 1, x.shape[-1], 1), x.unsqueeze(-1)], dim=-1)
        feats = self.gru(feats)
        outs = self.out(feats)
        return outs


class MLPW(MLP):
    def __init__(self, in_dim, n_layers, n_dim):
        super().__init__(in_dim, n_layers, n_dim)

    def forward(self, xs, x, lrain, nrain, valid):
        feats = self.dist(1 / (xs[:, :, :-4] + 1e-6))
        weights = self.weight(feats)
        weights = torch.sigmoid(weights)
        alpha = weights / (torch.sum(weights, dim=1, keepdim=True) + 1e-8)
        return torch.sum(x * alpha, dim=1)


class MLPR(nn.Module):
    def __init__(self, in_dim, n_layers, n_dim):
        super().__init__()
        self.in_dim = in_dim
        self.n_dim = n_dim
        self.out_dim = 1

        modules = []
        for i in range(n_layers):
            modules.extend(
                [
                    Mlp(
                        in_features=self.in_dim if i == 0 else n_dim,
                        out_features=n_dim,
                        add_noise=False,
                    ),
                    nn.Dropout(p=0.1),
                ]
            )

        self.dist = nn.Sequential(
            *modules,
        )

        self.gru = minGRU(dim=n_dim)
        self.in_mlp = nn.Linear(in_features=1, out_features=n_dim)
        
        self.weight = nn.Linear(in_features=n_dim, out_features=1)
        self.rweight = nn.Linear(in_features=n_dim, out_features=1)

        self.out = nn.Linear(in_features=2 * n_dim + 1, out_features=1)

    def forward(self, xs, x, lrain, nrain, valid):
        xs = xs[:, :, 18:]
        lrain = lrain.unsqueeze(1).unsqueeze(-1)
        feats_lr = self.in_mlp(lrain)
        feats_lr = self.gru(feats_lr)

        nrain = nrain.unsqueeze(-1)
        feats_nr = self.in_mlp(nrain)
        feats_nr = self.gru(feats_nr)

        feats = self.dist(xs)
        spatial_weights = self.weight(feats)
        spatial_weights = torch.sigmoid(spatial_weights) * valid
        
        diff_r = feats_nr * feats_lr
        r_weights = self.rweight(diff_r)
        r_weights = torch.sigmoid(r_weights).squeeze(-1) * valid

        weights = spatial_weights * r_weights
        weights = weights / (torch.sum(weights, dim=1, keepdim=True) + 1e-6)

        outs = torch.cat([diff_r, feats.unsqueeze(-2).repeat(1, 1, diff_r.shape[2], 1), x.unsqueeze(-1)], dim=-1)
        outs = self.out(outs).squeeze(-1) + x

        weights = weights * valid
        alpha = weights / (torch.sum(weights, dim=1, keepdim=True) + 1e-8)

        return torch.sum(outs * alpha, dim=1)


class MLPRW(nn.Module):
    def __init__(self, in_dim, n_layers, n_dim):
        super().__init__()
        self.in_dim = in_dim
        self.n_dim = n_dim
        self.out_dim = 1

        self.dist = Mlp(
            in_features=self.in_dim,
            out_features=n_dim,
            add_noise=False,
        )

        self.dist_loc = Mlp(
            in_features=2,
            out_features=n_dim,
            add_noise=False,
        )

        self.dist_loc_2 = Mlp(
            in_features=2 * n_dim,
            out_features=1,
            add_noise=False,
        )

        self.gru = minGRU(dim=n_dim)
        self.in_mlp = nn.Linear(in_features=1, out_features=n_dim)
        self.weight = nn.Linear(in_features=n_dim + n_dim, out_features=1)

    def forward(self, xs, x, lrain, nrain, valid):
        tgt_pos = xs[:, :, -4:-2]
        src_pos = xs[:, :, -2:]
        rel = xs[:, :, :-4]
        
        mod = self.dist_loc_2(torch.cat([
            self.dist_loc(tgt_pos), self.dist_loc(src_pos)
        ], dim=-1))

        lrain = lrain.unsqueeze(1).unsqueeze(-1)
        feats_lr = self.in_mlp(lrain)
        feats_lr = self.gru(feats_lr)

        nrain = nrain.unsqueeze(-1)
        feats_nr = self.in_mlp(nrain)
        feats_nr = self.gru(feats_nr)

        diff_r = feats_nr * feats_lr

        feats = self.dist(rel)

        outs = torch.cat([diff_r, feats.unsqueeze(-2).repeat(1, 1, diff_r.shape[2], 1)], dim=-1)

        weights = self.weight(outs).squeeze(-1) * mod
        
        weights = torch.sigmoid(weights) #* valid
        alpha = weights / (torch.sum(weights, dim=1, keepdim=True) + 1e-8)

        return torch.sum(x * alpha, dim=1)


def get_i_inputs(xs, inputs):
    if inputs == 'update':
        ixs = torch.cat(
            [
                xs[:, :, 0:1],    # distance
                -xs[:, :, 1:2],   # displacement[0]
                -xs[:, :, 2:3],   # displacement[1]
                xs[:, :, 3:9],    # elevations stats
                -xs[:, :, 9:10],  # elevation diff
                xs[:, :, 11:12],  # key slope
                xs[:, :, 10:11],  # slope
                xs[:, :, 12:],    # slope stats
            ], dim=2
        )

    return ixs
