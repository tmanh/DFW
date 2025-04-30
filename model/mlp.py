import torch
import torch.nn as nn
import torch.nn.functional as F

from model.common import Mlp


class MLP(nn.Module):
    def __init__(self, in_dim, n_layers, n_dim, fmts):
        super().__init__()
        self.in_dim = in_dim
        self.n_dim = n_dim
        self.out_dim = 1
        self.fmts = fmts

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

        if 'wo' in fmts:
            self.out_dim = 2

        self.dist = nn.Sequential(
            *modules,
        )

        self.out = nn.Linear(in_features=n_dim, out_features=1)
        self.weight = nn.Linear(in_features=n_dim, out_features=1)

    def forward_wo(self, x, xs):
        outs, weights = self.step_wo(x, xs)
        return torch.sum(outs * weights, dim=1)

    def forward_w(self, xs, x):
        weights = self.step(xs)
        return torch.sum(
            x * torch.softmax(weights, dim=1),
            dim=1,
            keepdim=True
        )

    def forward_o(self, x, xs):
        return self.step(x, xs)

    def forward(self, xs, x, inputs, train, stage):
        if train:
            if self.fmts == 'w':
                return self.forward_w(xs, x)
            elif 'i' not in self.fmts:
                if stage == 1:
                    return self.forward_o(x, xs)
                elif stage == 2:
                    return self.forward_wo(x, xs)
            elif 'i' in self.fmts:
                return self.forward_o(x, xs)
        else:
            if self.fmts == 'w':
                return self.forward_w(xs, x)
            elif self.fmts == 'o':
                return self.forward_o(x, xs)
            elif self.fmts == 'wo':
                return self.forward_wo(x, xs)
            elif self.fmts == 'io':
                return self.forward_io(xs, x, inputs)
            else:
                raise ValueError('Invalid format')

    def forward_io(self, xs, x, inputs):
        outs = self.forward_o(x, xs)

        ixs = get_i_inputs(xs, inputs)
        iouts = self.forward_o(outs, ixs)

        diff = torch.abs(x - iouts) / torch.abs(x)
        diff = F.softmax(1 / (diff + 1e-7), dim=1)

        if x.shape[1] == 1:
            return torch.sum(x * diff, dim=1, keepdim=True)
        else:
            return torch.sum(outs * diff, dim=1, keepdim=True)

    def step_wo(self, x, xs):
        x = x.unsqueeze(-1)
        xs = xs.unsqueeze(-2).repeat(1, 1, x.shape[2], 1)
        xs = torch.cat([x, xs], dim=-1)

        feats = self.dist(xs)
        outs = (x + self.out(feats)).squeeze(-1)
        weights = self.weight(feats).squeeze(-1)

        weights = torch.softmax(weights, dim=1)

        return outs, weights

    def step(self, x, xs):
        x = x.unsqueeze(-1)
        xs = xs.unsqueeze(-2).repeat(1, 1, x.shape[2], 1)

        xs = torch.cat([x, xs], dim=-1)

        feats = self.dist(xs)
        outs = (self.out(feats) + x).squeeze(-1)
        return outs

    def freeze(self):
        for param in self.dist.parameters():
            param.requires_grad = False
        for param in self.out.parameters():
            param.requires_grad = False
        self.dist.eval()
        self.out.eval()

    def unfreeze(self):
        for param in self.dist.parameters():
            param.requires_grad = True
        for param in self.out.parameters():
            param.requires_grad = True
        self.dist.train()
        self.out.train()


class MLPW(MLP):
    def __init__(self, in_dim, n_layers, n_dim, fmts):
        super().__init__(in_dim, n_layers, n_dim, fmts)

    def forward(self, xs, x, valid, inputs=None, train=None, stage=None):
        feats = self.dist(xs)
        weights = self.weight(feats)
        weights = torch.softmax(weights, dim=1) * valid
        weights = weights / (torch.sum(weights, dim=1, keepdim=True) + 1e-8)
        return torch.sum(x * weights, dim=1, keepdim=True)


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
