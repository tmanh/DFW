# https://arxiv.org/abs/2410.01201v1

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Linear, Identity, Module

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# appendix B
# https://github.com/glassroom/heinsen_sequence

def heinsen_associative_scan_log(log_coeffs, log_values):
    a_star = log_coeffs.cumsum(dim = 1)
    log_h0_plus_b_star = (log_values - a_star).logcumsumexp(dim = 1)
    log_h = a_star + log_h0_plus_b_star
    return log_h.exp()

# appendix B.3

def g(x):
    return torch.where(x >= 0, x + 0.5, x.sigmoid())

def log_g(x):
    return torch.where(x >= 0, (F.relu(x) + 0.5).log(), -F.softplus(-x))

# log-space version of minGRU - B.3.1
# they enforce the hidden states to be positive


def selective_blend(x, iouts, outs, weights, pweights, threshold=0.05):
    # x, iouts, outs: (B, N, T) tensors

    # Compute normalized difference
    diff = torch.abs(x - iouts)
    diff = diff / (torch.abs(x) + 1e-7)
    diff = 1 / (diff + 1e-7) # torch.log(1 / (diff + 1e-7))
    diff = diff / torch.sum(diff, dim=1, keepdim=True)

    # Compute attention weights
    weights = weights / torch.sum(weights, dim=1, keepdim=True)
    alpha = F.softmax(weights * diff * pweights, dim=1)  # (B, N, T)

    # Use x where diff > threshold, else use outs
    mask = (diff > threshold).float()  # 1 where we want to use x
    final_values = mask * x + (1 - mask) * outs  # selectively mix x and outs

    # Blend using weights
    result = torch.sum(final_values * alpha, dim=1, keepdim=True)  # (B, 1, T)
    return result


class minGRU(Module):
    def __init__(self, dim, expansion_factor = 1., proj_out = None):
        super().__init__()

        dim_inner = int(dim * expansion_factor)
        proj_out = default(proj_out, expansion_factor != 1.)

        self.to_hidden_and_gate = Linear(dim, dim_inner * 2, bias = False)
        self.to_out = Linear(dim_inner, dim, bias = False) if proj_out else Identity()

    def forward(self, x, prev_hidden = None, return_next_prev_hidden = False):
        seq_len = x.shape[1]
        hidden, gate = self.to_hidden_and_gate(x).chunk(2, dim = -1)

        if seq_len == 1:
            hidden = g(hidden)
            gate = gate.sigmoid()
            out = torch.lerp(prev_hidden, hidden, gate) if exists(prev_hidden) else (hidden * gate)
        else:
            # parallel
            log_coeffs = -F.softplus(gate)

            log_z = -F.softplus(-gate)
            log_tilde_h = log_g(hidden)
            log_values = log_z + log_tilde_h

            if exists(prev_hidden):
                log_values = torch.cat((prev_hidden.log(), log_values), dim = 1)
                log_coeffs = F.pad(log_coeffs, (0, 0, 1, 0))

            out = heinsen_associative_scan_log(log_coeffs, log_values)
            out = out[:, -seq_len:]

        next_prev_hidden = out[:, -1:]

        out = self.to_out(out)

        if not return_next_prev_hidden:
            return out

        return out, next_prev_hidden
    

class GRU(nn.Module):
    def __init__(self, in_dim, n_layers, n_dim, fmts):
        super().__init__()
        self.in_dim = in_dim
        self.n_dim = n_dim
        self.out_dim = 1
        self.fmts = fmts

        self.in_mlp = nn.Sequential(
            nn.Linear(in_dim, n_dim),
            nn.ReLU(),
        )
        
        modules = []
        for _ in range(n_layers):
            modules.extend(
                [
                    minGRU(dim=n_dim),
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
        weights = 1 / xs[:, :, 0:1]

        outs, pweights = self.step_wo(x, xs)

        ixs = get_i_inputs(xs, inputs)
        iouts = self.forward_o(outs, ixs)

        return selective_blend(x, iouts, outs, weights, pweights)
    
    def step_wo(self, x, xs):
        x = x.unsqueeze(-1)
        xs = xs.unsqueeze(-2).repeat(1, 1, x.shape[2], 1)
        xs = torch.cat([x, xs], dim=-1)

        feats = self.in_mlp(xs)
        B, N, L, C = feats.shape
        feats = feats.permute(0, 2, 1, 3).view(B * L, N, C)
        feats = self.dist(feats)
        feats = feats.view(B, L, N, C).permute(0, 2, 1, 3)
        outs = (self.out(feats) + x).squeeze(-1)
        weights = self.weight(feats).squeeze(-1)
        
        weights = torch.softmax(weights, dim=1)

        return outs, weights
    
    def step(self, x, xs):
        x = x.unsqueeze(-1)
        xs = xs.unsqueeze(-2).repeat(1, 1, x.shape[2], 1)

        xs = torch.cat([x, xs], dim=-1)

        feats = self.in_mlp(xs)
        B, N, L, C = feats.shape
        feats = feats.permute(0, 2, 1, 3).view(B * L, N, C)
        feats = self.dist(feats)
        feats = feats.view(B, L, N, C).permute(0, 2, 1, 3)
        outs = (self.out(feats) + x).squeeze(-1)

        return outs

    def freeze(self):
        for param in self.in_mlp.parameters():
            param.requires_grad = False
        for param in self.dist.parameters():
            param.requires_grad = False
        for param in self.out.parameters():
            param.requires_grad = False
        self.in_mlp.eval()
        self.dist.eval()
        self.out.eval()

    def unfreeze(self):
        for param in self.in_mlp.parameters():
            param.requires_grad = True
        for param in self.dist.parameters():
            param.requires_grad = True
        for param in self.out.parameters(): 
            param.requires_grad = True
        self.in_mlp.train()
        self.dist.train()
        self.out.train()


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