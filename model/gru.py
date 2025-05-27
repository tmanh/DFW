# https://arxiv.org/abs/2410.01201v1

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Linear, Identity, Module

from model.common import Mlp

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
    

class SensorEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SensorEncoder, self).__init__()
        self.time_encoder = nn.GRU(66, hidden_dim, batch_first=True)
        self.pos_encoder = nn.Linear(input_dim, hidden_dim)  # distance or other static features
        self.fc = nn.Linear(2 * hidden_dim, hidden_dim)

    def forward(self, x, pos):
        """
        x: (batch, sensors, time, features)
        pos: (batch, sensors, 1) - distance or other static info
        """
        B, S, T, F = x.shape
        
        feats = x.view(B * S, T, F)
        
        _, h = self.time_encoder(feats)  # h: (1, B*S, H)
        h = h.view(B, S, -1)
        
        pos_emb = self.pos_encoder(pos) # (B, S, pos_dim)
        combined = torch.cat([h, pos_emb], dim=-1)
        
        return self.fc(combined)  # (B, S, hidden_dim)


class AttentionFusion(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionFusion, self).__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.scale = hidden_dim ** 0.5

    def forward(self, encoded_sensors, x):
        """
        encoded_sensors: (B, S, H)
        """
        Q = self.query(encoded_sensors[:, 0:1])  # query from first sensor or dummy (B, 1, H)
        K = self.key(encoded_sensors)  # (B, S, H)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, 1, S)
        attn_weights = torch.softmax(attn_scores, dim=-1)  # (B, 1, S)
        print(attn_weights.shape, x.shape)
        exit()
        context = torch.matmul(attn_weights, x.permute(2, 1, 0))  # (B, 1, H)
        return context.squeeze(1)  # (B, H)


class GRU(nn.Module):
    def __init__(self, in_dim=1, n_dim=64):
        super(GRU, self).__init__()
        self.encoder = SensorEncoder(in_dim, n_dim)
        self.attn = AttentionFusion(n_dim)
        self.decoder = nn.Sequential(
            nn.Linear(n_dim, n_dim),
            nn.ReLU(),
            nn.Linear(n_dim, 1)
        )

    def forward(self, xs, x, lrain, nrain, valid):
        """
        x: (B, S, T, F) - sensor time series
        pos: (B, S, 1) - sensor distances
        """
        sensor_repr = self.encoder(x, xs)  # (B, S, H)

        B, S, C = sensor_repr.shape

        fused = self.attn(sensor_repr, x)  # (B, H)
        fused = fused.view(B, -1, 1)
        
        return fused  # (B, output_len)