import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import scipy
from torch_geometric.utils import softmax
from torch_geometric.nn.conv import MessagePassing

from sklearn.neighbors import NearestNeighbors
from model.common import Mlp
from model.gru import minGRU

from typing import Callable, Optional, Tuple


class fullGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=True, dropout=0.0, bidirectional=False):
        super(fullGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)

    def forward(self, x):
        # x: [B, T, C]
        out, _ = self.gru(x)  # [B, T, 2*H]
        return out


class EdgeAttrGNNLayer(MessagePassing):
    def __init__(self, in_channels, edge_dim, out_channels=8):
        super().__init__(aggr='add')  # or 'mean', 'max'

        self.lin_edge = nn.Sequential(
            torch.nn.Linear(edge_dim, 16),
            torch.nn.GELU(),
            torch.nn.Linear(16, 16),
            torch.nn.GELU(),
            torch.nn.Linear(16, 1)
        )

        self.enc = nn.Sequential(
            torch.nn.Linear(1, 8),
            torch.nn.GELU(),
        )
        
        self.node_update = nn.Identity()   # drop-in point for per-node MLP if you want
        if out_channels != -1:
            self.post_update = nn.Sequential(
                # minGRU(out_channels, expansion_factor=2),
                # nn.Linear(8, 8), 
                fullGRU(8, 8, batch_first=True),
                torch.nn.GELU(),
                nn.Linear(8, 1),
            )
        else:
            self.post_update = nn.Identity()

    @torch.no_grad()
    def _hops_to_target0(self, edge_index, num_nodes):
        """Directed hop distance to node 0 via reverse-BFS. Unreachable = -1."""
        src, dst = edge_index
        rev_adj = [[] for _ in range(num_nodes)]
        for s, d in zip(src.tolist(), dst.tolist()):
            rev_adj[d].append(s)
        hop = torch.full((num_nodes,), -1, dtype=torch.long, device=src.device)
        q = [0]
        hop[0] = 0
        i = 0
        while i < len(q):
            v = q[i]; i += 1
            for u in rev_adj[v]:
                if hop[u] == -1:
                    hop[u] = hop[v] + 1
                    q.append(u)
        return hop

    def forward(self, x, edge_index, edge_attr, valid=None, pos_feats=None):
        N, L, C = x.shape
        device = x.device

        if valid is None:
            valid = torch.ones(N, dtype=torch.bool, device=device)
        elif valid.dim() == 2 and valid.size(-1) == 1:
            valid = valid.squeeze(-1)
        valid = valid.bool()

        src, dst = edge_index[0], edge_index[1]

        # ---- your attention stack applied ONCE to get per-edge logits ----
        logits = self.lin_edge(edge_attr).squeeze(-1)  # [E, 1]
        # ------------------------------------------------------------------

        # Level-schedule: only edges that move one hop closer to the target(0)
        hops = self._hops_to_target0(edge_index, N)     # [N]
        on_path = (hops[src] >= 0) & (hops[dst] >= 0) & (hops[src] == hops[dst] + 1)
        if not torch.any(on_path):
            return x  # nothing flows to target(0)

        x_out = x  # [N, C] -> [N, 8]
        max_hop = int(hops.max().item())

        # process farthest -> nearest so intermediates update before the target
        for h in range(max_hop, 0, -1):
            mask = on_path & (hops[src] == h) & (hops[dst] == h - 1)
            if not torch.any(mask):
                continue

            ei = edge_index[:, mask]     # [2, Eh]
            log_h = logits[mask]         # [Eh]

            self.length = x.shape[-1]

            out_h = self.propagate(
                ei, x=x_out.view(x.shape[0], -1),
                edge_weight=log_h, size=(N, N),
                valid=valid.squeeze(-1),
                pos_feats=pos_feats
            )
            out_h = out_h.view(*x.shape)  # [N, 8] -> [N, 8]
            
            x_out = x_out + self.node_update(out_h)  # [N, 8] -> [N, 1]

        return self.post_update(x_out).squeeze(-1)  # [N, 8] -> [N, 1]
        # return x_out.squeeze(-1)  # [N, 8] -> [N, 1]

    def forward_legacy(self, x, edge_index, edge_attr, valid):
        # x: [num_nodes, in_channels]
        # edge_index: [2, num_edges]
        # edge_attr: [num_edges, edge_dim]
        # x = self.lin_node(x)
        edge_attr = self.lin_edge(edge_attr)
        
        return self.propagate(edge_index, x=x, edge_weight=edge_attr, valid=valid.squeeze(-1))

    def message(self, x_j: torch.Tensor, edge_weight: torch.Tensor, index: torch.Tensor, valid_j: torch.Tensor, pos_feats_i: torch.Tensor, pos_feats_j: torch.Tensor) -> torch.Tensor:
        """
        x_j:         [E_h, C]   source features for this level
        edge_weight: [E_h]      scalar logits per edge
        index:       [E_h]      destination node indices (grouping key)
        """
        valid_j = torch.mean(valid_j.float(), dim=-1)  # Debugging line
        alpha = softmax(edge_weight * valid_j, index)      # normalize over incoming edges per dst
        return x_j.contiguous().view(x_j.shape[0], -1) * alpha.unsqueeze(-1)

    def update(self, aggr_out):
        return aggr_out  # self.lin_update(aggr_out)


class GATWithEdgeAttr(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=1):
        super(GATWithEdgeAttr, self).__init__()
        hidden_channels = 48
        edge_dim = 16
        self.hidden_channels = hidden_channels
        self.gat = EdgeAttrGNNLayer(hidden_channels, edge_dim=edge_dim, out_channels=-1)

    def forward(self, nodes, edge_index, edge_attr, valid, r, fx):
        with torch.no_grad():
            valid = valid.squeeze(0).unsqueeze(-1)
            nodes = nodes * valid

            N, L, C = nodes.shape
            nodes = nodes.view(N, -1)

        pred = self.gat(nodes.unsqueeze(-1), edge_index, edge_attr, valid)

        return pred[:1]
    

class RainModel(torch.nn.Module):
    def __init__(self):
        super(RainModel, self).__init__()
        
        self.rain_mlp = nn.Sequential(
            nn.Linear(1, 8),
            nn.GELU(),
            nn.Linear(8, 8),
        )

        self.rain_pos_enc = nn.Sequential(
            nn.Linear(2, 8),
            nn.GELU(),
            nn.Linear(8, 8),
        )

    def forward(self, r, pos):
        # r: [N, L, 1]
        # pos: [N, L, 2]
        r = self.rain_mlp(r.squeeze(0).unsqueeze(-1)).squeeze(-1)
        return r
    

class MLPW(nn.Module):
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

    def forward(self, xs, x):
        feats = self.dist(xs)
        weights = self.weight(feats)
        weights = torch.relu(weights)
        alpha = weights / (torch.sum(weights, dim=0, keepdim=True) + 1e-8)
        return torch.sum(x * alpha, dim=0, keepdim=True)


def forward_distance(xs, x):
    """
    xs: Feature vectors of neighboring stations - batch x stations x channels
    x: Time series of neighboring stations - batch x stations x time series (168)
    lrain: Time series of rainfall of the target station  - batch x time series (168)
    nrain: Time series of rainfall of the neighboring stations  - batch x stations x time series (168)
    valid: Valid mask - batch x stations x time series (168)
    """

    xs[:, :1][xs[:, :1] < 0] = 9999
    weights = 1 / (torch.abs(xs[:, :1]) )
    alpha = weights / (torch.sum(weights, dim=0, keepdim=True) + 1e-8)
    return torch.sum(x * alpha, dim=0, keepdim=True)


class GATWithEdgeAttrRain(torch.nn.Module):
    def __init__(self, hidden_channels=48, edge_dim = 12):
        super(GATWithEdgeAttrRain, self).__init__()

        edge_dim = 3
        self.hidden_channels = hidden_channels
        self.gat = EdgeAttrGNNLayer(hidden_channels, edge_dim=edge_dim, out_channels=8)

        self.residual_mlp = MLPW(12, 2, 8)
        self.residual_gat = EdgeAttrGNNLayer(hidden_channels, edge_dim=edge_dim, out_channels=-1)
        self.rain_mlp = RainModel()

        self.theta = torch.nn.Parameter(torch.Tensor([0.1, 0.1, 0.1]))

    def forward(self, nodes, edge_index, edge_attr, valid, r, fx, loc, earray):
        with torch.no_grad():
            valid = valid.squeeze(0).unsqueeze(-1)
            nodes = nodes * valid

            N, L, C = nodes.shape
            nodes = nodes.view(N, -1)
            earray = earray.squeeze(0)

        rf = self.rain_mlp(r, fx)

        pred_coarse = self.gat.post_update(rf)

        res_idx = earray.shape[0] # self.find_n_neighbors(r, rf)

        # pred = pred_coarse.squeeze(
        # -1)
        # edge_attr = edge_attr[:, :19]
        edge_attr = edge_attr[:, :3]
        # edge_attr = torch.cat([edge_attr[:, :3], edge_attr[:, 9:19]], dim=1)
        # edge_attr = torch.cat([edge_attr[:, :9], edge_attr[:, 17:19]], dim=1)
        pred = self.gat(rf, edge_index, edge_attr, valid, fx[0])
        residual = nodes - pred
        residual[:, 0] = 0
        
        # pred_residual = self.residual_gat(
        #     residual.unsqueeze(-1), edge_index, edge_attr, valid, fx[0]
        # )
        pred_residual = self.residual_mlp(
            earray[:res_idx][:, :19], residual[1:res_idx+1],
        )
        # pred_residual = forward_distance(
        #     earray[:res_idx], residual[1:res_idx+1]
        # )

        adjusted_nodes = pred[:1] + pred_residual
        
        valid = valid.squeeze(-1)
        original_valid = valid.clone()

        return (
            adjusted_nodes, pred[:res_idx],
            pred_coarse[:res_idx], original_valid
        )
    
    def find_n_neighbors(self, r, rf):
        res_idx = rf.shape[0]
        for i in range(rf.shape[0]):
            if torch.sum(r[0, i]) < 0.01:
                res_idx = i
                break
        return res_idx
    

def distance(coord1: torch.Tensor,
             coord2: torch.Tensor
             ) -> torch.Tensor:
    """Distance matrix between two sets of points

    Calculate the pairwise distance between two sets of locations.

    Parameters:
        coord1:
            The nxd coordinates array for the first set.
        coord12:
            The nxd coordinates array for the second set.

    Returns:
        dist:
            The distance matrix.
    """
    if coord1.ndim == 1:
        m = 1
        coord1 = coord1.unsqueeze(0)
    else:
        m = coord1.shape[0]
    if coord2.ndim == 1:
        n = 1
        coord2 = coord2.unsqueeze(0)
    else:
        n = coord2.shape[0]

    #### Can improve (resolved)
    coord1 = coord1.unsqueeze(0)
    coord2 = coord2.unsqueeze(1)
    dists = torch.sqrt(torch.sum((coord1 - coord2) ** 2, axis=-1))
    return dists