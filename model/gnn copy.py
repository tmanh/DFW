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


def time_varying_conv1d(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    Causal time-varying FIR.
    x: [B, T, C]  (signal)
    w: [B, T, K]  (kernel at each time step; shared across channels)
    returns y: [B, T, C]  (same length)
    """
    B, T, C = x.shape
    Bw, Tw, K = w.shape
    assert B == Bw and T == Tw, "Batch/time dims must match"

    # causal same-length: pad K-1 on the left so each t uses x[t-K+1 ... t]
    x_pad = F.pad(x.permute(0, 2, 1), (K-1, 0))      # [B, C, T+K-1]
    # make sliding windows of length K along time
    x_win = x_pad.unfold(dimension=-1, size=K, step=1)  # [B, C, T, K]

    # w is [B, T, K] -> broadcast over channels: [B, 1, T, K]
    w_exp = w.unsqueeze(1)                               # [B, 1, T, K]

    # dot product over the K dimension -> [B, C, T]
    y = (x_win * w_exp).sum(dim=-1)                      # [B, C, T]
    return y.permute(0, 2, 1)                            # [B, T, C]


class HyperFIR(nn.Module):
    """
    Generate a per-sample FIR kernel from static features z,
    then convolve with x_main.
    """
    def __init__(self):
        super().__init__()

        self.k_len = 9
        self.pos_enc = minGRU(8, expansion_factor=2)
        self.mlp = nn.Sequential(
            nn.Linear(10, 8),
            nn.GELU(),
            nn.Linear(8, 8),
        )

        self.kernel = nn.Sequential(
            minGRU(8, expansion_factor=2),
            nn.GELU(),
            nn.Linear(8, self.k_len),
        )

    def forward(self, x_main: torch.Tensor, zi: torch.Tensor, zj: torch.Tensor):
        """
        x_main: [B,8,T], z: [B,S]
        returns: y_lin [B,1,T]
        """
        x_main = x_main.view(x_main.shape[0], -1, 8)  # [B,1,T]

        feats = self.mlp(torch.cat([zi[:, -5:], zj[:, -5:]], dim=-1))                   # [B,8]
        pos = self.pos_enc(x_main)                    # [B, K]

        feats = feats.unsqueeze(1) + pos                           # [B, K]

        w = self.kernel(feats)           # [B, 1, K]
        w = F.normalize(w, dim=-1)
        # w = w / (w.abs().sum(dim=-1, keepdim=True) + 1e-6)

        y = time_varying_conv1d(x_main, w)
        return y


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
        # self.rain_mlp = nn.Sequential(
        #     nn.Linear(1, 8),
        #     nn.GELU(),
        #     minGRU(8),
        # )
        # self.rain_mlp = nn.Sequential(
        #     nn.Linear(1, 8),
        #     nn.GELU(),
        #     nn.GRU(8, 8, batch_first=True)
        # )

        self.rain_pos_enc = nn.Sequential(
            nn.Linear(2, 8),
            nn.GELU(),
            nn.Linear(8, 8),
        )

    def forward(self, r, pos):
        # r: [N, L, 1]
        # pos: [N, L, 2]
        r = self.rain_mlp(r.squeeze(0).unsqueeze(-1)).squeeze(-1)
        # r = self.rain_mlp(r.squeeze(0).unsqueeze(-1))[0].contiguous()
        # pos_enc = self.rain_pos_enc(pos[:, :, -2:].squeeze(0)).unsqueeze(1).squeeze(-1)
        # r = r * pos_enc
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
    

def gaussian_variogram(h, nugget=0.0, sill=1.0, range_=1.0):
    return nugget + sill * (1.0 - torch.exp(- (h ** 2) / (range_ ** 2)))


def pairwise_distances(x1, x2):
    x1_sq = torch.sum(x1**2, dim=1, keepdim=True)          # [N, 1]
    x2_sq = torch.sum(x2**2, dim=1, keepdim=True)          # [M, 1]
    dist_sq = x1_sq - 2 * (x1 @ x2.T) + x2_sq.T            # [N, M] -> if x1 is train, x2 is pred, transpose accordingly
    return torch.sqrt(torch.clamp(dist_sq, min=1e-12))


def ordinary_kriging(train_coords, train_values, pred_coords, variogram_fn, eps=1e-8):
    """
    Ordinary Kriging (vectorized over prediction points).
    Shapes:
      train_coords: [N, D]
      train_values: [N] or [N, 1]
      pred_coords : [M, D]
    Returns:
      preds: [M]
    """
    # Ensure proper shapes/dtypes/devices
    device = train_coords.device
    dtype = train_coords.dtype
    train_coords = train_coords.reshape(train_coords.shape[0], -1).to(device=device, dtype=dtype)
    pred_coords  = pred_coords.reshape(pred_coords.shape[0],  -1).to(device=device, dtype=dtype)

    N = train_coords.shape[0]
    M = pred_coords.shape[0]

    # Build (N+1)x(N+1) kriging matrix with Lagrange multiplier
    d_train = pairwise_distances(train_coords, train_coords)        # [N, N]
    gamma_tt = variogram_fn(d_train)                                # [N, N]

    K = torch.zeros((N+1, N+1), dtype=dtype, device=device)
    K[:N, :N] = gamma_tt
    K[N, :N] = 1.0
    K[:N, N] = 1.0
    # K[N, N] = 0.0  # already zero

    # Stable inverse
    K_inv = torch.linalg.inv(K + eps * torch.eye(N+1, device=device, dtype=dtype))

    # RHS for all M predictions at once
    d_pt = pairwise_distances(pred_coords, train_coords)            # [M, N]
    gamma_pt = variogram_fn(d_pt)                                   # [M, N]
    ones = torch.ones((M, 1), device=device, dtype=dtype)
    k = torch.cat([gamma_pt, ones], dim=1)                          # [M, N+1]

    # Weights for all M: (K_inv @ k_i) for each i -> [M, N+1]
    weights = (K_inv @ k.T).T                                       # [M, N+1]
    w = weights[:, :-1].squeeze(0).unsqueeze(-1)
    w = w / torch.sum(w, dim=0, keepdim=True)

    # Predictions: sum_j w_ij * z_j
    preds = (w * train_values).sum(dim=0, keepdim=True)              # [M]

    return preds


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
    

def krig_pred(w_train: torch.Tensor,
              coord_train: torch.Tensor,
              coord_test: torch.Tensor,
              theta: tuple[float, float, float],
              neighbor_size: Optional[int] = 20,
              q: Optional[float] = 0.95
              ) -> torch.Tensor:
    """Kriging prediction (Gaussian process regression) with confidence interval.

    Kriging prediction on testing locations based on the observations on the training locations. The kriging procedure
    assumes the observations are sampled from a Gaussian process, which is paramatrized here to have an exponential covariance
    structure using theta = [sigma^2, phi, tau]. NNGP appriximation is involved for efficient computation of matrix inverse.
    The conditional variance (kriging variance) is used to build the confidence interval using the quantiles (a/2, 1-a/2).
    (see https://arxiv.org/abs/2304.09157, section 4.3 for more details.)

    Parameters:
        w_train:
            Training observations of the spatial random effect without any fixed effect.
        coord_train:
            Spatial coordinates of the training observations.
        coord_test:
            Spatial coordinates of the locations for prediction
        theta:
            theta[0], theta[1], theta[2] represent sigma^2, phi, tau in the exponential covariance family.
        neighbor_size:
            The number of nearest neighbors used for NNGP approximation. Default being 20.
        q:
            Confidence coverage for the prediction interval. Default being 0.95.

    Returns:
        w_test: torch.Tensor
            The kriging prediction.
        pred_U: torch.Tensor
            Confidence upper bound.
        pred_L: torch.Tensor
            Confidence lower bound.

    See Also:
        Zhan, Wentao, and Abhirup Datta. 2024. “Neural Networks for Geospatial Data.”
        Journal of the American Statistical Association, June, 1–21. doi:10.1080/01621459.2024.2356293.
    """
    sigma_sq, phi, tau = theta
    tau_sq = tau * sigma_sq
    n_test = coord_test.shape[0]

    rank = make_rank(coord_train, neighbor_size, coord_test)

    w_test = torch.zeros(n_test)
    sigma_test = (sigma_sq + tau_sq) * torch.ones(n_test)
    for i in range(n_test):
        ind = rank[i, :]
        cov_sub = make_cov_full(distance(coord_train[ind, :], coord_train[ind, :]), theta, nuggets=True)
        cov_vec = make_cov_full(distance(coord_train[ind, :], coord_test[i, :]), theta, nuggets=False).reshape(-1)
        bi = torch.linalg.solve(cov_sub, cov_vec)
        w_test[i] = torch.dot(bi.T, w_train[ind]).squeeze()
        sigma_test[i] = sigma_test[i] - torch.dot(bi.reshape(-1), cov_vec)
    p = scipy.stats.norm.ppf((1 + q) / 2, loc=0, scale=1)
    sigma_test = torch.sqrt(sigma_test)
    pred_U = w_test + p * sigma_test
    pred_L = w_test - p * sigma_test

    return w_test, pred_U, pred_L


def make_cov_full(dist: torch.Tensor | np.ndarray,
                  theta: tuple[float, float, float],
                  nuggets: Optional[bool] = False,
                  ) -> torch.Tensor | np.ndarray:
    """Compose covariance matrix from the distance matrix with dense representation.

    Compose a covariance matrix in the exponential covariance family (other options to be implemented) from the distance
    matrix. The returned object class depends on the input distance matrix.

    Parameters:
        dist:
            The nxn distance matrix
        theta:
            theta[0], theta[1], theta[2] represent sigma^2, phi, tau in the exponential covariance family.
        nuggets:
            Whether to include nuggets term in the covariance matrix (added to the diagonal).

    Returns:
        cov:
            A covariance matrix.
    """
    sigma_sq, phi, tau = theta
    tau_sq = tau * sigma_sq
    if isinstance(dist, float) or isinstance(dist, int):
        dist = torch.Tensor(dist)
        n = 1
    else:
        n = dist.shape[-1]
    if isinstance(dist, torch.Tensor):
        cov = sigma_sq * torch.exp(-phi * dist)
    else:
        cov = sigma_sq * np.exp(-phi * dist)
    if nuggets:
        shape_temp = list(cov.shape)[:-2] + [1 ,1]
        if isinstance(dist, torch.Tensor):
            cov += tau_sq * torch.eye(n).repeat(*shape_temp).squeeze()
        else:
            cov += tau_sq * np.eye(n).squeeze() #### need improvement
    return cov


def make_rank(coord: torch.Tensor,
              neighbor_size: int,
              coord_ref = None
              ) -> np.ndarray:
    """Compose the nearest neighbor index list based on the coordinates.

    Find the indexes of nearest neighbors in reference set for each location i in the main set.
    The index is based on the increasing order of the distances between ith location and the locations in the reference set.

    Parameters:
        coord:
            The nxd coordinates array of target locations.
        neighbor_size:
        `   Suppose neighbor_size = k, only the top k-nearest indexes will be returned.
        coord_ref:
            The n_refxd coordinates array of reference locations. If None, use the target set itself as the reference.
            (Any location's neighbor does not include itself.)

    Returns:
        rank_list:
            A nxp array. The ith row is the indexes of the nearest neighbors for the ith location, ordered by the distance.
    """
    if coord_ref is None:
        neighbor_size += 1

    knn = NearestNeighbors(n_neighbors=neighbor_size)
    knn.fit(coord.detach().numpy())
    if coord_ref is None:
        coord_ref = coord
        rank = knn.kneighbors(coord_ref)[1]
        return rank[:, 1:]
    else:
        rank = knn.kneighbors(coord_ref)[1]
        return rank[:, 0:]


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