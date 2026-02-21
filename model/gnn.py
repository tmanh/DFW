import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.utils import softmax
from torch_geometric.nn.conv import MessagePassing

from model.common import Mlp
from model.gru import minGRU
from model.neural_kriging import LatentKriging1Target


class fullGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=True, dropout=0.0, bidirectional=False):
        super(fullGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)

    def forward(self, x):
        # x: [B, T, C]
        out, _ = self.gru(x)  # [B, T, 2*H]
        return out


class EdgeAttrGNNLayer(MessagePassing):
    def __init__(self, case, edge_dim, out_channels=8):
        super().__init__(aggr='add')  # or 'mean', 'max'

        self.case = case
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
            if self.case != "mlp":
                self.post_update = nn.Sequential(
                    # fullGRU(8, 8, batch_first=True),
                    # torch.nn.Tanh(),
                    minGRU(8),
                    nn.Linear(8, 1),
                )
            else:
                self.post_update = nn.Sequential(
                    nn.Linear(8, 8),
                    torch.nn.GELU(),
                    nn.Linear(8, 1),
                )
        else:
            self.post_update = nn.Identity()

    def freeze(self):
        self.post_update.eval()
        for param in self.post_update.parameters():
            param.requires_grad_ = False

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


class RainModel(torch.nn.Module):
    def __init__(self):
        super(RainModel, self).__init__()
        
        self.rain_mlp = nn.Sequential(
            nn.Linear(1, 8),
            nn.GELU(),
            nn.Linear(8, 8),
        )

    def forward(self, r, pos):
        # r: [N, L, 1]
        # pos: [N, L, 2]
        r = self.rain_mlp(r.squeeze(0).unsqueeze(-1)).squeeze(-1)
        return r
    

def centered_cosine(a, b, eps=1e-8):
    # a,b: (K, D)
    a = a - a.mean(dim=-1, keepdim=True)
    b = b - b.mean(dim=-1, keepdim=True)
    return (a*b).sum(-1) / (a.norm(dim=-1)*b.norm(dim=-1) + eps)  # (K,)


def consensus_score(series, eps=1e-8):
    """
    series: (K,T)
    returns cons: (K,1) in [0,1] approximately
    """
    K, T = series.shape
    z = series - series.mean(dim=1, keepdim=True)
    z = z / (series.std(dim=1, keepdim=True) + eps)
    C = (z @ z.t()) / T                 # approx corr matrix (K,K)
    cons = C.abs().mean(dim=1, keepdim=True)  # (K,1)
    cons = (cons - cons.min()) / (cons.max() - cons.min() + eps)
    return cons


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

    def forward(self, xs, x, return_alpha=False):
        feats = self.dist(xs)
        weights = self.weight(feats)
        weights = torch.relu(weights)
        # alpha = sparsemax(weights, dim=0)      # (K,1), nonneg, sums to 1
        alpha = weights / (torch.sum(weights, dim=0, keepdim=True) + 1e-8)
        
        out = torch.sum(x * alpha, dim=0, keepdim=True)

        if return_alpha:
            return out, alpha
        
        return out

class StationLevelGate(nn.Module):
    def __init__(self, mlpw: MLPW, init_theta=0.2, init_s=10.0):
        super().__init__()
        self.mlpw = mlpw
        self.theta = nn.Parameter(torch.tensor(init_theta))
        self.log_s = nn.Parameter(torch.log(torch.tensor(init_s)))

    def forward(self, xs, raw_nb, pred_nb, pred0):
        """
        xs:     (K, F)  e.g. earray[:K, :3]
        raw_nb: (K, D)  neighbor observed water level (flattened)
        pred_nb:(K, D)  rainfall-based prediction at neighbors
        pred0:  (1, D)  rainfall-based prediction at virtual
        """
        # weights from distance features only (your original idea)
        _, alpha = self.mlpw(xs, raw_nb, return_alpha=True)  # alpha: (K,1)

        # residual at neighbors
        res_nb = raw_nb - pred_nb  # (K,D)

        # per-station "rain works here" score
        rho = centered_cosine(pred_nb, raw_nb)  # (K,)

        s = F.softplus(self.log_s) + 1e-6
        g = torch.sigmoid(s * (rho - self.theta))  # (K,) in [0,1]

        # two absolute candidates per station
        v_res = pred0 + res_nb                 # (K,D) via broadcast of pred0
        v_dir = raw_nb                         # (K,D)

        # station-level gate
        v = g[:, None] * v_res + (1.0 - g)[:, None] * v_dir  # (K,D)

        # final weighted average
        y0 = torch.sum(v * alpha, dim=0, keepdim=True)  # (1,D)

        return y0, alpha, g, rho


class GATWithEdgeAttrRain(torch.nn.Module):
    def __init__(self, hidden_channels=48, edge_dim = 12):
        super(GATWithEdgeAttrRain, self).__init__()

        # full, idw, mlp, only, kriging, split
        self.case = "mlp"

        edge_dim = 3
        self.hidden_channels = hidden_channels
        self.gat = EdgeAttrGNNLayer(self.case, edge_dim=edge_dim, out_channels=8)

        self.residual_mlp = MLPW(12, 1, 8)
        self.rain_mlp = RainModel()

        if self.case == "kriging":
            self.neural_kriging = LatentKriging1Target(12, 2, 8)

        if self.case == 'split':
            self.station_gate = StationLevelGate(self.residual_mlp)

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

        if self.case == 'split':
            edge_attr = edge_attr[:, :3]
            pred = self.gat(rf, edge_index, edge_attr, valid, fx[0])

            K = res_idx
            xs = earray[:K]          # (K,3) distance/displacement
            raw_nb = nodes[1:K+1]        # (K,D)
            pred_nb = pred[1:K+1]        # (K,D)
            pred0 = pred[:1]             # (1,D)

            adjusted_nodes, alpha, g, rho = self.station_gate(xs, raw_nb, pred_nb, pred0)

            valid = valid.squeeze(-1)
            original_valid = valid.clone()
            return (
                adjusted_nodes, pred[:res_idx],
                pred_coarse[:res_idx], original_valid
            )
        elif self.case != 'only':
            edge_attr = edge_attr[:, :3]
            pred = self.gat(rf, edge_index, edge_attr, valid, fx[0])

            K = res_idx
            raw_nb = nodes[1:K+1]       # (K,T)
            pred_nb = pred[1:K+1]       # (K,T)
            pred0   = pred[:1]
            res_nb  = raw_nb - pred_nb  # (K,T)

            xs = earray[:res_idx]

            if self.case in ["full", "mlp"]:
                pred_residual = self.residual_mlp(
                    xs, res_nb
                )  # pred_residual: (1,T), alpha: (K,1)
                adjusted_nodes = pred[:1] + pred_residual
            elif self.case == "idw":
                pred_residual = forward_distance(
                    xs, res_nb
                )
                adjusted_nodes = pred[:1] + pred_residual
            elif self.case == 'kriging':
                pred_residual = self.neural_kriging(
                    xs, res_nb
                )
                adjusted_nodes = pred[:1] + pred_residual

            valid = valid.squeeze(-1)
            original_valid = valid.clone()
            return (
                adjusted_nodes, pred[:res_idx],
                pred_coarse[:res_idx], original_valid
            )
        else:
            valid = valid.squeeze(-1)
            original_valid = valid.clone()
            return pred_coarse[:1].squeeze(-1), pred_coarse[:res_idx].squeeze(-1), pred_coarse[:res_idx], original_valid

    def find_n_neighbors(self, r, rf):
        res_idx = rf.shape[0]
        for i in range(rf.shape[0]):
            if torch.sum(r[0, i]) < 0.01:
                res_idx = i
                break
        return res_idx
    

def sparsemax(logits, dim=0):
    # logits: (K,1) or (K,)
    z = logits.squeeze(-1)
    z = z - z.max(dim=dim, keepdim=True).values
    z_sorted, _ = torch.sort(z, descending=True)
    k = torch.arange(1, z.numel()+1, device=z.device, dtype=z.dtype)
    z_cumsum = torch.cumsum(z_sorted, dim=0)
    
    # find k(z)
    cond = 1 + k * z_sorted > z_cumsum
    k_z = torch.max(k[cond]).long()
    tau = (z_cumsum[k_z-1] - 1) / k_z
    p = torch.clamp(z - tau, min=0)
    
    return p.unsqueeze(-1)


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