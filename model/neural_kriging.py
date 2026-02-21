import torch
import torch.nn as nn
import torch.nn.functional as F

from model.common import Mlp


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

        self.weight = nn.Linear(in_features=n_dim, out_features=1)
        self.out = nn.Linear(in_features=n_dim, out_features=1)

    def forward(self, xs):
        feats = self.dist(xs)
        weights = self.weight(feats)
        weights = torch.relu(weights)
        alpha = weights / (torch.sum(weights, dim=0, keepdim=True) + 1e-8)
        return alpha


class LatentKriging1Target(nn.Module):
    def __init__(self, in_dim, n_layers, n_dim):
        super().__init__()
        self.net_x = MLP(in_dim, n_layers, n_dim)  # outputs (K,) or (B,K)
        self.net_y = MLP(in_dim, n_layers, n_dim)

        # positive params via softplus
        self.log_sigma = nn.Parameter(torch.tensor(0.0))
        self.log_lx    = nn.Parameter(torch.tensor(0.0))
        self.log_ly    = nn.Parameter(torch.tensor(0.0))
        self.log_tau   = nn.Parameter(torch.tensor(-3.0))  # nugget

    def forward(self, feats_neighbors, Y):  # feats_neighbors: (K, ...) ; Y: (K, T)
        x = self.net_x(feats_neighbors).squeeze(-1)  # (K,)
        y = self.net_y(feats_neighbors).squeeze(-1)  # (K,)

        sigma2 = F.softplus(self.log_sigma)**2
        lx = F.softplus(self.log_lx) + 1e-6
        ly = F.softplus(self.log_ly) + 1e-6
        tau2 = F.softplus(self.log_tau)**2

        # pairwise squared distances (anisotropic)
        dx2 = (x[:, None] - x[None, :])**2 / (lx**2)
        dy2 = (y[:, None] - y[None, :])**2 / (ly**2)
        d2 = dx2 + dy2  # (K,K)

        Kmat = sigma2 * torch.exp(-d2) + tau2 * torch.eye(len(x), device=x.device)

        # cross-cov to p0=(0,0)
        d20 = (x**2)/(lx**2) + (y**2)/(ly**2)  # (K,)
        kvec = sigma2 * torch.exp(-d20)         # (K,)

        # ordinary kriging augmented solve
        K = len(x)
        ones = torch.ones(K, device=x.device)

        A = torch.zeros(K+1, K+1, device=x.device)
        A[:K,:K] = Kmat
        A[:K, K] = ones
        A[K, :K] = ones

        b = torch.zeros(K+1, device=x.device)
        b[:K] = kvec
        b[K] = 1.0

        sol = torch.linalg.solve(A, b)
        w = sol[:K]  # (K,)

        y_hat = (w[:, None] * Y).sum(dim=0)  # (T,)
        return y_hat
