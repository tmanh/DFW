import torch
import torch.nn as nn
import torch.nn.functional as F

from model.common import Mlp

class AttentionBlock(nn.Module):
    def __init__(self, dim, n_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.ReLU(),
            nn.Linear(4 * dim, dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Multi-Head Attention
        attn_out, _ = self.attn(x, x, x)  # Self-attention (Q = K = V)
        x = self.norm1(x + self.dropout(attn_out))  # Add & Norm

        # Feedforward
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))  # Add & Norm
        return x


class Attention(nn.Module):
    def __init__(self, in_dim, n_layers, n_dim, n_heads=4):
        super().__init__()
        self.in_dim = in_dim
        self.n_dim = n_dim
        self.out_dim = 1

        self.in_mlp = nn.Sequential(
            nn.Linear(in_dim, n_dim),
            nn.ReLU(),
        )
        
        modules = []
        for _ in range(n_layers):
            modules.extend(
                [
                    AttentionBlock(dim=n_dim, n_heads=n_heads),  # Multi-Head Attention Layer
                    nn.Dropout(p=0.1),
                ]
            )

        self.dist = nn.Sequential(*modules)
        
        self.out = nn.Linear(in_features=n_dim, out_features=1)
        self.weight = nn.Linear(in_features=n_dim, out_features=1)

    def forward(self, xs, x, valid):
        B, N, C = xs.shape
        xs = xs.view(B * N, self.in_dim)

        feats = self.in_mlp(xs).view(B, N, -1)
        feats = self.dist(feats).view(B * N, -1)  # Pass through attention layers
        outs = self.out(feats)

        outs = outs.view(B, N)

        return outs


class MLPAttention(nn.Module):
    def __init__(self, in_dim, n_layers, n_dim, n_heads=4):
        super().__init__()
        self.in_dim = in_dim
        self.n_dim = n_dim
        self.out_dim = 1
        self.using_attention = False
        
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

        self.dist = nn.Sequential(*modules)

        self.attn = AttentionBlock(dim=n_dim, n_heads=n_heads)  # Multi-Head Attention Layer
        
        self.out = nn.Linear(in_features=n_dim, out_features=1)
        self.weight = nn.Linear(in_features=n_dim, out_features=1)

    def forward(self, xs, x, valid):
        B, N, C = xs.shape
        xs = xs.view(B * N, self.in_dim)

        feats = self.dist(xs)  # Pass through attention layers
        if not self.using_attention:
            outs = self.out(feats.view(B * N, -1))
        else:
            feats = self.attn(feats.view(B, N, -1)).view(B * N, -1)
            outs = self.out(feats)

        outs = outs.view(B, N)

        return outs
