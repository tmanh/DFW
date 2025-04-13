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
    def __init__(self, in_dim, n_layers, n_dim, fmts, n_heads=4):
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
                    AttentionBlock(dim=n_dim, n_heads=n_heads),  # Multi-Head Attention Layer
                    nn.Dropout(p=0.1),
                ]
            )

        if 'wo' in fmts:
            self.out_dim = 2

        self.dist = nn.Sequential(*modules)
        
        self.out = nn.Linear(in_features=n_dim, out_features=1)
        self.weight = nn.Linear(in_features=n_dim, out_features=1)

    def forward_wo(self, xs):
        outs, weights = self.step_wo(xs)
        return torch.sum(outs * weights, dim=1, keepdim=True)
    
    def forward_dwo(self, xs, x):
        outs, weights = self.step_wo(xs)
        return torch.sum((x + outs) * weights, dim=1, keepdim=True)
    
    def forward_do(self, xs, x):
        B, N, C = xs.shape
        outs = self.step(xs)
        return x + outs.view(B, N)
    
    def forward_w(self, xs, x):
        weights = self.step(xs)
        return torch.sum(
            x * torch.softmax(weights, dim=1),
            dim=1,
            keepdim=True
        )

    def forward_o(self, xs):
        return self.step(xs)

    def forward(self, xs, x, inputs, train, stage):
        if train:
            if self.fmts == 'w':
                return self.forward_w(xs, x)

            if 'i' not in self.fmts:
                if 'd' in self.fmts:
                    if stage == 1:
                        return self.forward_do(xs, x)
                    elif stage == 2:
                        return self.forward_dwo(xs, x)
                else:
                    if stage == 1:
                        return self.forward_o(xs)
                    elif stage == 2:
                        return self.forward_wo(xs)

            if 'i' in self.fmts:
                return self.forward_o(xs)
        else:
            if self.fmts == 'w':
                return self.forward_w(xs, x)
            elif self.fmts == 'o':
                return self.forward_o(xs)
            elif self.fmts == 'wo':
                return self.forward_wo(xs)
            elif self.fmts == 'io':
                return self.forward_io(xs, x, inputs)
            elif self.fmts == 'dwo':
                return self.forward_dwo(xs, x)
            elif self.fmts == 'dio':
                return self.forward_dio(xs, x, inputs)
            else:
                raise ValueError('Invalid format')
        
    def forward_dio(self, xs, x, inputs):
        outs, weights = self.step_wo(xs)
        ixs = get_i_inputs(xs, outs, inputs)
        iouts, _ = self.step_wo(ixs)

        diff = torch.abs(torch.abs(outs) - torch.abs(iouts))
        diff = F.softmax(1 / (diff + 1e-7), dim=1)

        return torch.sum((x + outs) * diff, dim=1, keepdim=True)

    def forward_io(self, xs, x, inputs):
        outs = self.forward_o(xs)
        ixs = get_i_inputs(xs, outs, inputs)
        iouts = self.forward_o(ixs)

        diff = torch.abs(x - iouts) / torch.abs(x)
        diff = F.softmax(1 / (diff + 1e-7), dim=1)

        return torch.sum(outs * diff, dim=1, keepdim=True)
    
    def step_wo(self, xs):
        B, N, C = xs.shape
        xs = xs.view(B * N, self.in_dim)

        feats = self.in_mlp(xs).view(B, N, -1)
        feats = self.dist(feats).view(B * N, -1)  # Pass through attention layers
        outs = self.out(feats)
        weights = self.weight(feats)
        
        weights = torch.softmax(weights.view(B, N), dim=1)

        return outs.view(B, N), weights
    
    def step(self, xs):
        B, N, C = xs.shape
        xs = xs.view(B * N, self.in_dim)

        feats = self.in_mlp(xs).view(B, N, -1)
        feats = self.dist(feats).view(B * N, -1)  # Pass through attention layers
        outs = self.out(feats)

        outs = outs.view(B, N)

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


class MLPAttention(nn.Module):
    def __init__(self, in_dim, n_layers, n_dim, fmts, n_heads=4):
        super().__init__()
        self.in_dim = in_dim
        self.n_dim = n_dim
        self.out_dim = 1
        self.fmts = fmts
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

        if 'wo' in fmts:
            self.out_dim = 2

        self.dist = nn.Sequential(*modules)

        self.attn = AttentionBlock(dim=n_dim, n_heads=n_heads)  # Multi-Head Attention Layer
        
        self.out = nn.Linear(in_features=n_dim, out_features=1)
        self.weight = nn.Linear(in_features=n_dim, out_features=1)

    def forward_wo(self, xs):
        outs, weights = self.step_wo(xs)
        return torch.sum(outs * weights, dim=1, keepdim=True)
    
    def forward_dwo(self, xs, x):
        outs, weights = self.step_wo(xs)
        return torch.sum((x + outs) * weights, dim=1, keepdim=True)
    
    def forward_do(self, xs, x):
        B, N, C = xs.shape
        outs = self.step(xs)
        return x + outs.view(B, N)
    
    def forward_w(self, xs, x):
        weights = self.step(xs)
        return torch.sum(
            x * torch.softmax(weights, dim=1),
            dim=1,
            keepdim=True
        )

    def forward_o(self, xs):
        return self.step(xs)

    def forward(self, xs, x, inputs, train, stage):
        if train:
            if self.fmts == 'w':
                return self.forward_w(xs, x)

            if 'i' not in self.fmts:
                if 'd' in self.fmts:
                    if stage == 1:
                        return self.forward_do(xs, x)
                    elif stage == 2:
                        return self.forward_dwo(xs, x)
                else:
                    if stage == 1:
                        return self.forward_o(xs)
                    elif stage == 2:
                        return self.forward_wo(xs)

            if 'i' in self.fmts:
                return self.forward_o(xs)
        else:
            if self.fmts == 'w':
                return self.forward_w(xs, x)
            elif self.fmts == 'o':
                return self.forward_o(xs)
            elif self.fmts == 'wo':
                return self.forward_wo(xs)
            elif self.fmts == 'io':
                return self.forward_io(xs, x, inputs)
            elif self.fmts == 'dwo':
                return self.forward_dwo(xs, x)
            elif self.fmts == 'dio':
                return self.forward_dio(xs, x, inputs)
            else:
                raise ValueError('Invalid format')
        
    def forward_dio(self, xs, x, inputs):
        outs, weights = self.step_wo(xs)
        ixs = get_i_inputs(xs, outs, inputs)
        iouts, _ = self.step_wo(ixs)

        diff = torch.abs(torch.abs(outs) - torch.abs(iouts))
        diff = F.softmax(1 / (diff + 1e-7), dim=1)

        return torch.sum((x + outs) * diff, dim=1, keepdim=True)

    def forward_io(self, xs, x, inputs):
        outs = self.forward_o(xs)
        ixs = get_i_inputs(xs, outs, inputs)
        iouts = self.forward_o(ixs)

        diff = torch.abs(x - iouts) / torch.abs(x)
        diff = F.softmax(1 / (diff + 1e-7), dim=1)

        return torch.sum(outs * diff, dim=1, keepdim=True)
    
    def step_wo(self, xs):
        B, N, C = xs.shape
        xs = xs.view(B * N, self.in_dim)

        feats = self.dist(xs)  # Pass through attention layers

        if not self.using_attention:
            feats = feats.view(B * N, -1)
        else:
            feats = self.attn(feats.view(B, N, -1)).view(B * N, -1)

        outs = self.out(feats)
        weights = self.weight(feats)
        
        weights = torch.softmax(weights.view(B, N), dim=1)

        return outs.view(B, N), weights
    
    def step(self, xs):
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

    def freeze(self):
        for param in self.dist.parameters():
            param.requires_grad = False
        for param in self.out.parameters():
            param.requires_grad = False
        self.dist.eval()

    def unfreeze(self):
        for param in self.dist.parameters():
            param.requires_grad = True
        for param in self.out.parameters(): 
            param.requires_grad = True
        self.dist.train()


def get_i_inputs(xs, o, inputs):
    if inputs == 'p':
        in_dim = 20
    elif inputs == 'pt':
        in_dim = 21
    elif inputs == 'ptn':
        in_dim = 22
    elif inputs == 'pte':
        in_dim = 28
    elif inputs == 'ptev':
        in_dim = 29
    elif inputs == 'shortv':
        in_dim = 8
    elif inputs == 'short':
        in_dim = 7
    elif inputs == 'update':
        o = o.unsqueeze(2)
        ixs = torch.cat(
            # [
            #     o,
            #     -xs[:, :, 1:2],  # dt
            #     xs[:, :, 3:4],  # w2 / 66
            #     xs[:, :, 2:3],  # w1 / 66
            #     xs[:, :, 4:6],  # distance, bidistance
            #     xs[:, :, 7:8], # h2
            #     xs[:, :, 6:7],  # h1
            #     xs[:, :, 8:],  # mean_w, std_w, max_zw
            # ], dim=2
            [
                o,
                -xs[:, :, 1:2],  # dt
                xs[:, :, 3:4],  # w2 / 66
                xs[:, :, 2:3],  # w1 / 66
                xs[:, :, 4:6],  # distance, bidistance
                -xs[:, :, 6:8], # *dp
                xs[:, :, 9:10], # h2
                xs[:, :, 8:9],  # h1
                xs[:, :, 10:],  # mean_w, std_w, max_zw
            ], dim=2
        )
    
    return ixs