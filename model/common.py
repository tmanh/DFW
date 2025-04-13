import torch
import torch.nn as nn


class Mlp(nn.Module):
    def __init__(self, in_features, out_features=None, act_layer=nn.ReLU, add_noise=False):
        super().__init__()
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, out_features)
        self.act = act_layer(inplace=True)
        self.add_noise = add_noise

    def forward(self, x):
        x = self.fc1(x)
        if self.add_noise:
            x += torch.randn_like(x) * 0.1
        x = self.act(x)
        return x
    
    