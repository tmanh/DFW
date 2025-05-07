import torch
import torch.nn as nn

import torch

from torch_geometric.utils import softmax
from torch_geometric.nn.conv import MessagePassing


from model.gru import minGRU

class EdgeAttrGNNLayer(MessagePassing):
    def __init__(self, in_channels, edge_dim, out_channels):
        super().__init__(aggr='add')  # or 'mean', 'max'

        self.lin_edge = nn.Sequential(
            torch.nn.Linear(edge_dim, out_channels),
            torch.nn.GELU(),
            torch.nn.Linear(out_channels, out_channels),
            torch.nn.ReLU(inplace=True)
        )
        self.gru = minGRU(dim=out_channels)
        self.weight_mlp = torch.nn.Linear(out_channels, 1)

    def forward(self, x, edge_index, edge_attr, valid):
        # x: [num_nodes, in_channels]
        # edge_index: [2, num_edges]
        # edge_attr: [num_edges, edge_dim]
        # x = self.lin_node(x)
        edge_attr = self.lin_edge(edge_attr)
        x = torch.cat([x, valid.squeeze(-1)], dim=-1)
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr, index):
        # x_j: features of source nodes
        edge_attr = edge_attr.unsqueeze(0)
        edge_attr = self.gru(edge_attr)
        edge_attr = edge_attr.squeeze(0)
        edge_attr = self.weight_mlp(edge_attr)
        edge_attr = torch.mean(x_j[:, x_j.shape[-1]//2:], dim=-1, keepdim=True) * edge_attr
        
        weights = softmax(edge_attr, index)  # shape: [num_edges, out_channels]

        return x_j * weights  # element-wise multiplication info

    def update(self, aggr_out):
        return aggr_out  # self.lin_update(aggr_out)


class GATWithEdgeAttr(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=1):
        super(GATWithEdgeAttr, self).__init__()
        hidden_channels = 16
        edge_dim = 18
        self.hidden_channels = hidden_channels
        self.in_mlp = nn.Sequential(
            nn.Linear(1, hidden_channels),
            nn.GELU(),
        )
        self.fuse_mlp = nn.Linear(3, 1)
        self.gat = EdgeAttrGNNLayer(hidden_channels, edge_dim=edge_dim, out_channels=hidden_channels)
        self.out_mlp = nn.Sequential(
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, nodes, edge_index, edge_attr, valid):
        with torch.no_grad():
            valid = valid.squeeze(0).unsqueeze(-1)
            not_valid = 1 - valid
            nodes = nodes * valid
        
        N, L, C = nodes.shape
        nodes = nodes.view(N, -1)

        not_valid = not_valid.repeat(1, 1, C)
        not_valid_mask = not_valid.view(N, -1)
        for _ in range(3):
            new_nodes_values = self.gat(nodes, edge_index, edge_attr, valid)
            new_nodes = new_nodes_values[:, :new_nodes_values.shape[-1]//2].view(N, L, C)
            new_nodes = torch.cat([new_nodes, nodes.view(N, L, C), not_valid], dim=-1)
            mask = torch.sigmoid(self.fuse_mlp(new_nodes).view(N, -1))
            
            nodes = (1 - mask) * nodes + not_valid_mask * mask * new_nodes_values[:, :new_nodes_values.shape[-1]//2]
        return nodes[0].unsqueeze(0).squeeze(-1)