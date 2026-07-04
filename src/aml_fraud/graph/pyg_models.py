from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GAE, GCNConv, SAGEConv


class GCNEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, latent_dim: int = 16):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        return self.conv2(self.conv1(x, edge_index).relu(), edge_index)


class GraphSAGEClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        h = self.conv1(x, edge_index).relu()
        h = self.conv2(h, edge_index).relu()
        return self.head(h).squeeze(-1)


def build_graph_autoencoder(input_dim: int, hidden_dim: int = 32, latent_dim: int = 16):
    return GAE(GCNEncoder(input_dim, hidden_dim, latent_dim))
