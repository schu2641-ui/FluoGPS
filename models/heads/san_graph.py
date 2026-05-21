import torch
import torch.nn as nn

import torch_geometric.graphgym.register as register
from torch_geometric.graphgym import cfg
from torch_geometric.nn import global_mean_pool

class SANGraphHead(nn.Module):
    """
    SAN prediction head for graph prediction tasks.

    Args:
        dim_in (int): Input dimension.
        dim_out (int): Output dimension. For binary prediction, dim_out=1.
        L (int): Number of hidden layers.
    """

    def __init__(self, dim_in, dim_out, L=2):
        super().__init__()
        self.pooling_fun = global_mean_pool
        mean_pooling_dim = dim_in * 3
        list_FC_layers = [
            nn.Linear(mean_pooling_dim // 2 ** l,
                      mean_pooling_dim // 2 ** (l + 1),
                      bias=True)
            for l in range(L)]
        list_FC_layers.append(
            nn.Linear(mean_pooling_dim // 2 ** L, dim_out, bias=True))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L
        self.activation = nn.ReLU()

    def _apply_index(self, batch):
        return batch.graph_feature, batch.y

    def forward(self, batch):
        solute_mask = batch.role == 0
        solvent_mask = batch.role == 1

        solute_emb = global_mean_pool(
            batch.x[solute_mask],
            batch.batch[solute_mask],
            size=batch.num_graphs,
        )
        solvent_emb = global_mean_pool(
            batch.x[solvent_mask],
            batch.batch[solvent_mask],
            size=batch.num_graphs,
        )
        graph_emb = self.pooling_fun(
            batch.x,
            batch.batch,
            size=batch.num_graphs,
        )
        graph_emb = torch.cat([solute_emb, solvent_emb, graph_emb], dim=-1)

        for l in range(self.L):
            graph_emb = self.FC_layers[l](graph_emb)
            graph_emb = self.activation(graph_emb)
        graph_emb = self.FC_layers[self.L](graph_emb)
        batch.graph_feature = graph_emb
        pred, label = self._apply_index(batch)
        return pred, label
