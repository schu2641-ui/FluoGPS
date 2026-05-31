from __future__ import annotations

import torch
from torch_geometric.utils import scatter, to_dense_adj
from torch_geometric.utils.num_nodes import maybe_num_nodes


DEFAULT_RWSE_STEPS = list(range(1, 17))


def normalize_kernel(kernel=None):
    return DEFAULT_RWSE_STEPS.copy() if kernel is None else list(kernel)


class RWSETransform:
    def __init__(self, kernel=None):
        self.rwse_steps = normalize_kernel(kernel)

    def __call__(self, data):
        if hasattr(data, "solute_edge_index") and hasattr(data, "solvent_edge_index"):
            data.solute_pestat_RWSE = get_rw_landing_probs(
                ksteps=self.rwse_steps,
                edge_index=data.solute_edge_index,
                num_nodes=data.solute_x.size(0),
            )
            data.solvent_pestat_RWSE = get_rw_landing_probs(
                ksteps=self.rwse_steps,
                edge_index=data.solvent_edge_index,
                num_nodes=data.solvent_x.size(0),
            )
            return data

        data.pestat_RWSE = get_rw_landing_probs(
            ksteps=self.rwse_steps,
            edge_index=data.edge_index,
            num_nodes=data.num_nodes,
        )
        return data

    def __repr__(self):
        return f"{self.__class__.__name__}(kernel={self.rwse_steps})"


def create_rwse_transform(kernel=None):
    return RWSETransform(kernel=kernel)


def get_rw_landing_probs(ksteps, edge_index, edge_weight=None, num_nodes=None, space_dim=0):
    """Compute random-walk landing probabilities for the provided steps."""
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)

    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    source = edge_index[0]
    deg = scatter(edge_weight, source, dim=0, dim_size=num_nodes, reduce="sum")
    deg_inv = deg.pow(-1.0)
    deg_inv.masked_fill_(deg_inv == float("inf"), 0)

    if edge_index.numel() == 0:
        transition = edge_weight.new_zeros((num_nodes, num_nodes))
    else:
        adjacency = to_dense_adj(
            edge_index,
            edge_attr=edge_weight,
            max_num_nodes=num_nodes,
        ).squeeze(0)
        transition = torch.diag(deg_inv) @ adjacency

    rw_steps = normalize_kernel(ksteps)
    start_step = min(rw_steps)
    end_step = max(rw_steps)
    consecutive = rw_steps == list(range(start_step, end_step + 1))
    landings = []

    if consecutive:
        transition_k = transition.matrix_power(start_step)
        for step in range(start_step, end_step + 1):
            landing = torch.diagonal(transition_k, 0).unsqueeze(0)
            landings.append(landing * (step ** (space_dim / 2)))
            transition_k = transition_k @ transition
    else:
        for step in rw_steps:
            landing = torch.diagonal(transition.matrix_power(step), 0).unsqueeze(0)
            landings.append(landing * (step ** (space_dim / 2)))

    return torch.cat(landings, dim=0).transpose(0, 1)
