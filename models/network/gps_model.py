import torch
from torch_geometric.graphgym.models.encoder import BondEncoder, AtomEncoder
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool

from models.layers.GPS_layer import GPSLayer
from models.encoders.rwse_encoder import RWSEEncoder
from models.heads.san_graph import SANDualGraphHead, SANGraphHead


class FeatureEncoder(torch.nn.Module):
    """
    Encoding node and edge features

    Args:
        dim_in (int): Input feature dimension (not used for AtomEncoder, kept for compatibility)
        dim_emb (int): Embedding dimension
        rwse_steps (list): RWSE kernel steps
    """
    def __init__(self, dim_emb, rwse_steps=None):
        super(FeatureEncoder, self).__init__()
        # RWSE takes 20 dimensions, so AtomEncoder gets the rest
        dim_pe = 20  # RWSE dimension
        role_dim_out = 8 # RoleEncoder output dimension (assuming 2 roles)
        self.role_encoder = torch.nn.Embedding(2, role_dim_out)  # Assuming 2 roles: 0 and 1
        atom_dim_out = dim_emb - dim_pe - role_dim_out  # AtomEncoder output dimension
        # Encode integer node features via nn.Embeddings (Long -> Float)
        self.atom_encoder = AtomEncoder(atom_dim_out)
        # Encode role features via nn.Embeddings (Long -> Float)
        # Add RWSE positional encoding (expand_x=False since x is already encoded)
        self.rwse_encoder = RWSEEncoder(dim_in=atom_dim_out + role_dim_out, 
                                        dim_emb=dim_emb,
                                        rwse_steps=rwse_steps, 
                                        expand_x=False)
        # Encode integer edge features via nn.Embeddings
        self.edge_encoder = BondEncoder(dim_emb)

    def forward(self, batch):
        # First encode atom features (Long -> Float)
        batch = self.atom_encoder(batch)
        # Then add RWSE positional encoding
        role_emb = self.role_encoder(batch.role)  # Encode role features
        batch.x = torch.cat([batch.x, role_emb], dim=-1)  #
        batch = self.rwse_encoder(batch)
        # Finally encode edge features
        batch = self.edge_encoder(batch)
        return batch


class BranchFeatureEncoder(torch.nn.Module):
    def __init__(self, dim_emb, rwse_steps=None):
        super().__init__()
        dim_pe = 20
        atom_dim_out = dim_emb - dim_pe
        self.atom_encoder = AtomEncoder(atom_dim_out)
        self.rwse_encoder = RWSEEncoder(
            dim_in=atom_dim_out,
            dim_emb=dim_emb,
            rwse_steps=rwse_steps,
            expand_x=False,
        )
        self.edge_encoder = BondEncoder(dim_emb)

    def forward(self, batch):
        batch = self.atom_encoder(batch)
        batch = self.rwse_encoder(batch)
        batch = self.edge_encoder(batch)
        return batch


def make_gps_stack(dim_h, num_heads, dropout, attn_dropout, num_layers):
    layers = []
    for _ in range(num_layers):
        layers.append(GPSLayer(
            dim_h,
            num_heads=num_heads,
            act='relu',
            equivstable_pe=False,
            dropout=dropout,
            attn_dropout=attn_dropout,
            layer_norm=False,
            batch_norm=True,
            log_attn_weights=False,
        ))
    return torch.nn.Sequential(*layers)


class GPSModel(torch.nn.Module):
    """Dual-FluoGPS graph transformer.

    Solute and solvent nodes share the same GPS stack, but global attention is
    role-blocked so their first explicit interaction happens in SANGraphHead.

    https://arxiv.org/abs/2205.12454
    Rampasek, L., Galkin, M., Dwivedi, V. P., Luu, A. T., Wolf, G., & Beaini, D.
    Recipe for a general, powerful, scalable graph transformer. (NeurIPS 2022)
    """

    def __init__(self, dim_h, num_heads=4, dropout=0.05,attn_dropout=0.05, num_layers=10, dim_out=1, rwse_steps=None):
        super().__init__()
        dim_emb = dim_h
        self.encoder = FeatureEncoder(dim_emb, rwse_steps=rwse_steps)
        layers = []
        for _ in range(num_layers):
            layers.append(GPSLayer(
                dim_h,
                num_heads=num_heads,
                act='relu',
                equivstable_pe=False,
                dropout=dropout,
                attn_dropout=attn_dropout,
                layer_norm=False,
                batch_norm=True,
                log_attn_weights=False,
                block_cross_role_attention=True,
            ))
        self.layers = torch.nn.Sequential(*layers)

        self.post_mp = SANGraphHead(dim_in=dim_h, dim_out=dim_out)

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch


class DualGraphGPSModel(torch.nn.Module):
    """Dual-graph FluoGPS model.

    Solute and solvent are encoded as separate graphs with configurable GPS stacks.
    Their first explicit interaction is the pooled embedding concatenation.
    """

    def __init__(
        self,
        dim_h,
        num_heads=4,
        dropout=0.05,
        attn_dropout=0.05,
        num_layers=10,
        dim_out=1,
        rwse_steps=None,
        dual_weight_mode="shared",
    ):
        super().__init__()
        if dual_weight_mode not in {"shared", "separate"}:
            raise ValueError(
                "dual_weight_mode must be 'shared' or 'separate', "
                f"got {dual_weight_mode!r}"
            )
        self.dual_weight_mode = dual_weight_mode
        if dual_weight_mode == "shared":
            self.encoder = BranchFeatureEncoder(dim_h, rwse_steps=rwse_steps)
            self.layers = make_gps_stack(dim_h, num_heads, dropout, attn_dropout, num_layers)
        else:
            self.solute_encoder = BranchFeatureEncoder(dim_h, rwse_steps=rwse_steps)
            self.solute_layers = make_gps_stack(dim_h, num_heads, dropout, attn_dropout, num_layers)
            self.solvent_encoder = BranchFeatureEncoder(dim_h, rwse_steps=rwse_steps)
            self.solvent_layers = make_gps_stack(dim_h, num_heads, dropout, attn_dropout, num_layers)
        self.post_mp = SANDualGraphHead(dim_in=dim_h, dim_out=dim_out)

    def _make_branch_batch(self, batch, prefix):
        return Batch(
            x=getattr(batch, f"{prefix}_x"),
            edge_index=getattr(batch, f"{prefix}_edge_index"),
            edge_attr=getattr(batch, f"{prefix}_edge_attr"),
            batch=getattr(batch, f"{prefix}_x_batch"),
            pestat_RWSE=getattr(batch, f"{prefix}_pestat_RWSE"),
        )

    def _branch_modules(self, prefix):
        if self.dual_weight_mode == "shared":
            return self.encoder, self.layers
        return getattr(self, f"{prefix}_encoder"), getattr(self, f"{prefix}_layers")

    def _encode_branch(self, branch_batch, num_graphs, prefix):
        encoder, layers = self._branch_modules(prefix)
        branch_batch = encoder(branch_batch)
        branch_batch = layers(branch_batch)
        return global_mean_pool(branch_batch.x, branch_batch.batch, size=num_graphs)

    def forward(self, batch):
        solute_batch = self._make_branch_batch(batch, "solute")
        solvent_batch = self._make_branch_batch(batch, "solvent")
        solute_emb = self._encode_branch(solute_batch, batch.num_graphs, "solute")
        solvent_emb = self._encode_branch(solvent_batch, batch.num_graphs, "solvent")
        return self.post_mp(solute_emb, solvent_emb, batch.y)
