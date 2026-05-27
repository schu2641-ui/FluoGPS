import torch
from torch_geometric.graphgym.models.encoder import BondEncoder, AtomEncoder

from models.layers.GPS_layer import GPSLayer
from models.encoders.rwse_encoder import RWSEEncoder
from models.heads.san_graph import SANGraphHead


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

class GPSModel(torch.nn.Module):
    """General-Powerful-Scalable graph transformer.
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
                log_attn_weights=False
            ))
        self.layers = torch.nn.Sequential(*layers)

        self.post_mp = SANGraphHead(dim_in=dim_h, dim_out=dim_out)

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch
