"""
支持消融实验的GPS模型
"""
import torch
from torch_geometric.graphgym.models.encoder import BondEncoder, AtomEncoder
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool

from models.layers.GPS_layer import GPSLayer
from models.encoders.rwse_encoder import RWSEEncoder
from models.heads.san_graph import SANGraphHead
from ablation_config import AblationConfig


class FeatureEncoder(torch.nn.Module):
    """特征编码器，支持消融"""
    
    def __init__(self, dim_emb, config: AblationConfig):
        super(FeatureEncoder, self).__init__()
        self.config = config
        
        # 计算各部分维度
        if config.use_rwse:
            dim_pe = config.rwse_dim_pe
            atom_dim_out = dim_emb - dim_pe
        else:
            atom_dim_out = dim_emb
            
        # 原子编码器
        if config.use_atom_encoder:
            self.atom_encoder = AtomEncoder(atom_dim_out)
        else:
            self.atom_encoder = None
            
        # RWSE位置编码
        if config.use_rwse:
            self.rwse_encoder = RWSEEncoder(
                dim_in=atom_dim_out,
                dim_emb=dim_emb,
                rwse_steps=config.rwse_steps,
                expand_x=config.use_atom_encoder,
                dim_pe=config.rwse_dim_pe
            )
        else:
            self.rwse_encoder = None
            
        # 边编码器
        if config.use_edge_encoder:
            self.edge_encoder = BondEncoder(dim_emb)
        else:
            self.edge_encoder = None
            
    def forward(self, batch):
        # 原子特征编码
        if self.atom_encoder is not None:
            batch = self.atom_encoder(batch)
            
        # RWSE位置编码
        if self.rwse_encoder is not None:
            batch = self.rwse_encoder(batch)
            
        # 边特征编码
        if self.edge_encoder is not None:
            batch = self.edge_encoder(batch)
            
        return batch


class GPSLayerAblation(torch.nn.Module):
    """支持消融的GPS层"""
    
    def __init__(self, dim_h, config: AblationConfig):
        super().__init__()
        self.config = config
        
        # Local MPNN
        if config.use_local_gnn:
            from models.layers.gatedgcn_layer import GatedGCNLayer
            self.local_model = GatedGCNLayer(
                dim_h, dim_h,
                dropout=config.dropout,
                residual=True,
                act='relu'
            )
        else:
            self.local_model = None
            
        # Global Attention
        if config.use_global_attn:
            self.self_attn = torch.nn.MultiheadAttention(
                dim_h, config.num_heads,
                dropout=config.attn_dropout,
                batch_first=True
            )
        else:
            self.self_attn = None
            
        # Normalization
        self.norm1_local = torch.nn.BatchNorm1d(dim_h)
        self.norm1_attn = torch.nn.BatchNorm1d(dim_h)
        self.dropout_local = torch.nn.Dropout(config.dropout)
        self.dropout_attn = torch.nn.Dropout(config.dropout)
        
        # Feed Forward
        self.ff_linear1 = torch.nn.Linear(dim_h, dim_h * 2)
        self.ff_linear2 = torch.nn.Linear(dim_h * 2, dim_h)
        self.act_fn_ff = torch.nn.ReLU()
        self.norm2 = torch.nn.BatchNorm1d(dim_h)
        self.ff_dropout1 = torch.nn.Dropout(config.dropout)
        self.ff_dropout2 = torch.nn.Dropout(config.dropout)
        
    def forward(self, batch):
        from torch_geometric.utils import to_dense_batch
        
        h = batch.x
        h_in1 = h
        
        h_out_list = []
        
        # Local MPNN
        if self.local_model is not None:
            from torch_geometric.data import Batch
            local_out = self.local_model(Batch(
                batch=batch,
                x=h,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr if hasattr(batch, 'edge_attr') else None
            ))
            h_local = local_out.x
            h_local = self.norm1_local(h_local)
            h_out_list.append(h_local)
            
        # Global Attention
        if self.self_attn is not None:
            h_dense, mask = to_dense_batch(h, batch.batch)
            h_attn = self.self_attn(h_dense, h_dense, h_dense,
                                    key_padding_mask=~mask)[0][mask]
            h_attn = self.dropout_attn(h_attn)
            h_attn = h_in1 + h_attn
            h_attn = self.norm1_attn(h_attn)
            h_out_list.append(h_attn)
            
        # Combine
        if len(h_out_list) > 0:
            if self.config.local_global_combine == "sum":
                h = sum(h_out_list)
            elif self.config.local_global_combine == "weighted":
                # 分别加权
                weighted_sum = 0
                if self.local_model is not None:
                    weighted_sum += h_out_list[0] * self.config.local_weight
                if self.self_attn is not None:
                    idx = 1 if self.local_model is not None else 0
                    weighted_sum += h_out_list[idx] * self.config.global_weight
                h = weighted_sum
        else:
            h = h_in1
            
        # Feed Forward
        h = h + self._ff_block(h)
        h = self.norm2(h)
        
        batch.x = h
        return batch
        
    def _ff_block(self, x):
        x = self.ff_dropout1(self.act_fn_ff(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x))


class GPSModelAblation(torch.nn.Module):
    """支持消融实验的GPS模型"""
    
    def __init__(self, config: AblationConfig):
        super().__init__()
        self.config = config
        dim_h = config.dim_hidden
        
        # 特征编码器
        self.encoder = FeatureEncoder(dim_h, config)
        
        # GPS层
        layers = []
        for _ in range(config.num_layers):
            layers.append(GPSLayerAblation(dim_h, config))
        self.layers = torch.nn.Sequential(*layers)
        
        # 预测头
        self.pooling = self._get_pooling(config.pooling_type)
        self.post_mp = SANGraphHead(
            dim_in=dim_h,
            dim_out=1,
            L=config.head_layers
        )
        
    def _get_pooling(self, pooling_type):
        pooling_dict = {
            "mean": global_mean_pool,
            "max": global_max_pool,
            "add": global_add_pool
        }
        return pooling_dict.get(pooling_type, global_mean_pool)
        
    def forward(self, batch):
        # 编码
        batch = self.encoder(batch)
        
        # GPS层
        for layer in self.layers:
            batch = layer(batch)
            
        # 池化
        graph_emb = self.pooling(batch.x, batch.batch)
        batch.graph_feature = graph_emb
        
        # 预测
        pred, label = self.post_mp(batch)
        return pred, label
