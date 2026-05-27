"""
GPSModelAblation: 用于消融实验的 GPS 模型变体。

通过传入 AblationConfig 字典动态控制各组件的开启/关闭：
    - use_rwse   : 是否使用 RWSE 位置编码
    - rwse_steps : RWSE 随机游走步数列表
    - use_local  : GPS 层是否使用局部 MPNN (GatedGCN)
    - use_global : GPS 层是否使用全局 Attention (Transformer)

示例：
    cfg = AblationConfig(use_rwse=False, rwse_steps=list(range(1,17)),
                         use_local=True, use_global=True)
    model = GPSModelAblation(dim_h=128, ablation_cfg=cfg)
"""

import torch
import torch.nn as nn
from typing import TypedDict, List

from torch_geometric.graphgym.models.encoder import BondEncoder, AtomEncoder

from models.layers.GPS_layer import GPSLayer
from models.encoders.rwse_encoder import RWSEEncoder
from models.heads.san_graph import SANGraphHead


class AblationConfig(TypedDict):
    use_rwse: bool       # 是否启用 RWSE 位置编码
    rwse_steps: List     # 随机游走步数列表，用于 create_loader 和 RWSEEncoder
    use_local: bool      # GPS 层是否启用局部 MPNN
    use_global: bool     # GPS 层是否启用全局 Attention


class FeatureEncoderAblation(nn.Module):
    """
    消融版 FeatureEncoder。

    use_rwse=True  → AtomEncoder(dim_h - 20 - role_dim) + RoleEmbedding(role_dim) + RWSEEncoder(20) → 输出 dim_h
    use_rwse=False → AtomEncoder(dim_h - role_dim) + RoleEmbedding(role_dim)                         → 输出 dim_h
    """

    def __init__(self, dim_emb: int, use_rwse: bool = True, rwse_steps=None):
        super().__init__()
        self.use_rwse = use_rwse
        role_dim_out = 8
        self.role_encoder = torch.nn.Embedding(2, role_dim_out)

        if use_rwse:
            dim_pe = 20
            atom_dim_out = dim_emb - dim_pe - role_dim_out
            if atom_dim_out <= 0:
                raise ValueError(
                    f"dim_emb={dim_emb} is too small for dim_pe={dim_pe} "
                    f"and role_dim={role_dim_out}."
                )
            self.atom_encoder = AtomEncoder(atom_dim_out)
            self.rwse_encoder = RWSEEncoder(
                dim_in=atom_dim_out + role_dim_out,
                dim_emb=dim_emb,
                rwse_steps=rwse_steps,
                expand_x=False,
            )
        else:
            # 不使用 RWSE，AtomEncoder + role embedding 共同输出 dim_emb 维度
            atom_dim_out = dim_emb - role_dim_out
            if atom_dim_out <= 0:
                raise ValueError(
                    f"dim_emb={dim_emb} is too small for role_dim={role_dim_out}."
                )
            self.atom_encoder = AtomEncoder(atom_dim_out)

        self.edge_encoder = BondEncoder(dim_emb)

    def forward(self, batch):
        batch = self.atom_encoder(batch)
        role_emb = self.role_encoder(batch.role)
        batch.x = torch.cat([batch.x, role_emb], dim=-1)
        if self.use_rwse:
            batch = self.rwse_encoder(batch)
        batch = self.edge_encoder(batch)
        return batch


class GPSModelAblation(torch.nn.Module):
    """
    消融实验版 GPS 模型。

    相比 GPSModel，额外接受一个 ablation_cfg 参数来控制各组件开关。
    其余超参数（dim_h, num_heads, dropout 等）与 GPSModel 保持一致。
    """

    def __init__(
        self,
        dim_h: int,
        num_heads: int = 4,
        dropout: float = 0.05,
        attn_dropout: float = 0.05,
        num_layers: int = 10,
        dim_out: int = 1,
        ablation_cfg: AblationConfig = None,
    ):
        super().__init__()

        # 默认配置等价于完整模型（baseline）
        if ablation_cfg is None:
            ablation_cfg = AblationConfig(
                use_rwse=True,
                rwse_steps=list(range(1, 17)),
                use_local=True,
                use_global=True,
            )

        self.ablation_cfg = ablation_cfg

        # ---- 编码器 ----
        self.encoder = FeatureEncoderAblation(
            dim_emb=dim_h,
            use_rwse=ablation_cfg["use_rwse"],
            rwse_steps=ablation_cfg["rwse_steps"],
        )

        # ---- GPS 层栈 ----
        layers = []
        for _ in range(num_layers):
            layers.append(
                GPSLayer(
                    dim_h,
                    num_heads=num_heads,
                    act="relu",
                    equivstable_pe=False,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    layer_norm=False,
                    batch_norm=True,
                    log_attn_weights=False,
                    use_local=ablation_cfg["use_local"],
                    use_global=ablation_cfg["use_global"],
                )
            )
        self.layers = torch.nn.Sequential(*layers)

        # ---- 预测头 ----
        self.post_mp = SANGraphHead(dim_in=dim_h, dim_out=dim_out)

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch
