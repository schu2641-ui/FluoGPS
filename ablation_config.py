"""
消融实验配置文件
"""
from dataclasses import dataclass
from typing import List, Optional
import itertools

@dataclass
class AblationConfig:
    """消融实验配置"""
    exp_name: str = "baseline"
    
    # 位置编码消融
    use_rwse: bool = True
    rwse_dim_pe: int = 20
    rwse_steps: List[int] = None  # None表示使用默认[1,17]
    
    # GPS层结构消融
    use_local_gnn: bool = True
    use_global_attn: bool = True
    local_global_combine: str = "sum"  # "sum" or "weighted"
    local_weight: float = 1.0
    global_weight: float = 1.0
    
    # 模型结构消融
    num_layers: int = 10
    dim_hidden: int = 128
    num_heads: int = 4
    dropout: float = 0.0
    attn_dropout: float = 0.1
    
    # 特征编码消融
    use_atom_encoder: bool = True
    use_edge_encoder: bool = True
    
    # 预测头消融
    pooling_type: str = "mean"  # "mean", "max", "add"
    head_layers: int = 2
    
    # 训练消融
    use_normalization: bool = True
    use_warmup: bool = True
    loss_type: str = "mae"  # "mae", "mse", "huber"
    
    def __post_init__(self):
        if self.rwse_steps is None:
            self.rwse_steps = list(range(1, 17))
    
    def to_dict(self):
        return self.__dict__


# 预定义的消融实验配置
ABLATION_EXPERIMENTS = {
    # 1. 位置编码消融
    "rwse_off": AblationConfig(exp_name="rwse_off", use_rwse=False),
    "rwse_dim_10": AblationConfig(exp_name="rwse_dim_10", rwse_dim_pe=10),
    "rwse_dim_40": AblationConfig(exp_name="rwse_dim_40", rwse_dim_pe=40),
    "rwse_steps_8": AblationConfig(exp_name="rwse_steps_8", rwse_steps=list(range(1, 9))),
    "rwse_steps_32": AblationConfig(exp_name="rwse_steps_32", rwse_steps=list(range(1, 33))),
    
    # 2. GPS层结构消融
    "local_only": AblationConfig(exp_name="local_only", use_global_attn=False),
    "global_only": AblationConfig(exp_name="global_only", use_local_gnn=False),
    
    # 3. 模型深度消融
    "layers_2": AblationConfig(exp_name="layers_2", num_layers=2),
    "layers_4": AblationConfig(exp_name="layers_4", num_layers=4),
    "layers_6": AblationConfig(exp_name="layers_6", num_layers=6),
    "layers_8": AblationConfig(exp_name="layers_8", num_layers=8),
    
    # 4. 注意力机制消融
    "heads_1": AblationConfig(exp_name="heads_1", num_heads=1),
    "heads_2": AblationConfig(exp_name="heads_2", num_heads=2),
    "heads_8": AblationConfig(exp_name="heads_8", num_heads=8),
    "attn_dropout_0": AblationConfig(exp_name="attn_dropout_0", attn_dropout=0.0),
    "attn_dropout_2": AblationConfig(exp_name="attn_dropout_2", attn_dropout=0.2),
    
    # 5. 特征编码消融
    "no_edge_encoder": AblationConfig(exp_name="no_edge_encoder", use_edge_encoder=False),
    
    # 6. 预测头消融
    "pooling_max": AblationConfig(exp_name="pooling_max", pooling_type="max"),
    "pooling_add": AblationConfig(exp_name="pooling_add", pooling_type="add"),
    "head_layers_1": AblationConfig(exp_name="head_layers_1", head_layers=1),
    "head_layers_3": AblationConfig(exp_name="head_layers_3", head_layers=3),
    
    # 7. Dropout消融
    "dropout_01": AblationConfig(exp_name="dropout_01", dropout=0.1),
    "dropout_02": AblationConfig(exp_name="dropout_02", dropout=0.2),
    "dropout_05": AblationConfig(exp_name="dropout_05", dropout=0.5),
    
    # 8. 训练策略消融
    "no_normalization": AblationConfig(exp_name="no_normalization", use_normalization=False),
    "no_warmup": AblationConfig(exp_name="no_warmup", use_warmup=False),
    "loss_mse": AblationConfig(exp_name="loss_mse", loss_type="mse"),
}


def get_ablation_configs(ablation_types: List[str] = None):
    """获取消融实验配置列表"""
    if ablation_types is None:
        return [("baseline", AblationConfig())]
    
    configs = [("baseline", AblationConfig())]
    for exp_name in ablation_types:
        if exp_name in ABLATION_EXPERIMENTS:
            configs.append((exp_name, ABLATION_EXPERIMENTS[exp_name]))
        else:
            print(f"Warning: Unknown experiment '{exp_name}'")
    
    return configs


def generate_grid_search_configs(param_grid: dict):
    """生成网格搜索配置
    示例:
    param_grid = {
        'num_layers': [2, 4, 6],
        'dropout': [0.0, 0.1, 0.2],
        'num_heads': [2, 4]
    }
    """
    keys = param_grid.keys()
    values = param_grid.values()
    
    configs = []
    for combination in itertools.product(*values):
        params = dict(zip(keys, combination))
        exp_name = "_".join([f"{k}_{v}" for k, v in params.items()])
        config = AblationConfig(exp_name=exp_name, **params)
        configs.append((exp_name, config))
    
    return configs


if __name__ == "__main__":
    # 示例：获取特定消融实验
    configs = get_ablation_configs(["rwse_off", "local_only", "layers_4"])
    for name, config in configs:
        print(f"\n{name}:")
        print(f"  use_rwse: {config.use_rwse}")
        print(f"  use_local_gnn: {config.use_local_gnn}")
        print(f"  use_global_attn: {config.use_global_attn}")
        print(f"  num_layers: {config.num_layers}")
