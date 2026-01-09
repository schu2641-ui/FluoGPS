# Clean GPS - 荧光分子预测项目

## 项目概述

从 GraphGPS 项目中提取核心代码，构建一个**简洁、易理解**的图神经网络训练框架，专注于荧光分子物理特性预测。

**核心目标**：
- 移除 GraphGym 框架依赖
- 理解深度学习训练的完整流程
- 掌握图神经网络的实现细节

**技术栈**：
- Python 3.10
- PyTorch 1.13 + CUDA 11.7
- PyTorch Geometric 2.2
- RDKit (分子处理)
- OGB (图基准)

---

## 项目架构

```
clean_gps/
├── data/                   # 数据加载模块
│   ├── dataset.py          # Fluorescence数据集类
│   └── loader.py           # DataLoader + 数据集划分
│
├── models/                 # 模型定义
│   ├── gps_model.py        # GPS模型主类
│   ├── encoders/           # 特征编码器
│   │   ├── node_encoder.py # 节点特征编码 (Atom Encoder)
│   │   ├── edge_encoder.py # 边特征编码 (Bond Encoder)
│   │   └── pos_encoder.py  # 位置编码 (RWSE)
│   ├── layers/             # 核心层
│   │   └── gps_layer.py    # GPS Layer (Local MPNN + Global Attention)
│   └── heads/              # 预测头
│       └── graph_head.py   # 图级预测头
│
├── utils/                  # 工具模块
│   ├── trainer.py          # 训练器
│   └── metrics.py          # 评估指标
│
├── configs/                # 配置
│   └── config.py           # 训练超参数
│
└── scripts/                # 脚本
    ├── train.py            # 训练入口
    └── check_assembly.py   # 组装检查脚本
```

---

## 核心组件说明

### 1. 数据模块 (data/)

**fluorescence.py** - 荧光数据集类
- 继承 `InMemoryDataset`
- 读取手动划分的 CSV 文件 (train_solvent.csv, val_solvent.csv, test_solvent.csv)
- 使用 OGB 的 `smiles2graph` 转换分子
- 数据归一化 (z-score normalization)
- 保存为 PyTorch 格式

**关键方法**：
```python
# 加载训练集
train_dataset = FluorescenceDataset(
    csv_file='train_solvent.csv',
    root='/data/young/text2/GraphGPS/datasets/Fluorescence'
)

# 获取单个样本
data = train_dataset[0]
# data.x: 节点特征 [num_nodes, 9]
# data.edge_index: 边索引 [2, num_edges]
# data.edge_attr: 边特征 [num_edges, 3]
# data.y: 归一化的标签 [1]
```

**数据集文件位置**：
```
datasets/Fluorescence/
├── raw/
│   └── FluoDB-Full.csv          # 原始完整数据
├── splits/
│   ├── train_solvent.csv        # 训练集 (手动划分)
│   ├── val_solvent.csv          # 验证集 (手动划分)
│   └── test_solvent.csv         # 测试集 (手动划分)
└── processed/
    ├── fluorescence_train_processed.pt
    ├── fluorescence_val_processed.pt
    └── fluorescence_test_processed.pt
```

**loader.py** - 数据加载器
- 从 splits/ 目录加载手动划分的 CSV 文件
- 为 train/val/test 创建数据集
- 创建 DataLoader
- 支持溶剂划分 (solvent) 和随机划分 (random)

**使用方式**：
```python
from data.loader import get_data_loaders

train_loader, val_loader, test_loader = get_data_loaders(
    split_type='solvent',  # 使用溶剂划分
    batch_size=32,
    num_workers=4
)
```

---

### 2. 模型模块 (models/)

**GPS 模型架构**：
```
Input (分子图)
    ↓
Node Encoder (Atom特征 → 嵌入)
    ↓
Edge Encoder (Bond特征 → 嵌入)
    ↓
GPS Layer × N
    ├── Local MPNN (GINE/GCN)
    └── Global Attention (Transformer)
    ↓
Graph Head (池化 + 预测)
    ↓
Output (预测值)
```

**gps_layer.py** - GPS核心层
- **Local Message Passing**: 局部消息传递 (GINE, GCN, GAT)
- **Global Attention**: 全局注意力机制 (Transformer)
- **Feed Forward**: 前馈网络
- **Residual + Normalization**: 残差连接和归一化

**gps_model.py** - 完整模型
- 特征编码
- GPS 层堆叠
- 预测头

---

### 3. 训练流程 (utils/)

**trainer.py** - 训练器
- 训练循环
- 验证评估
- 模型保存
- 最佳模型选择

**训练循环伪代码**：
```python
for epoch in range(max_epoch):
    # 训练
    model.train()
    for batch in train_loader:
        pred, true = model(batch)
        loss = compute_loss(pred, true)
        loss.backward()
        optimizer.step()

    # 验证
    model.eval()
    for batch in val_loader:
        pred, true = model(batch)
        metrics = evaluate(pred, true)

    # 保存最佳模型
    if metrics['mae'] < best_mae:
        save_model()
```

---

## 🔄 完整训练流程（从Config到训练到预测）

### 调用链路图

```
scripts/train.py (训练入口)
    ↓
    1. 加载配置: configs.config.py
    ↓
    2. 加载数据: data.loader.get_data_loaders()
        ↓ FluorescenceDataset (读取CSV + 归一化)
        ↓ compute_rwse_for_dataset() [计算RWSE]
        ↓ DataLoader (批处理)
    ↓
    3. 创建模型: models.gps_model.GPSModel()
        ↓ node_encoder (AtomEncoder + RWSEEncoder)
        ↓ edge_encoder (BondEncoder)
        ↓ GPSLayer × N (Local MPNN + Global Attention)
        ↓ graph_head (预测)
    ↓
    4. 训练器: utils.trainer.Trainer
        ↓ train_epoch() [训练循环]
        ↓ evaluate() [评估 + 反归一化]
        ↓ save_checkpoint() [保存模型]
    ↓
    5. 最终预测: test_mae = trainer.evaluate(test_loader)
```

### 详细代码流程

#### Step 1: 训练入口 (scripts/train.py)

```python
import sys
sys.path.append('/data/young/text2/clean_gps')

import torch
from configs.config import DATA, MODEL, OPTIM, POS_ENC
from data.loader import get_data_loaders
from models.gps_model import GPSModel
from utils.trainer import Trainer

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 加载数据
    print("Loading data...")
    train_loader, val_loader, test_loader = get_data_loaders(
        batch_size=DATA['batch_size'],
        num_workers=DATA['num_workers'],
        split_type=DATA['split_type'],
        compute_rwse=POS_ENC['RWSE']['enable']
    )
    print(f"Train: {len(train_loader.dataset)} graphs")
    print(f"Val: {len(val_loader.dataset)} graphs")
    print(f"Test: {len(test_loader.dataset)} graphs")

    # 2. 创建模型
    print("Creating model...")
    model = GPSModel(
        dim_in=9,  # 原子特征维度
        dim_out=1,  # 预测维度
        dim_hidden=MODEL['dim_inner'],
        num_layers=MODEL['num_layers'],
        num_heads=MODEL['num_heads'],
        local_gnn_type=MODEL['local_gnn_type'],
        global_attn_type=MODEL['global_attn_type'],
        node_encoder_name='Atom+RWSE',
        edge_encoder_name='Bond'
    ).to(device)

    # 打印模型参数量
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # 3. 优化器
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=OPTIM['lr'],
        weight_decay=OPTIM['weight_decay']
    )

    # 4. 训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=None,
        device=device,
        config={
            'y_mean': train_loader.dataset.y_mean,
            'y_std': train_loader.dataset.y_std,
            'max_epoch': OPTIM['max_epoch']
        }
    )

    # 5. 训练
    print("Starting training...")
    trainer.train()

    # 6. 测试
    print("\nEvaluating on test set...")
    test_mae = trainer.evaluate(test_loader, split='test')
    print(f"Test MAE: {test_mae:.4f} nm")

if __name__ == '__main__':
    main()
```

#### Step 2: 数据加载 (data/loader.py)

```python
def get_data_loaders(batch_size=32, num_workers=4, split_type='solvent',
                     compute_rwse=True):
    """
    加载fluorescence数据集的DataLoader

    Args:
        batch_size: 批大小
        num_workers: 数据加载线程数
        split_type: 'solvent' (溶剂划分) 或 'random' (随机划分)
        compute_rwse: 是否计算RWSE位置编码

    Returns:
        train_loader, val_loader, test_loader
    """
    from torch_geometric.loader import DataLoader
    from data.fluorescence import FluorescenceDataset

    root = '/data/young/text2/GraphGPS/datasets/Fluorescence'

    # 加载数据集
    if split_type == 'solvent':
        train_dataset = FluorescenceDataset(
            root=root, csv_file='train_solvent.csv'
        )
        val_dataset = FluorescenceDataset(
            root=root, csv_file='val_solvent.csv'
        )
        test_dataset = FluorescenceDataset(
            root=root, csv_file='test_solvent.csv'
        )
    else:
        # 随机划分
        full_dataset = FluorescenceDataset(root=root)
        # ... 划分逻辑

    print(f"Dataset normalization: mean={train_dataset.y_mean:.4f}, "
          f"std={train_dataset.y_std:.4f}")

    # 计算RWSE（如果需要）
    if compute_rwse:
        print("Computing RWSE for all graphs...")
        from models.encoders.compute_rwse import compute_rwse_for_dataset
        ksteps = [1, 2, 4, 8, 16]
        compute_rwse_for_dataset(train_dataset, ksteps=ksteps)
        compute_rwse_for_dataset(val_dataset, ksteps=ksteps)
        compute_rwse_for_dataset(test_dataset, ksteps=ksteps)
        print("RWSE computation complete!")

    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader
```

#### Step 3: 模型定义 (models/gps_model.py)

```python
import torch
import torch.nn as nn
from models.encoders.node_encoder import AtomEncoder
from models.encoders.edge_encoder import BondEncoder
from models.encoders.pos_encoder import RWSEEncoder
from models.layers.gps_layer import GPSLayer
from models.heads.graph_head import GraphHead

class GPSModel(nn.Module):
    """General Powerful Scalable Graph Transformer"""

    def __init__(self, dim_in, dim_out, dim_hidden=76, num_layers=4,
                 num_heads=4, local_gnn_type='GINE',
                 global_attn_type='Transformer',
                 node_encoder_name='Atom', edge_encoder_name='Bond'):
        super().__init__()

        # 节点编码器
        if node_encoder_name == 'Atom+RWSE':
            self.node_encoder = AtomRWSEEncoder(
                dim_emb=dim_hidden,
                dim_pe=20,  # RWSE嵌入维度
                num_rw_steps=16
            )
        else:
            self.node_encoder = AtomEncoder(dim_emb=dim_hidden)

        # 边编码器
        if edge_encoder_name == 'Bond':
            self.edge_encoder = BondEncoder(dim_emb=dim_hidden)

        # GPS层堆叠
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                GPSLayer(
                    dim_h=dim_hidden,
                    local_gnn_type=local_gnn_type,
                    global_model_type=global_attn_type,
                    num_heads=num_heads
                )
            )

        # 预测头
        self.head = GraphHead(dim_in=dim_hidden, dim_out=dim_out)

    def forward(self, batch):
        # 编码节点和边特征
        batch = self.node_encoder(batch)  # 添加RWSE并编码
        batch = self.edge_encoder(batch)

        # 通过GPS层
        for layer in self.layers:
            batch = layer(batch)

        # 预测
        pred = self.head(batch)

        return pred
```

#### Step 4: 训练器 (utils/trainer.py)

```python
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error
import numpy as np

class Trainer:
    """训练器类"""

    def __init__(self, model, train_loader, val_loader, test_loader,
                 optimizer, scheduler, device, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config

        # 归一化参数（用于反归一化）
        self.y_mean = config['y_mean']
        self.y_std = config['y_std']

        self.best_mae = float('inf')
        self.best_epoch = 0

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0

        for batch in self.train_loader:
            batch = batch.to(self.device)

            # 前向传播
            pred = self.model(batch)
            loss = F.mse_loss(pred, batch.y)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    @torch.no_grad()
    def evaluate(self, loader, split='val'):
        """评估模型（包含反归一化）"""
        self.model.eval()
        pred_list = []
        true_list = []

        for batch in loader:
            batch = batch.to(self.device)

            # 前向传播（输出归一化的预测值）
            pred = self.model(batch)

            # 反归一化
            pred_real = pred.cpu().numpy() * self.y_std + self.y_mean
            true_real = batch.y.cpu().numpy() * self.y_std + self.y_mean

            pred_list.append(pred_real)
            true_list.append(true_real)

        # 合并所有预测
        pred_all = np.concatenate(pred_list)
        true_all = np.concatenate(true_list)

        # 计算MAE
        mae = mean_absolute_error(true_all, pred_all)

        if split == 'val':
            print(f"Val MAE: {mae:.4f} nm")
        elif split == 'test':
            print(f"Test MAE: {mae:.4f} nm")

        return mae

    def save_checkpoint(self, epoch):
        """保存模型检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_mae': self.best_mae,
        }
        torch.save(checkpoint, 'best_model.pt')
        print(f"Saved best model at epoch {epoch}")

    def train(self):
        """完整训练循环"""
        max_epochs = self.config['max_epoch']

        for epoch in range(max_epochs):
            # 训练
            train_loss = self.train_epoch(epoch)
            print(f"Epoch {epoch+1}/{max_epochs}, Train Loss: {train_loss:.4f}")

            # 验证（每5个epoch验证一次）
            if (epoch + 1) % 5 == 0:
                val_mae = self.evaluate(self.val_loader, split='val')

                # 保存最佳模型
                if val_mae < self.best_mae:
                    self.best_mae = val_mae
                    self.best_epoch = epoch
                    self.save_checkpoint(epoch)

        print(f"\nTraining complete! Best Val MAE: {self.best_mae:.4f} nm at epoch {self.best_epoch}")
```

---

## 🔬 RWSE（随机游走结构编码）完整流程

### RWSE是什么？

RWSE（Random Walk Structural Encoding）通过计算随机游走k步后返回原节点的概率来捕获图的结构信息。

### 完整流程图

```
┌─────────────────────────────────────────────────────────────┐
│ 数据加载阶段 (data/loader.py)                                │
├─────────────────────────────────────────────────────────────┤
│ 1. 检查是否需要RWSE                                          │
│    if config['posenc_RWSE']['enable']:                      │
│                                                               │
│ 2. 调用 compute_rwse_for_dataset(dataset, ksteps=[1,2,4,8,16])│
│                                                               │
│ 3. 对每个图调用 get_rw_landing_probs()                       │
│    ├─ 计算出度: deg = scatter(edge_weight, source)          │
│    ├─ 构建转移矩阵: P = D^-1 * A                            │
│    ├─ 计算 P^k 的对角线: diagonal(P^k)                      │
│    └─ 应用校正: * k^(space_dim/2)                           │
│                                                               │
│ 4. 保存到 data.pestat_RWSE                                   │
│    shape: (num_nodes, num_ksteps)                           │
│    例如: (25, 5) 表示25个节点，5个步数                       │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 模型训练阶段 (models/encoders/rwse_encoder.py)               │
├─────────────────────────────────────────────────────────────┤
│ 1. 从batch读取RWSE统计特征                                   │
│    pos_enc = batch.pestat_RWSE  # (total_nodes, num_ksteps) │
│                                                               │
│ 2. 可选归一化                                                │
│    pos_enc = BatchNorm(pos_enc)                             │
│                                                               │
│ 3. MLP编码: 将ksteps维映射到dim_pe维                         │
│    pos_enc = pe_encoder(pos_enc)                            │
│    (total_nodes, num_ksteps) → (total_nodes, dim_pe)       │
│    例如: (800, 5) → (800, 20)                               │
│                                                               │
│ 4. 与原子特征拼接                                            │
│    atom_feat: (total_nodes, dim_in)  # 如 (800, 9)         │
│    pos_enc: (total_nodes, dim_pe)    # 如 (800, 20)        │
│    batch.x = torch.cat((atom_feat, pos_enc), 1)             │
│    结果: (total_nodes, dim_in + dim_pe) # (800, 29)        │
│                                                               │
│ 5. 输入到GPS层进行消息传递                                    │
└─────────────────────────────────────────────────────────────┘
```

### 核心代码实现

#### RWSE计算 (models/encoders/compute_rwse.py)

```python
import torch
from torch_geometric.utils import scatter, to_dense_adj
from torch_geometric.utils.num_nodes import maybe_num_nodes

def get_rw_landing_probs(ksteps, edge_index, edge_weight=None,
                         num_nodes=None, space_dim=0):
    """
    计算随机游走k步后返回原节点的概率

    Args:
        ksteps: 随机游走步数列表，如 [1, 2, 4, 8, 16]
        edge_index: 边索引 [2, num_edges]
        edge_weight: 边权重
        num_nodes: 节点数
        space_dim: 空间维度（用于校正）

    Returns:
        rw_landing: (num_nodes, len(ksteps)) RWSE统计特征
    """
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)

    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    source, dest = edge_index[0], edge_index[1]

    # 1. 计算节点的出度
    deg = scatter(edge_weight, source, dim=0, dim_size=num_nodes, reduce='sum')
    deg_inv = deg.pow(-1.)
    deg_inv.masked_fill_(deg_inv == float('inf'), 0)

    # 2. 构建转移概率矩阵 P = D^-1 * A
    if edge_index.numel() == 0:
        P = edge_index.new_zeros((1, num_nodes, num_nodes))
    else:
        P = torch.diag(deg_inv) @ to_dense_adj(edge_index, max_num_nodes=num_nodes)

    # 3. 计算不同步数的RWSE
    rws = []
    if ksteps == list(range(min(ksteps), max(ksteps) + 1)):
        # 优化：连续步数时递归计算
        Pk = P.clone().detach().matrix_power(min(ksteps))
        for k in range(min(ksteps), max(ksteps) + 1):
            # 取对角线元素（返回原节点的概率）
            rws.append(torch.diagonal(Pk, dim1=-2, dim2=-1) * (k ** (space_dim / 2)))
            Pk = Pk @ P  # 下一步
    else:
        # 非连续步数，分别计算
        for k in ksteps:
            rws.append(torch.diagonal(P.matrix_power(k), dim1=-2, dim2=-1) * \
                       (k ** (space_dim / 2)))

    # 4. 拼接结果: (num_nodes, num_ksteps)
    rw_landing = torch.cat(rws, dim=0).transpose(0, 1)

    return rw_landing

def compute_rwse_for_dataset(dataset, ksteps=[1, 2, 4, 8, 16]):
    """为数据集中的每个图计算RWSE"""
    for data in dataset:
        rw_landing = get_rw_landing_probs(
            ksteps=ksteps,
            edge_index=data.edge_index,
            num_nodes=data.num_nodes
        )
        data.pestat_RWSE = rw_landing  # 保存到data对象
```

#### RWSE编码器 (models/encoders/rwse_encoder.py)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class RWSENodeEncoder(nn.Module):
    """RWSE位置编码器"""

    def __init__(self, dim_emb, dim_pe=20, num_rw_steps=16,
                 model_type='mlp', raw_norm_type='none'):
        super().__init__()
        self.dim_pe = dim_pe
        self.num_rw_steps = num_rw_steps

        # 原始特征归一化
        if raw_norm_type == 'batchnorm':
            self.raw_norm = nn.BatchNorm1d(num_rw_steps)
        else:
            self.raw_norm = None

        # PE编码器：将num_rw_steps维映射到dim_pe维
        if model_type == 'mlp':
            self.pe_encoder = nn.Sequential(
                nn.Linear(num_rw_steps, 2 * dim_pe),
                nn.ReLU(),
                nn.Linear(2 * dim_pe, dim_pe),
                nn.ReLU()
            )
        elif model_type == 'linear':
            self.pe_encoder = nn.Linear(num_rw_steps, dim_pe)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def forward(self, batch):
        """
        Args:
            batch: PyG batch对象，包含batch.pestat_RWSE

        Returns:
            batch: 添加了编码后的位置编码
        """
        # 1. 读取RWSE统计特征
        pos_enc = batch.pestat_RWSE  # (total_nodes, num_rw_steps)

        # 2. 可选归一化
        if self.raw_norm is not None:
            pos_enc = self.raw_norm(pos_enc)

        # 3. MLP编码
        pos_enc = self.pe_encoder(pos_enc)  # (total_nodes, dim_pe)

        return pos_enc

class AtomRWSEEncoder(nn.Module):
    """组合原子特征编码器和RWSE编码器"""

    def __init__(self, dim_emb, dim_pe=20, num_rw_steps=16):
        super().__init__()

        from torch_geometric.graphgym.models.encoder import AtomEncoder

        # 原子编码器
        self.atom_encoder = AtomEncoder(dim_emb - dim_pe)

        # RWSE编码器
        self.rwse_encoder = RWSENodeEncoder(
            dim_emb=dim_emb,
            dim_pe=dim_pe,
            num_rw_steps=num_rw_steps
        )

    def forward(self, batch):
        # 1. 编码原子特征
        h = self.atom_encoder(batch.x)  # (num_nodes, dim_emb - dim_pe)

        # 2. 编码RWSE
        pos_enc = self.rwse_encoder(batch)  # (num_nodes, dim_pe)

        # 3. 拼接
        batch.x = torch.cat([h, pos_enc], dim=1)  # (num_nodes, dim_emb)

        return batch
```

### RWSE维度变化示例

```
单个图（25个节点）:
    初始: edge_index [2, 60]  (30条边)
    ↓ get_rw_landing_probs(ksteps=[1,2,4,8,16])
    pestat_RWSE: [25, 5]  (5个步数)

Batch中（32个图，共800个节点）:
    batch.pestat_RWSE: [800, 5]
    ↓ RWSENodeEncoder (MLP)
    pos_enc: [800, 20]  (dim_pe=20)
    ↓ 与原子特征拼接
    atom_feat: [800, 9]  (原始原子特征)
    batch.x: [800, 29]  (9 + 20)
```

---

## 📊 数据归一化/反归一化完整流程

### 为什么需要归一化？

荧光分子的吸收/发射波长范围：200-800nm，数值较大且分布不均匀，直接训练会导致：
- 梯度爆炸/消失
- 训练不稳定
- 收敛速度慢

### 归一化公式

```python
# z-score归一化
y_normalized = (y - mean) / std

# 反归一化
y_original = y_normalized * std + mean
```

### 完整代码流程

#### 阶段1: 数据处理 (data/fluorescence.py)

```python
from torch_geometric.data import InMemoryDataset
from ogb.utils import smiles2graph
import pandas as pd
import torch
import numpy as np

class FluorescenceDataset(InMemoryDataset):
    def __init__(self, root, csv_file):
        self.csv_file = csv_file
        super().__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return f'fluorescence_{self.csv_file.replace(".csv", "")}_processed.pt'

    def process(self):
        # 1. 读取CSV文件
        df = pd.read_csv(f'{self.raw_dir}/splits/{self.csv_file}')

        # 2. 提取SMILES和标签
        smiles_list = df['SMILES'].values
        labels = df['absorption/nm'].values  # 原始标签，如 450.5 nm

        # 3. 计算归一化参数（使用所有样本的统计量）
        self.y_mean = labels.mean()
        self.y_std = labels.std()
        print(f"Dataset: mean={self.y_mean:.2f}, std={self.y_std:.2f}")

        # 4. 归一化标签
        labels_normalized = (labels - self.y_mean) / self.y_std

        # 5. 转换SMILES为图
        data_list = []
        for i, smiles in enumerate(smiles_list):
            graph = smiles2graph(smiles)  # 使用OGB转换
            data = torch_geometric.data.Data(
                x=torch.from_numpy(graph['node_feat']).float(),  # [num_nodes, 9]
                edge_index=torch.from_numpy(graph['edge_index']).long(),  # [2, num_edges]
                edge_attr=torch.from_numpy(graph['edge_feat']).float(),  # [num_edges, 3]
                y=torch.tensor([labels_normalized[i]], dtype=torch.float)  # [1] 归一化后的标签
            )
            data_list.append(data)

        # 6. 保存
        self.save(data_list, self.processed_paths[0])
```

#### 阶段2: 训练阶段 (utils/trainer.py)

```python
class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader,
                 optimizer, scheduler, device, config):
        self.model = model
        self.train_loader = train_loader
        self.device = device

        # 从数据集获取归一化参数
        self.y_mean = config['y_mean']  # 例如: 450.23
        self.y_std = config['y_std']    # 例如: 87.45

    def train_epoch(self, epoch):
        """训练：使用归一化的标签"""
        self.model.train()
        for batch in self.train_loader:
            batch = batch.to(self.device)

            # batch.y已经是归一化的标签 (范围约在-3到3之间)
            pred = self.model(batch)  # 模型输出归一化的预测值
            loss = F.mse_loss(pred, batch.y)  # 在归一化空间计算损失

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    @torch.no_grad()
    def evaluate(self, loader, split='val'):
        """评估：需要反归一化才能计算真实的MAE"""
        self.model.eval()
        pred_list = []
        true_list = []

        for batch in loader:
            batch = batch.to(self.device)

            # 1. 模型输出（归一化的预测值）
            pred_norm = self.model(batch)  # 例如: tensor([[0.5234]])

            # 2. 提取归一化的真实标签
            true_norm = batch.y  # 例如: tensor([[0.6123]])

            # 3. 反归一化到原始空间
            pred_real = pred_norm.cpu().numpy() * self.y_std + self.y_mean
            # 例如: 0.5234 * 87.45 + 450.23 = 495.87 nm

            true_real = true_norm.cpu().numpy() * self.y_std + self.y_mean
            # 例如: 0.6123 * 87.45 + 450.23 = 503.65 nm

            pred_list.append(pred_real)
            true_list.append(true_real)

        # 4. 合并所有预测
        pred_all = np.concatenate(pred_list)  # 例如: [495.87, 512.34, ...]
        true_all = np.concatenate(true_list)  # 例如: [503.65, 508.21, ...]

        # 5. 在原始空间计算MAE（单位：nm）
        mae = mean_absolute_error(true_all, pred_all)
        # 例如: |503.65 - 495.87| / N = 7.23 nm

        return mae
```

#### 阶段3: 预测新分子 (scripts/predict.py)

```python
def predict_single_molecule(model, smiles, y_mean, y_std, device):
    """预测单个分子的吸收波长"""
    from ogb.utils import smiles2graph
    import torch

    # 1. 转换SMILES为图
    graph = smiles2graph(smiles)
    data = torch_geometric.data.Data(
        x=torch.from_numpy(graph['node_feat']).float().unsqueeze(0),
        edge_index=torch.from_numpy(graph['edge_index']).long().unsqueeze(0),
        edge_attr=torch.from_numpy(graph['edge_feat']).float().unsqueeze(0),
    ).to(device)

    # 2. 模型预测（归一化空间）
    model.eval()
    with torch.no_grad():
        pred_norm = model(data)  # 例如: tensor([[0.4523]])

    # 3. 反归一化到原始空间
    pred_real = pred_norm.cpu().item() * y_std + y_mean
    # 例如: 0.4523 * 87.45 + 450.23 = 489.67 nm

    return pred_real

# 使用示例
model = GPSModel(...)
checkpoint = torch.load('best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)

smiles = "CCO"  # 乙醇
pred_wavelength = predict_single_molecule(
    model, smiles,
    y_mean=450.23,
    y_std=87.45,
    device=device
)
print(f"Predicted absorption: {pred_wavelength:.2f} nm")
```

### 归一化效果示例

```
未归一化（直接训练）:
    标签范围: [200, 800] nm
    损失值: 500-10000
    梯度: 不稳定
    训练: 难以收敛

归一化后:
    标签范围: [-3, +3] (归一化空间)
    损失值: 0.1-2.0
    梯度: 稳定
    训练: 快速收敛
    评估时反归一化得到真实MAE
```

---

## 数据归一化

### 为什么需要归一化？

原始标签范围：200-800nm (吸收波长)

归一化公式：
```python
y_normalized = (y - mean) / std
```

反归一化（评估时）：
```python
y_original = y_normalized * std + mean
```

**好处**：
- 梯度更稳定
- 训练更快收敛
- 不同量纲的特征可比

---

## GraphGym 依赖移除

### 原始代码 (GraphGym)

```python
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_network

@register_network('GPSModel')
class GPSModel(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        dim_h = cfg.gt.dim_hidden      # 从配置读取
        num_layers = cfg.gt.layers
        local_type = cfg.gt.layer_type  # 如 "GINE+Transformer"
```

### 简化代码 (无依赖)

```python
class GPSModel(nn.Module):
    def __init__(self, dim_in, dim_out,
                 dim_hidden=76,
                 num_layers=4,
                 local_gnn_type='GINE',
                 global_attn_type='Transformer'):
        super().__init__()
        dim_h = dim_hidden           # 参数传入
        # ... 直接使用参数
```

**需要移除的依赖**：
- `cfg` 配置对象
- `@register_network` 装饰器
- `@register_train` 装饰器
- GraphGym 的日志系统

---

## 关键文件映射

从 GraphGPS 复制文件时的映射关系：

| GraphGPS 原文件 | clean_gps 目标文件 | 修改点 |
|---|---|---|
| `graphgps/loader/dataset/fluorescence.py` | `data/dataset.py` | 基本不变 |
| `graphgps/layer/gps_layer.py` | `models/layers/gps_layer.py` | 移除 cfg 依赖 |
| `graphgps/network/gps_model.py` | `models/gps_model.py` | 移除 cfg 和 register |
| `graphgps/encoder/atom_encoder.py` | `models/encoders/node_encoder.py` | 简化配置 |
| `graphgps/encoder/bond_encoder.py` | `models/encoders/edge_encoder.py` | 简化配置 |
| `graphgps/encoder/rwse_encoder.py` | `models/encoders/pos_encoder.py` | 简化配置 |
| `graphgps/head/graph_head.py` | `models/heads/graph_head.py` | 简化配置 |
| `graphgps/train/custom_train.py` | `utils/trainer.py` | 提取核心逻辑 |

---

## 训练配置

### 超参数说明

```python
MODEL = {
    'dim_inner': 76,        # 隐藏层维度 (原子嵌入维度)
    'num_layers': 4,        # GPS 层数
    'num_heads': 4,         # 注意力头数
    'local_gnn_type': 'GINE',    # 局部消息传递类型
    'global_attn_type': 'Transformer',  # 全局注意力类型
    'dropout': 0.0,
    'attn_dropout': 0.0,
}

OPTIM = {
    'optimizer': 'Adam',
    'lr': 0.001,
    'weight_decay': 1e-5,
    'max_epoch': 100,
}
```

### 模型变体

**小模型** (快速实验):
```python
dim_inner=32, num_layers=2, num_heads=2
```

**标准模型** (推荐):
```python
dim_inner=76, num_layers=4, num_heads=4
```

**大模型** (最佳性能):
```python
dim_inner=256, num_layers=8, num_heads=8
```

---

## 常见问题

### 1. 特征维度不匹配

**错误**: `RuntimeError: mat1 and mat2 shapes cannot be multiplied`

**原因**: 原始节点特征维度 (9) 与模型期望维度不符

**解决**: 检查 `dim_in` 参数是否正确

```python
model = GPSModel(
    dim_in=9,   # 原始节点特征维度
    dim_out=1,  # 输出维度
    ...
)
```

### 2. 数据加载失败

**错误**: `FileNotFoundError: ... FluoDB-Full.csv`

**解决**: 检查 `config.py` 中的 `DATA['root']` 路径

### 3. CUDA 内存不足

**解决**:
- 减小 `batch_size`
- 减小模型尺寸 (`dim_inner`, `num_layers`)
- 使用 CPU 训练

---

## 开发工作流

### 1. 修改数据集

```python
# data/dataset.py
# 修改 target_col 即可预测不同属性
target_col = 'emission/nm'  # 改为预测发射波长
```

### 2. 修改模型架构

```python
# models/gps_model.py
# 修改 GPS 层的类型
local_gnn_type='GCN'  # 改用 GCN
global_attn_type='Performer'  # 改用 Performer
```

### 3. 调整训练参数

```python
# configs/config.py
# 修改学习率、batch size 等
OPTIM['lr'] = 0.0005
DATA['batch_size'] = 64
```

---

## 基准测试

在荧光数据集上的预期性能：

| 模型 | MAE (nm) | 参数量 | 训练时间 |
|---|---|---|---|
| GPS-Small | ~30 | 50K | ~1h |
| GPS-Base | ~25 | 200K | ~3h |
| GPS-Large | ~20 | 1M | ~10h |

---

## 学习路径

### 初学者
1. ✅ 理解数据加载流程
2. ✅ 理解前向传播
3. ✅ 理解训练循环
4. ✅ 尝试修改超参数

### 进阶
1. ✅ 理解 GPS Layer 细节
2. ✅ 尝试不同的编码器
3. ✅ 修改模型架构
4. ✅ 实现新的位置编码

### 高级
1. ✅ 理解注意力机制
2. ✅ 实现自定义消息传递
3. ✅ 多任务学习
4. ✅ 模型解释性分析

---

## 参考文献

**GPS 论文**:
- Rampášek et al., "Recipe for a General, Powerful, Scalable Graph Transformer", NeurIPS 2022
- https://arxiv.org/abs/2205.12454

**相关库**:
- PyTorch Geometric: https://pyg.org/
- OGB: https://ogb.stanford.edu/
- RDKit: https://www.rdkit.org/

---

## 项目状态

- [x] 项目框架搭建
- [ ] 数据加载模块
- [ ] 模型定义
- [ ] 训练器实现
- [ ] 训练脚本
- [ ] 模型评估

**当前进度**: 框架搭建完成，开始组装核心组件

---

## 联系方式

遇到问题时：
1. 查看 `ASSEMBLY_GUIDE.md`
2. 运行 `python scripts/check_assembly.py` 检查
3. 查看 `README.md` 的快速开始指南
4. 询问 Claude 获取帮助
