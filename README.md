# Clean GPS - 荧光分子预测项目 (精简版)

## 📖 项目简介

从 GraphGPS 项目中提取核心代码，构建一个简洁的图神经网络训练框架，专注于荧光分子物理特性预测。

**目标**: 移除 GraphGym 依赖，理解深度学习训练的核心流程。

---

## 🗂️ 项目结构

```
clean_gps/
├── ASSEMBLY_GUIDE.md      # 📚 组装指南 - 从这里开始！
├── README.md              # 本文件
├── configs/
│   └── config.py          # 训练配置参数
├── data/                  # 数据加载模块
├── models/                # 模型定义
│   ├── encoders/          # 特征编码器
│   ├── layers/            # 核心层 (GPS Layer)
│   └── heads/             # 预测头
├── utils/                 # 工具函数
└── scripts/               # 训练脚本
    ├── check_assembly.py  # 🔍 组装检查脚本
    └── train.py           # 训练入口 (待创建)
```

---

## 🚀 快速开始

### 第一步：阅读组装指南

打开 `ASSEMBLY_GUIDE.md`，了解需要从 GraphGPS 复制哪些文件。

### 第二步：开始组装

按以下顺序完成：

1. **数据模块** (`data/`)
   - 复制 `fluorescence.py` → `data/dataset.py`
   - 创建 `data/loader.py`

   **检查**: `python scripts/check_assembly.py` (检查1-2)

2. **编码器** (`models/encoders/`)
   - 复制 `atom_encoder.py` → `node_encoder.py`
   - 复制 `bond_encoder.py` → `edge_encoder.py`
   - 复制 `rwse_encoder.py` → `pos_encoder.py`

   **检查**: `python scripts/check_assembly.py` (检查3)

3. **GPS 层** (`models/layers/`)
   - 复制 `gps_layer.py` → `models/layers/gps_layer.py`
   - **重要**: 移除 `cfg` 依赖

   **检查**: `python scripts/check_assembly.py` (检查4)

4. **完整模型** (`models/`)
   - 复制 `gps_model.py` → `models/gps_model.py`
   - **重要**: 移除 `register_network` 和 `cfg` 依赖

   **检查**: `python scripts/check_assembly.py` (检查5)

5. **训练器** (`utils/`)
   - 参考 `custom_train.py` 创建 `utils/trainer.py`
   - 创建 `scripts/train.py`

   **检查**: `python scripts/check_assembly.py` (检查6)

---

## 🔧 关键修改点

### ❌ 原始代码 (GraphGym 依赖)

```python
from torch_geometric.graphgym.config import cfg

class GPSModel(nn.Module):
    def __init__(self, dim_in, dim_out):
        dim_h = cfg.gt.dim_hidden
        # ...
```

### ✅ 修改后 (无依赖)

```python
class GPSModel(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden=76, num_layers=4):
        dim_h = dim_hidden
        # ...
```

---

## 📝 使用方法

### 训练模型

```bash
cd /data/young/text2/clean_gps
python scripts/train.py
```

### 检查组装进度

```bash
python scripts/check_assembly.py
```

---

## 💡 学习要点

通过这个项目，你将学会：

1. ✅ 如何设计图数据集类
2. ✅ 如何构建图神经网络模型
3. ✅ 如何实现训练循环
4. ✅ 如何处理分子数据 (SMILES → 图)
5. ✅ 如何进行数据归一化和反归一化

---

## 🆘 遇到问题？

1. 查看 `ASSEMBLY_GUIDE.md` 中的详细说明
2. 运行 `python scripts/check_assembly.py` 定位问题
3. 查看错误信息的 stack trace
4. 随时问我！

---

## 📊 项目进度

- [ ] 1. 数据加载模块
- [ ] 2. 编码器模块
- [ ] 3. GPS 层
- [ ] 4. 完整模型
- [ ] 5. 训练器
- [ ] 6. 训练脚本
