# Clean GPS 项目组装指南

## 📁 项目结构说明

```
clean_gps/
├── data/                   # 数据加载模块
│   ├── __init__.py
│   ├── dataset.py          # [需创建] Fluorescence数据集
│   └── loader.py           # [需创建] DataLoader封装
│
├── models/                 # 模型定义模块
│   ├── __init__.py
│   ├── gps_model.py        # [需搬运] GPS模型主文件
│   ├── encoders/           # 特征编码器
│   │   ├── __init__.py
│   │   ├── node_encoder.py # [需搬运] 节点特征编码
│   │   ├── edge_encoder.py # [需搬运] 边特征编码
│   │   └── rwse_encoder.py # [需搬运] RWSE位置编码
│   ├── layers/             # 核心层
│   │   ├── __init__.py
│   │   └── gps_layer.py    # [需搬运] GPS核心层
│   └── heads/              # 预测头
│       ├── __init__.py
│       └── graph_head.py   # [需搬运] 预测头
│
├── utils/                  # 工具函数
│   ├── __init__.py
│   ├── trainer.py          # [需创建] 训练器
│   └── metrics.py          # [需搬运/创建] 评估指标
│
├── configs/                # 配置文件
│   └── config.yaml         # [需创建] 训练配置
│
└── scripts/                # 训练脚本
    ├── __init__.py
    └── train.py            # [需创建] 训练入口
```

---

## 🔧 组装步骤

### 第一步：数据加载模块 (data/)

**需要从 GraphGPS 复制的文件：**

1. `graphgps/loader/dataset/fluorescence.py`
   - 复制到: `data/dataset.py`
   - **注意**: 移除 GraphGym 依赖，简化配置

**需要自己创建的文件：**

2. `data/loader.py`
   - 功能: 封装 DataLoader，处理数据集划分
   - 提示: 使用 `torch_geometric.loader.DataLoader`

---

### 第二步：模型模块 (models/)

**需要从 GraphGPS 复制的文件：**

1. `graphgps/layer/gps_layer.py` → `models/layers/gps_layer.py`
   - **注意**: 移除 `cfg` 依赖，改用 `__init__` 参数

2. `graphgps/network/gps_model.py` → `models/gps_model.py`
   - **注意**:
     - 移除 `register_network` 装饰器
     - 移除 `cfg` 依赖
     - 将配置改为 `__init__` 参数

3. 编码器 (`graphgps/encoder/`):
   - `atom_encoder.py` → `models/encoders/node_encoder.py`
   - `bond_encoder.py` → `models/encoders/edge_encoder.py`
   - `rwse_encoder.py` → `models/encoders/pos_encoder.py`

4. 预测头 (`graphgps/head/`):
   - `graph_head.py` → `models/heads/graph_head.py`

---

### 第三步：工具模块 (utils/)

**需要从 GraphGPS 复制的文件：**

1. `graphgps/train/custom_train.py` → 参考创建 `utils/trainer.py`
   - 提取核心训练循环
   - 移除 GraphGym 日志系统

2. 评估指标:
   - 可以从 `graphgps/metric_wrapper.py` 参考创建 `utils/metrics.py`
   - 或者自己实现简单的 MAE 计算

---

### 第四步：配置文件 (configs/)

**需要创建的文件：**

1. `configs/config.yaml`
   - 定义超参数
   - 使用 `yaml` 或简单的 Python 字典

---

### 第五步：训练脚本 (scripts/)

**需要创建的文件：**

1. `scripts/train.py`
   - 训练入口
   - 加载配置、数据、模型
   - 调用训练器

---

## ⚠️ 重要提示：移除 GraphGym 依赖

GraphGPS 大量使用了 GraphGym 的配置系统：

```python
# ❌ 原始代码 (依赖 GraphGym)
from torch_geometric.graphgym.config import cfg

class GPSModel(nn.Module):
    def __init__(self, dim_in, dim_out):
        dim_h = cfg.gt.dim_hidden  # 从配置读取
        layers = cfg.gt.layers
```

**需要改成：**

```python
# ✅ 简化版本 (不依赖 GraphGym)
class GPSModel(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden=76, num_layers=4):
        dim_h = dim_hidden  # 通过参数传入
        layers = num_layers
```

---

## 🎯 建议的组装顺序

1. ✅ 先完成数据加载 (`data/`)
2. ✅ 再完成核心层 (`models/layers/gps_layer.py`)
3. ✅ 然后组装模型 (`models/gps_model.py`)
4. ✅ 最后完成训练循环 (`utils/trainer.py`, `scripts/train.py`)

---

## 📝 每个步骤后的检查清单

- [ ] 数据集能正常加载
- [ ] 模型能正常初始化
- [ ] 前向传播能运行
- [ ] 训练循环能运行
- [ ] 能保存和加载模型

---

## 🆘 遇到问题？

常见问题：
1. **导入错误**: 检查是否正确移除了 `cfg` 依赖
2. **维度不匹配**: 检查特征维度配置
3. **注册器错误**: 移除 `@register_network` 等装饰器

随时提问，我会帮助你！
