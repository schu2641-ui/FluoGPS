# FluoGPS

FluoGPS 是一个面向荧光分子性质预测的研究仓库，核心模型基于 GraphGPS 思路，当前整理为“源码层 + 脚本入口 + 实验材料 + 本地产物”四层结构。

## 目录

```text
FluoGPS/
├── data/              # 数据集定义与图构建
├── models/            # 模型主体
├── train/             # 训练循环
├── utils/             # 调度、日志、推理公共函数
├── scripts/           # CLI 入口
├── experiments/       # 按论文图序组织的实验、绘图、渲染脚本
├── notebooks/         # 探索性 notebook
├── docs/              # 说明文档与归档
├── examples/inputs/   # 可跟踪的样例输入
├── datasets/          # 原始数据与 PyG 缓存根目录
├── checkpoints/       # 本地模型权重，默认不进 Git
└── outputs/           # 训练结果、预测结果、图像产物，默认不进 Git
```

## CLI 入口

统一通过 `python -m ...` 运行。

训练：

```bash
python -m scripts.train
```

单任务预测：

```bash
python -m scripts.predict \
  --input_csv examples/inputs/data.csv \
  --checkpoint checkpoints/abs.pt
```

批量 `tztz` 预测：

```bash
python -m scripts.predict_tztz
```

消融实验：

```bash
python -m scripts.run_ablation --experiments baseline norwse
```

## 默认输入输出

- 样例输入放在 `examples/inputs/`
- 训练日志和测试预测放在 `outputs/runs/`
- 预测结果放在 `outputs/predictions/`
- 绘图和渲染结果放在 `outputs/figures/`
- 发布型或手工维护的模型权重放在 `checkpoints/`

## 实验脚本

`experiments/` 已按 manuscript 中 Fig. 1 到 Fig. 5 的顺序整理：

```text
experiments/
├── fig01_model_overview/
├── fig02_prediction_benchmark/
├── fig03_ablation/
├── fig04_abs_explainability/
└── fig05_fluoinvent_tztz_design/
```

每个目录内有独立 `README.md`，记录该图对应的实验目的、默认输入输出和运行命令。统一从仓库根目录运行，例如：

```bash
/home/panshangyang/ENTER/envs/graphgps/bin/python -m experiments.fig02_prediction_benchmark.plot_plqy_scatter
```

## 依赖说明

核心训练与预测依赖：

- `torch`
- `torch_geometric`
- `ogb`
- `pandas`
- `numpy`
- `scikit-learn`

实验可视化与渲染额外依赖：

- `matplotlib`
- `tensorboard`
- `rdkit`
- `Pillow`

## 仓库约定

- `scripts/` 只放入口和参数解析
- 训练逻辑留在 `train/`
- 数据图构造留在 `data/`
- 推理公共逻辑集中在 `utils/inference.py`
- `outputs/` 和 `checkpoints/` 视为本地工作目录，默认不纳入 Git

## 归档

根目录旧版说明文档已经移出主工作区。历史背景说明见 [legacy_notes.md](/data/panshangyang/FluoGPS/docs/archive/legacy_notes.md)。
