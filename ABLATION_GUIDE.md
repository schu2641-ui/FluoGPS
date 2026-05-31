# 消融实验使用指南

## 快速开始

### 1. 运行预定义的消融实验

```bash
# 运行所有预定义消融实验
python run_ablation.py

# 运行特定消融实验
python run_ablation.py --experiments rwse_off local_only layers_4

# 自定义参数
python run_ablation.py \
    --batch_size 64 \
    --max_epoch 50 \
    --lr 0.001 \
    --device cuda:0
```

### 2. 自定义消融实验

#### 方法1: 修改配置文件

编辑 `ablation_config.py`，添加新的实验配置：

```python
ABLATION_EXPERIMENTS["my_experiment"] = AblationConfig(
    exp_name="my_experiment",
    use_rwse=False,
    num_layers=6,
    dropout=0.2
)
```

#### 方法2: 网格搜索

创建脚本 `grid_search.py`:

```python
from ablation_config import generate_grid_search_configs, AblationConfig
from run_ablation import run_ablation_experiments

# 定义参数网格
param_grid = {
    'num_layers': [2, 4, 6, 8],
    'dropout': [0.0, 0.1, 0.2],
    'num_heads': [2, 4]
}

# 生成配置
configs = generate_grid_search_configs(param_grid)

# 运行实验
for exp_name, config in configs:
    run_single_experiment(config, args, save_dir)
```

### 3. 分析结果

```bash
# 分析最新结果
python analyze_ablation.py

# 分析指定结果
python analyze_ablation.py ./ablation_results/20240101_120000
```

## 实验分类

### A. 位置编码消融 (Priority: High)
- `rwse_off`: 完全移除RWSE位置编码
- `rwse_dim_10/40`: 改变RWSE维度
- `rwse_steps_8/32`: 改变随机游走步数

### B. GPS层结构消融 (Priority: High)
- `local_only`: 只保留局部MPNN
- `global_only`: 只保留全局注意力

### C. 模型深度消融 (Priority: High)
- `layers_2/4/6/8`: 测试不同层数

### D. 注意力机制消融 (Priority: Medium)
- `heads_1/2/8`: 测试不同注意力头数
- `attn_dropout_0/2`: 测试不同注意力dropout

### E. 特征编码消融 (Priority: Medium)
- `no_edge_encoder`: 移除边特征编码

### F. 预测头消融 (Priority: Medium)
- `pooling_max/add`: 测试不同池化方式
- `head_layers_1/3`: 测试不同预测头层数

### G. Dropout消融 (Priority: Medium)
- `dropout_01/02/05`: 测试不同dropout率

### H. 训练策略消融 (Priority: Low)
- `no_normalization`: 移除数据归一化
- `no_warmup`: 移除学习率warmup
- `loss_mse`: 使用MSE损失

## 结果示例

运行完成后，你会得到：

```
ablation_results/
├── 20240101_120000/
│   ├── ablation_summary.csv          # 汇总结果
│   ├── comparison.png                 # 性能对比图
│   ├── training_curves.png            # 训练曲线
│   ├── results_table.tex              # LaTeX表格
│   ├── baseline/                      # baseline实验
│   │   ├── config.json
│   │   ├── best_model.pt
│   │   └── training_history.csv
│   ├── rwse_off/                      # 移除RWSE实验
│   │   ├── config.json
│   │   ├── best_model.pt
│   │   └── training_history.csv
│   └── ...
```

## 常见问题

### Q1: 如何只运行部分实验？

```python
# 方法1: 命令行参数
python run_ablation.py --experiments rwse_off local_only

# 方法2: 修改run_ablation.py
ablation_types = ["rwse_off", "local_only", "layers_4"]
run_ablation_experiments(ablation_types=ablation_types)
```

### Q2: 如何并行运行实验？

创建 `parallel_run.py`:

```python
from multiprocessing import Pool
from run_ablation import run_single_experiment
from ablation_config import get_ablation_configs

def run_parallel(n_gpus=4):
    configs = get_ablation_configs()
    
    with Pool(n_gpus) as pool:
        pool.map(run_single_experiment_wrapper, configs)

def run_single_experiment_wrapper(args):
    exp_name, config = args
    # 分配GPU
    gpu_id = get_available_gpu()
    args.device = f"cuda:{gpu_id}"
    return run_single_experiment(config, args, save_dir)
```

### Q3: 如何添加自定义消融？

```python
# 在 ablation_config.py 中添加
ABLATION_EXPERIMENTS["custom_exp"] = AblationConfig(
    exp_name="custom_exp",
    # 自定义参数
    use_rwse=True,
    num_layers=6,
    dropout=0.15,
    local_weight=0.7,
    global_weight=0.3
)
```

### Q4: 如何可视化特定实验的训练过程？

```python
from analyze_ablation import AblationAnalyzer

analyzer = AblationAnalyzer('./ablation_results/20240101_120000')
analyzer.plot_training_curves(
    exp_names=['baseline', 'rwse_off', 'local_only'],
    save_path='selected_training_curves.png'
)
```

## 进阶用法

### 1. 渐进式消融

```python
# 逐步添加组件
experiments = [
    ("no_features", AblationConfig(use_atom_encoder=False, use_edge_encoder=False, use_rwse=False)),
    ("with_atom", AblationConfig(use_atom_encoder=True, use_edge_encoder=False, use_rwse=False)),
    ("with_atom_edge", AblationConfig(use_atom_encoder=True, use_edge_encoder=True, use_rwse=False)),
    ("full", AblationConfig(use_atom_encoder=True, use_edge_encoder=True, use_rwse=True)),
]
```

### 2. 组件交互消融

```python
# 测试RWSE和层数的交互
param_grid = {
    'use_rwse': [True, False],
    'num_layers': [2, 4, 6, 8, 10]
}
configs = generate_grid_search_configs(param_grid)
```

### 3. 统计显著性检验

```python
from scipy import stats
import numpy as np

# 运行多次实验
results = []
for seed in range(5):
    torch.manual_seed(seed)
    result = run_single_experiment(config, args, save_dir)
    results.append(result['best_val_mae'])

# t检验
baseline_results = [...]  # baseline的多次运行结果
t_stat, p_value = stats.ttest_ind(results, baseline_results)
print(f"p-value: {p_value:.4f}")
```

## 结果解读

### MAE变化分析

```
组件           MAE变化    相对变化   重要性
--------------------------------------------
rwse_off      +0.0234    +5.2%     高
local_only    +0.0187    +4.1%     高  
global_only   +0.0456   +10.1%     高
dropout_0.1   -0.0034    -0.8%     低
```

### 结论示例

1. **RWSE位置编码重要**: 移除后MAE上升5.2%
2. **Local+Global结合最好**: 单独使用效果都下降
3. **Dropout影响较小**: 在此任务上不明显
4. **最优层数**: 6-8层之间

## 发布论文

### 表格格式

运行 `analyze_ablation.py` 会自动生成LaTeX表格，可直接用于论文。

### 图表建议

1. **主图**: 性能对比条形图
2. **附录**: 训练曲线对比
3. **补充**: 组件重要性排序

## 注意事项

1. **控制变量**: 每次只改变一个组件
2. **随机种子**: 多次运行取平均
3. **统计检验**: 报告标准差和p值
4. **计算资源**: 合理安排实验顺序
5. **保存日志**: 记录所有实验细节
