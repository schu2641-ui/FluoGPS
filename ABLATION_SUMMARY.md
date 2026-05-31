# FluoGPS消融实验完整方案总结

## 📁 已创建的文件

### 1. 核心配置文件
- **`ablation_config.py`** - 消融实验配置定义
  - 包含30+预定义消融实验
  - 支持网格搜索
  - 可扩展自定义实验

### 2. 支持消融的模型
- **`models/network/gps_model_ablation.py`** - 可配置的GPS模型
  - 支持RWSE位置编码开关
  - 支持Local/Global模块开关
  - 支持各种池化方式
  - 支持不同的训练策略

### 3. 实验运行脚本
- **`run_ablation.py`** - 批量运行消融实验
  - 自动化实验流程
  - 支持早停
  - 自动保存结果
  
- **`quick_start_ablation.py`** - 快速演示脚本
  - 3个关键实验演示
  - 单实验运行模式

### 4. 结果分析工具
- **`analyze_ablation.py`** - 结果分析和可视化
  - 性能对比图
  - 训练曲线对比
  - LaTeX表格生成
  - 组件重要性分析
  - 热力图生成

### 5. 使用文档
- **`ABLATION_GUIDE.md`** - 详细使用指南
  - 快速开始
  - 自定义实验
  - 结果分析
  - 常见问题

## 🚀 快速开始

### 第一步：运行演示（推荐）

```bash
cd /data/panshangyang/FluoGPS

# 运行3个关键实验的演示
python quick_start_ablation.py --mode demo
```

这将运行：
1. **baseline** - 完整模型
2. **rwse_off** - 移除位置编码
3. **local_only** - 只保留局部MPNN

预计时间：每个实验10-30分钟（取决于GPU）

### 第二步：分析结果

```bash
# 分析最新结果
python analyze_ablation.py
```

输出：
- `comparison.png` - 性能对比条形图
- `training_curves.png` - 训练曲线对比
- `results_table.tex` - LaTeX表格（可直接用于论文）
- `ablation_summary.csv` - 结果汇总

## 📊 实验优先级

### 高优先级（必做）

1. **位置编码消融**
   ```bash
   python run_ablation.py --experiments rwse_off rwse_dim_10 rwse_dim_40 rwse_steps_8 rwse_steps_32
   ```

2. **GPS层结构消融**
   ```bash
   python run_ablation.py --experiments local_only global_only
   ```

3. **模型深度消融**
   ```bash
   python run_ablation.py --experiments layers_2 layers_4 layers_6 layers_8
   ```

### 中优先级（推荐）

4. **注意力机制消融**
   ```bash
   python run_ablation.py --experiments heads_1 heads_2 heads_8 attn_dropout_0 attn_dropout_2
   ```

5. **预测头消融**
   ```bash
   python run_ablation.py --experiments pooling_max pooling_add head_layers_1 head_layers_3
   ```

### 低优先级（可选）

6. **Dropout消融**
   ```bash
   python run_ablation.py --experiments dropout_01 dropout_02 dropout_05
   ```

7. **训练策略消融**
   ```bash
   python run_ablation.py --experiments no_normalization no_warmup loss_mse
   ```

## 🔧 自定义实验

### 方法1：修改配置文件

编辑 `ablation_config.py`：

```python
ABLATION_EXPERIMENTS["my_custom_exp"] = AblationConfig(
    exp_name="my_custom_exp",
    num_layers=7,
    dropout=0.15,
    use_rwse=True,
    num_heads=3
)
```

然后运行：
```bash
python run_ablation.py --experiments my_custom_exp
```

### 方法2：网格搜索

创建 `my_grid_search.py`：

```python
from ablation_config import generate_grid_search_configs
from run_ablation import run_ablation_experiments

param_grid = {
    'num_layers': [4, 6, 8],
    'dropout': [0.0, 0.1, 0.2],
    'num_heads': [2, 4]
}

configs = generate_grid_search_configs(param_grid)
# ... 运行实验
```

## 📈 结果示例

### 性能对比表格

| Experiment | Val MAE | Test MAE | Improvement |
|------------|---------|----------|-------------|
| baseline   | 0.4512  | 0.4623   | -           |
| rwse_off   | 0.4746  | 0.4837   | -5.2%       |
| local_only | 0.4698  | 0.4785   | -4.1%       |
| layers_4   | 0.4567  | 0.4689   | -1.2%       |

### 组件重要性排序

```
组件               MAE变化    相对影响
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RWSE位置编码      +0.0234    高
全局注意力        +0.0186    高
局部MPNN          +0.0456    非常高
Dropout           -0.0034    低
```

## 💡 最佳实践

### 1. 实验设计

✅ **控制变量法**：每次只改变一个组件
✅ **多次运行**：至少3次不同随机种子
✅ **统计检验**：报告均值±标准差

### 2. 资源管理

```bash
# 限制GPU内存
export CUDA_VISIBLE_DEVICES=0,1

# 减少worker数量
python run_ablation.py --num_workers 2

# 使用更小的batch size
python run_ablation.py --batch_size 64
```

### 3. 结果保存

每个实验自动保存：
- `config.json` - 实验配置
- `best_model.pt` - 最佳模型检查点
- `training_history.csv` - 训练历史
- `ablation_summary.csv` - 汇总结果

## 🎯 论文写作建议

### 主实验部分

1. **Table 1**: Baseline性能
2. **Table 2**: 主要消融实验结果
3. **Figure 1**: 性能对比条形图
4. **Figure 2**: 训练曲线对比

### 补充材料

- 完整消融实验表格
- 组件重要性分析
- 超参数敏感性分析

### 写作模板

```latex
\subsection{Ablation Study}

To understand the contribution of each component, we conducted 
comprehensive ablation experiments. Table~\ref{tab:ablation} shows 
the results.

\textbf{Position Encoding:} Removing RWSE increases MAE by 5.2\%, 
indicating that structural position information is crucial for 
fluorescence prediction.

\textbf{Model Architecture:} The combination of local MPNN and 
global attention outperforms either component alone, demonstrating 
the importance of capturing both local and global molecular patterns.

\textbf{Model Depth:} Performance improves with more layers up to 
8 layers, then plateaus, suggesting 6-8 layers as optimal depth.
```

## 📞 技术支持

### 常见错误

1. **CUDA out of memory**
   ```bash
   # 减小batch size
   python run_ablation.py --batch_size 64
   ```

2. **找不到模块**
   ```bash
   # 确保在正确目录
   cd /data/panshangyang/FluoGPS
   
   # 安装依赖
   pip install torch-geometric seaborn
   ```

3. **数据文件不存在**
   ```bash
   # 检查数据路径
   ls datasets/fluorescence/raw/
   ```

### 调试模式

```python
# 在 quick_start_ablation.py 中设置
args.max_epoch = 5  # 快速测试
args.batch_size = 32
```

## 🎓 进阶话题

### 1. 渐进式消融

逐步添加组件，观察性能提升：

```python
experiments = [
    ("minimal", AblationConfig(use_rwse=False, num_layers=2)),
    ("+rwse", AblationConfig(use_rwse=True, num_layers=2)),
    ("+depth", AblationConfig(use_rwse=True, num_layers=6)),
    ("+dropout", AblationConfig(use_rwse=True, num_layers=6, dropout=0.1)),
]
```

### 2. 组件交互研究

```python
param_grid = {
    'use_rwse': [True, False],
    'use_global_attn': [True, False]
}
# 2x2 = 4个实验
```

### 3. 超参数调优

```python
param_grid = {
    'dim_hidden': [64, 128, 256],
    'num_heads': [2, 4, 8],
    'dropout': [0.0, 0.1, 0.2]
}
# 3x3x3 = 27个实验
```

## 📚 相关文献

1. **GPS论文**: Rampasek et al., "Recipe for a General, Powerful, Scalable Graph Transformer", NeurIPS 2022
2. **消融实验最佳实践**: Meyes et al., "Ablation Studies in Artificial Neural Networks", 2019

## ✅ 检查清单

运行消融实验前，请确保：

- [ ] 数据文件存在且路径正确
- [ ] GPU可用且有足够内存
- [ ] 已安装所有依赖包
- [ ] 理解了各个消融实验的目的
- [ ] 准备好记录实验结果

## 🎉 开始实验

```bash
# 1. 快速演示（3个实验，~30分钟）
python quick_start_ablation.py --mode demo

# 2. 完整消融（30+实验，~10小时）
python run_ablation.py

# 3. 自定义实验
python run_ablation.py --experiments rwse_off local_only layers_4

# 4. 分析结果
python analyze_ablation.py
```

祝实验顺利！🚀
