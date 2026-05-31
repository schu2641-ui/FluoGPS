"""
消融实验结果分析和可视化
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List


class AblationAnalyzer:
    """消融实验结果分析器"""
    
    def __init__(self, results_dir: str):
        self.results_dir = results_dir
        self.summary_df = None
        self.all_configs = {}
        
    def load_results(self):
        """加载所有实验结果"""
        summary_path = os.path.join(self.results_dir, 'ablation_summary.csv')
        if os.path.exists(summary_path):
            self.summary_df = pd.read_csv(summary_path, index_col=0)
        
        # 加载所有配置
        for exp_name in os.listdir(self.results_dir):
            exp_dir = os.path.join(self.results_dir, exp_name)
            if os.path.isdir(exp_dir):
                config_path = os.path.join(exp_dir, 'config.json')
                if os.path.exists(config_path):
                    import json
                    with open(config_path, 'r') as f:
                        self.all_configs[exp_name] = json.load(f)
        
        return self.summary_df
    
    def plot_comparison(self, metric='best_val_mae', save_path=None):
        """绘制不同实验的性能对比"""
        if self.summary_df is None:
            self.load_results()
        
        # 排序
        df_sorted = self.summary_df.sort_values(metric)
        
        # 绘制条形图
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(df_sorted)), df_sorted[metric])
        
        # 设置颜色
        colors = ['green' if i == 0 else 'steelblue' for i in range(len(df_sorted))]
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.xticks(range(len(df_sorted)), df_sorted.index, rotation=45, ha='right')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.xlabel('Experiment Name')
        plt.title('Ablation Study Results Comparison')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return df_sorted
    
    def plot_training_curves(self, exp_names: List[str] = None, save_path=None):
        """绘制训练曲线对比"""
        if exp_names is None:
            exp_names = list(self.all_configs.keys())[:5]  # 默认前5个
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        for exp_name in exp_names:
            history_path = os.path.join(self.results_dir, exp_name, 'training_history.csv')
            if os.path.exists(history_path):
                df = pd.read_csv(history_path)
                
                # 训练MAE
                axes[0].plot(df['train_maes'], label=exp_name, linewidth=2)
                
                # 验证MAE
                axes[1].plot(df['val_maes'], label=exp_name, linewidth=2)
        
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Train MAE')
        axes[0].set_title('Training MAE Comparison')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Validation MAE')
        axes[1].set_title('Validation MAE Comparison')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_latex_table(self, save_path=None):
        """生成LaTeX表格"""
        if self.summary_df is None:
            self.load_results()
        
        # 排序
        df_sorted = self.summary_df.sort_values('best_val_mae')
        
        # 生成LaTeX
        latex_code = df_sorted.to_latex(
            float_format="%.4f",
            caption="Ablation Study Results",
            label="tab:ablation_results"
        )
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(latex_code)
        
        print(latex_code)
        return latex_code
    
    def analyze_component_importance(self):
        """分析各组件的重要性"""
        if self.summary_df is None:
            self.load_results()
        
        # 找到baseline结果
        baseline_mae = self.summary_df.loc['baseline', 'best_val_mae']
        
        importance = {}
        for exp_name, row in self.summary_df.iterrows():
            if exp_name == 'baseline':
                continue
            
            mae_diff = row['best_val_mae'] - baseline_mae
            importance[exp_name] = {
                'mae_diff': mae_diff,
                'relative_change': (mae_diff / baseline_mae) * 100,
                'impact': 'positive' if mae_diff > 0 else 'negative'
            }
        
        # 排序
        importance_sorted = sorted(importance.items(), 
                                  key=lambda x: abs(x[1]['mae_diff']), 
                                  reverse=True)
        
        print("\nComponent Importance Analysis:")
        print("-" * 60)
        for exp_name, metrics in importance_sorted[:10]:
            print(f"{exp_name:20s}: {metrics['mae_diff']:+.4f} "
                  f"({metrics['relative_change']:+.2f}%)")
        
        return importance_sorted
    
    def create_heatmap(self, param1: str, param2: str, save_path=None):
        """创建参数热力图（用于网格搜索结果）"""
        if self.summary_df is None:
            self.load_results()
        
        # 提取参数值
        data = []
        for exp_name, row in self.summary_df.iterrows():
            config = self.all_configs.get(exp_name, {})
            val1 = config.get(param1)
            val2 = config.get(param2)
            mae = row['best_val_mae']
            
            if val1 is not None and val2 is not None:
                data.append({param1: val1, param2: val2, 'mae': mae})
        
        if len(data) == 0:
            print("No data for heatmap")
            return
        
        df = pd.DataFrame(data)
        pivot = df.pivot(param1, param2, 'mae')
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot, annot=True, fmt='.4f', cmap='YlOrRd_r')
        plt.title(f'{param1} vs {param2} - Validation MAE')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def quick_analysis(results_dir: str):
    """快速分析并生成报告"""
    analyzer = AblationAnalyzer(results_dir)
    
    print("\n" + "="*60)
    print("ABLATION STUDY ANALYSIS REPORT")
    print("="*60)
    
    # 1. 加载结果
    summary = analyzer.load_results()
    print("\n1. Summary Statistics:")
    print(summary.describe())
    
    # 2. 性能对比
    print("\n2. Performance Comparison (Top 5 Best):")
    best_5 = summary.nsmallest(5, 'best_val_mae')
    print(best_5[['best_val_mae', 'best_test_mae', 'best_epoch']])
    
    # 3. 组件重要性
    print("\n3. Component Importance:")
    importance = analyzer.analyze_component_importance()
    
    # 4. 生成可视化
    print("\n4. Generating visualizations...")
    analyzer.plot_comparison(save_path=os.path.join(results_dir, 'comparison.png'))
    analyzer.plot_training_curves(save_path=os.path.join(results_dir, 'training_curves.png'))
    
    # 5. 生成LaTeX表格
    print("\n5. Generating LaTeX table...")
    analyzer.generate_latex_table(save_path=os.path.join(results_dir, 'results_table.tex'))
    
    return analyzer


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        # 查找最新的结果目录
        base_dir = './ablation_results'
        if os.path.exists(base_dir):
            subdirs = [d for d in os.listdir(base_dir) 
                      if os.path.isdir(os.path.join(base_dir, d))]
            if subdirs:
                latest = max(subdirs)
                results_dir = os.path.join(base_dir, latest)
            else:
                results_dir = base_dir
        else:
            print("No results found. Please run experiments first.")
            sys.exit(1)
    
    quick_analysis(results_dir)
