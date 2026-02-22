import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

# 1. 设置绘图风格 (出版级配置)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['mathtext.fontset'] = 'stix' 

def plot_log_scatter(csv_file):
    # --- 读取数据 ---
    try:
        df = pd.read_csv(csv_file)
        df.columns = [c.strip() for c in df.columns] # 去除列名空格
        
        raw_true = df['true']
        raw_pred = df['pred']
    except FileNotFoundError:
        print(f"错误: 找不到文件 {csv_file}")
        return

    # --- 关键步骤：数据预处理 (Log10) ---
    # 1. 过滤掉 <= 0 的非法值 (对数不能处理 0 或负数)
    valid_mask = (raw_true > 0) & (raw_pred > 0)
    if not valid_mask.all():
        print(f"警告: 过滤了 {len(raw_true) - valid_mask.sum()} 个非正数值")
    
    y_true_valid = raw_true[valid_mask]
    y_pred_valid = raw_pred[valid_mask]

    # 2. 取以 10 为底的对数
    y_true_log = np.log10(y_true_valid)
    y_pred_log = np.log10(y_pred_valid)

    # --- 计算指标 (基于 Log 后的数据) ---
    mae = mean_absolute_error(y_true_log, y_pred_log)
    r2 = r2_score(y_true_log, y_pred_log)

    # --- 开始绘图 ---
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)

    # 1. 绘制对角线 (灰色虚线)
    # 根据图片，范围大约在 2 到 6 之间
    ax.plot([2, 6], [2, 6], color='gray', linestyle=':', linewidth=1.5, zorder=1)

    # 2. 绘制散点
    # c='#FDB462' 是非常接近你图片的淡橙色
    # s=3 点的大小
    ax.scatter(y_true_log, y_pred_log, c='#98B7C0', s=4, alpha=0.8, edgecolors='none', zorder=2)

    # --- 标签与文本 ---
    
    # 标题
    ax.set_title(r'$\varepsilon_{max}$', fontsize=20, pad=10)

    # 坐标轴标签 (显示 log10)
    ax.set_xlabel(r'Experimental $\log_{10}\varepsilon_{max}$', fontsize=16)
    ax.set_ylabel(r'Predicted $\log_{10}\varepsilon_{max}$', fontsize=16)

    # 统计文本 (MAE 和 R2)
    text_str = f'MAE = {mae:.2f}\n$R^2$ = {r2:.2f}'
    # 放在左上角
    ax.text(0.05, 0.95, text_str, transform=ax.transAxes, 
            fontsize=16, verticalalignment='top', horizontalalignment='left')

    # --- 坐标轴美化 ---
    
    # 设置范围 (根据 Log 后的数值范围，通常 2-6 涵盖了 100 到 1,000,000)
    ax.set_xlim(2, 6)
    ax.set_ylim(2, 6)

    # 设置刻度 (向外，四周都有)
    ax.tick_params(direction='out', length=4, width=1.2, colors='black',
                   grid_color='black', grid_alpha=0.5, labelsize=14,
                   top=True, right=True)

    # 加粗边框
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    # 保持正方形比例
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.savefig('scatter_plot_log_epsilon.png', bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # 确保你的 CSV 文件里是原始数据 (例如 30000, 50000 等)
    plot_log_scatter('test_predictions.csv')