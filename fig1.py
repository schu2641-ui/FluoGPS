import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

# 1. 设置绘图风格 (可选，为了让字体更接近出版级)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['mathtext.fontset'] = 'stix' # 让数学公式字体更好看

def plot_scatter(csv_file):
    # --- 读取数据 ---
    try:
        df = pd.read_csv(csv_file)
        # 确保列名没有额外的空格
        df.columns = [c.strip() for c in df.columns]
        
        y_true = df['true']
        y_pred = df['pred']
    except FileNotFoundError:
        print(f"错误: 找不到文件 {csv_file}")
        # 为了演示，生成一些随机数据 (如果你没有文件，可以取消注释下面这块来测试)
        # np.random.seed(42)
        # y_true = np.random.rand(2000)
        # y_pred = y_true + np.random.normal(0, 0.1, 2000)
        # y_pred = np.clip(y_pred, 0, 1)
        return

    # --- 计算指标 ---
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # --- 开始绘图 ---
    # 创建正方形画布
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)

    # 1. 绘制对角线 (灰色虚线)
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1.5, zorder=1)

    # 2. 绘制散点
    # s=2 控制点的大小，alpha=0.6 控制透明度
    # c='#00BFFF' 是接近图中青蓝色的十六进制代码 (DeepSkyBlue)
    ax.scatter(y_true, y_pred, c="#E6882A", s=3, alpha=0.6, edgecolors='none', zorder=2)

    # --- 标签与文本 ---
    
    # 标题 (使用 LaTeX 语法)
    ax.set_title(r'$\Phi_{PL}$', fontsize=18, pad=10)

    # 坐标轴标签
    ax.set_xlabel(r'Experimental $\Phi_{PL}$', fontsize=16)
    ax.set_ylabel(r'Predicted $\Phi_{PL}$', fontsize=16)

    # 在图内添加统计文本
    # transform=ax.transAxes 意味着坐标使用 (0,0)到(1,1) 的相对坐标
    text_str = f'MAE = {mae:.2f}\n$R^2$ = {r2:.2f}'
    ax.text(0.1, 0.9, text_str, transform=ax.transAxes, 
            fontsize=16, verticalalignment='top', horizontalalignment='left')

    # --- 坐标轴美化 ---
    
    # 设置范围
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # 设置刻度样式 (方向向外，上下左右都有刻度)
    ax.tick_params(direction='out', length=4, width=1.2, colors='black',
                   grid_color='black', grid_alpha=0.5, labelsize=14,
                   top=True, right=True) # top=True, right=True 显示上右刻度

    # 加粗边框 (Spines)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    # 保持正方形比例
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    
    # 保存图片
    plt.savefig('scatter_plot_result.png', bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    plot_scatter('test_predictions.csv')