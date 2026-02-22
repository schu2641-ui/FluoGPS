import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def calculate_log_metrics(csv_file):
    # --- 1. 读取数据 ---
    try:
        df = pd.read_csv(csv_file)
        # 清理列名空格
        df.columns = [c.strip() for c in df.columns]
    except FileNotFoundError:
        print(f"错误: 找不到文件 {csv_file}")
        return

    # 获取原始数据
    raw_true = df['true']
    raw_pred = df['pred']

    # --- 2. 预处理检查 ---
    # 检查是否有 <= 0 的数值，因为 log10(0) 或 log10(负数) 会报错或产生 NaN
    if (raw_true <= 0).any() or (raw_pred <= 0).any():
        print("警告: 数据中包含 0 或负数，无法直接取 log10。")
        print("正在自动过滤掉这些非法数值...")
        valid_mask = (raw_true > 0) & (raw_pred > 0)
        raw_true = raw_true[valid_mask]
        raw_pred = raw_pred[valid_mask]
        print(f"剩余有效数据点: {len(raw_true)}")

    # --- 3. 执行 Log10 转换 ---
    # 这一步是关键：将原始的数万的数值转换为 4.x 的数值
    log_true = np.log10(raw_true)
    log_pred = np.log10(raw_pred)

    # --- 4. 计算指标 (基于 Log 数据) ---
    mae = mean_absolute_error(log_true, log_pred)
    mse = mean_squared_error(log_true, log_pred)
    rmse = np.sqrt(mse)

    # --- 5. 输出结果 ---
    print("=" * 40)
    print(f"文件处理: {csv_file}")
    print(f"计算模式: Log10 Transformed Metrics (对数尺度)")
    print("-" * 40)
    print(f"MAE  (Log10): {mae:.4f}")
    print(f"MSE  (Log10): {mse:.4f}")
    print(f"RMSE (Log10): {rmse:.4f}")
    print("=" * 40)
    
    # 顺便打印一下原始尺度的平均误差，让你有个概念
    # 注意：这只是一个近似参考，不能直接用于论文指标
    print("\n(参考) 原始尺度下的平均偏差估计:")
    # 计算平均的相对倍数误差
    mean_ratio = np.mean(raw_pred / raw_true)
    print(f"平均预测值是真实值的 {mean_ratio:.2f} 倍")

if __name__ == "__main__":
    # 请确保你的文件名是正确的，这里用了你提到的 test_prediction.csv
    calculate_log_metrics('test_predictions.csv')