import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

def calculate_metrics(csv_file):
    df = pd.read_csv(csv_file)
    
    df.columns = [c.strip() for c in df.columns]
    
    y_true = df['true']
    y_pred = df['pred']

    mse = mean_squared_error(y_true, y_pred)
    
    rmse = np.sqrt(mse)

    print("-" * 30)
    print(f"文件: {csv_file}")
    print("-" * 30)
    print(f"MSE  : {mse:.6f}")
    print(f"RMSE : {rmse:.6f}")
    print("-" * 30)

if __name__ == "__main__":
    calculate_metrics('test_predictions.csv')