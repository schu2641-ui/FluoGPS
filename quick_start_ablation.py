"""
快速开始消融实验示例
"""
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ablation_config import AblationConfig
from run_ablation import run_single_experiment


def quick_demo():
    """快速演示：运行3个关键消融实验"""
    
    print("\n" + "="*60)
    print("FLUOGPS ABLATION STUDY - QUICK DEMO")
    print("="*60)
    print("\nRunning 3 key ablation experiments:")
    print("1. Baseline (default configuration)")
    print("2. Without RWSE position encoding")
    print("3. Local GNN only (no global attention)")
    print("\n" + "="*60 + "\n")
    
    # 设置参数
    import argparse
    args = argparse.Namespace(
        train_csv='datasets/fluorescence/raw/e_train.csv',
        val_csv='datasets/fluorescence/raw/e_valid.csv',
        test_csv='datasets/fluorescence/raw/e_test.csv',
        batch_size=128,
        num_workers=4,
        lr=0.0007,
        weight_decay=0,
        warm_steps=6,
        max_epoch=50,  # 快速演示，减少epoch
        patience=30,
        device='cuda',
        save_dir='./ablation_results'
    )
    
    # 创建保存目录
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.save_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    
    # 定义3个关键实验
    experiments = [
        ("baseline", AblationConfig(exp_name="baseline")),
        ("rwse_off", AblationConfig(exp_name="rwse_off", use_rwse=False)),
        ("local_only", AblationConfig(exp_name="local_only", use_global_attn=False)),
    ]
    
    # 运行实验
    results = {}
    for exp_name, config in experiments:
        try:
            print(f"\n{'='*60}")
            print(f"Experiment: {exp_name}")
            print(f"{'='*60}")
            
            result = run_single_experiment(config, args, save_dir)
            results[exp_name] = {
                'best_val_mae': result['best_val_mae'],
                'best_test_mae': result['best_test_mae'],
                'best_epoch': result['best_epoch']
            }
            
            print(f"\n{exp_name} Results:")
            print(f"  Best Val MAE:  {result['best_val_mae']:.4f}")
            print(f"  Best Test MAE: {result['best_test_mae']:.4f}")
            print(f"  Best Epoch:    {result['best_epoch']}")
            
        except Exception as e:
            print(f"\nError in {exp_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            results[exp_name] = {'error': str(e)}
    
    # 打印汇总结果
    print("\n" + "="*60)
    print("SUMMARY RESULTS")
    print("="*60)
    
    import pandas as pd
    summary_df = pd.DataFrame(results).T
    print(summary_df)
    
    # 保存汇总结果
    summary_df.to_csv(os.path.join(save_dir, 'ablation_summary.csv'))
    
    # 计算性能变化
    if 'baseline' in results and 'best_val_mae' in results['baseline']:
        baseline_mae = results['baseline']['best_val_mae']
        print(f"\nPerformance Change (relative to baseline):")
        for exp_name, result in results.items():
            if exp_name != 'baseline' and 'best_val_mae' in result:
                mae_diff = result['best_val_mae'] - baseline_mae
                rel_change = (mae_diff / baseline_mae) * 100
                print(f"  {exp_name:15s}: {mae_diff:+.4f} ({rel_change:+.2f}%)")
    
    print("\n" + "="*60)
    print("Demo completed! Results saved to:", save_dir)
    print("="*60)
    
    return results


def single_experiment_demo():
    """运行单个实验示例"""
    
    # 创建配置
    config = AblationConfig(
        exp_name="my_experiment",
        num_layers=6,
        dropout=0.1,
        use_rwse=True
    )
    
    # 设置参数
    import argparse
    args = argparse.Namespace(
        train_csv='datasets/fluorescence/raw/e_train.csv',
        val_csv='datasets/fluorescence/raw/e_valid.csv',
        test_csv='datasets/fluorescence/raw/e_test.csv',
        batch_size=128,
        num_workers=4,
        lr=0.0007,
        weight_decay=0,
        warm_steps=6,
        max_epoch=100,
        patience=50,
        device='cuda',
        save_dir='./experiment_results'
    )
    
    # 运行
    result = run_single_experiment(config, args, args.save_dir)
    
    print("\nResults:")
    print(f"  Best Val MAE:  {result['best_val_mae']:.4f}")
    print(f"  Best Test MAE: {result['best_test_mae']:.4f}")
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='demo',
                       choices=['demo', 'single'],
                       help='Run mode: demo (3 experiments) or single')
    args = parser.parse_args()
    
    if args.mode == 'demo':
        quick_demo()
    elif args.mode == 'single':
        single_experiment_demo()
