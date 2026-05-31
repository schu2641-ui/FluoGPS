"""
消融实验训练脚本
"""
import os
import sys
import torch
import torch.nn.functional as F
import pandas as pd
import json
from datetime import datetime
from tqdm import tqdm

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.network.gps_model_ablation import GPSModelAblation
from train.trainer import create_loader
from ablation_config import AblationConfig, get_ablation_configs, generate_grid_search_configs
from simple_logger import create_simple_logger


def run_single_experiment(config: AblationConfig, args, save_dir):
    """运行单个消融实验"""
    
    print(f"\n{'='*60}")
    print(f"Running experiment: {config.exp_name}")
    print(f"{'='*60}")
    
    # 创建保存目录
    exp_dir = os.path.join(save_dir, config.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # 保存配置
    with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
    
    # 创建数据加载器
    if config.use_rwse:
        kernel = config.rwse_steps
    else:
        kernel = []  # 不使用RWSE
        
    train_loader, val_loader, test_loader = create_loader(
        train_dataset_dir=args.train_csv,
        val_dataset_dir=args.val_csv,
        test_dataset_dir=args.test_csv,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        kernel=kernel
    )
    
    # 设置归一化
    if not config.use_normalization:
        for loader in [train_loader, val_loader, test_loader]:
            loader.dataset.y_mean = 0.0
            loader.dataset.y_std = 1.0
    
    # 创建模型
    model = GPSModelAblation(config).to(args.device)
    
    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # 学习率调度器
    if config.use_warmup:
        from utils import schedule_with_warmup
        scheduler = schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=args.warm_steps,
            num_training_steps=args.max_epoch
        )
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=30, gamma=0.5
        )
    
    # 训练
    results = train_model(
        model=model,
        loaders=[train_loader, val_loader, test_loader],
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        args=args,
        exp_dir=exp_dir
    )
    
    return results


def train_model(model, loaders, optimizer, scheduler, config, args, exp_dir):
    """训练模型"""
    
    train_loader, val_loader, test_loader = loaders
    device = args.device
    
    best_val_mae = float('inf')
    best_test_mae = float('inf')
    patience_counter = 0
    patience = args.patience
    
    results = {
        'train_losses': [],
        'train_maes': [],
        'val_losses': [],
        'val_maes': [],
        'test_maes': [],
        'best_val_mae': None,
        'best_test_mae': None,
        'best_epoch': None
    }
    
    for epoch in range(args.max_epoch):
        # 训练
        model.train()
        train_loss_sum = 0
        train_preds = []
        train_targets = []
        
        for batch in train_loader:
            batch = batch.to(device)
            
            optimizer.zero_grad()
            pred, true = model(batch)
            
            # 选择损失函数
            if config.loss_type == "mae":
                loss = F.l1_loss(pred.squeeze(-1), true)
            elif config.loss_type == "mse":
                loss = F.mse_loss(pred.squeeze(-1), true)
            elif config.loss_type == "huber":
                loss = F.huber_loss(pred.squeeze(-1), true)
            else:
                loss = F.l1_loss(pred.squeeze(-1), true)
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss_sum += loss.item() * batch.num_graphs
            train_preds.append(pred.detach().cpu())
            train_targets.append(true.detach().cpu())
        
        train_loss = train_loss_sum / len(train_loader.dataset)
        
        # 反归一化计算MAE
        train_preds_tensor = torch.cat(train_preds, dim=0)
        train_targets_tensor = torch.cat(train_targets, dim=0)
        train_mae = denorm_mae(train_preds_tensor, train_targets_tensor,
                               train_loader.dataset.y_mean, train_loader.dataset.y_std)
        
        # 验证
        model.eval()
        val_loss_sum = 0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred, true = model(batch)
                
                if config.loss_type == "mae":
                    loss = F.l1_loss(pred.squeeze(-1), true)
                elif config.loss_type == "mse":
                    loss = F.mse_loss(pred.squeeze(-1), true)
                else:
                    loss = F.huber_loss(pred.squeeze(-1), true)
                    
                val_loss_sum += loss.item() * batch.num_graphs
                val_preds.append(pred.cpu())
                val_targets.append(true.cpu())
        
        val_loss = val_loss_sum / len(val_loader.dataset)
        val_preds_tensor = torch.cat(val_preds, dim=0)
        val_targets_tensor = torch.cat(val_targets, dim=0)
        val_mae = denorm_mae(val_preds_tensor, val_targets_tensor,
                            val_loader.dataset.y_mean, val_loader.dataset.y_std)
        
        # 测试
        test_mae = None
        if epoch >= int(args.max_epoch * 0.8):
            test_mae = evaluate_test(model, test_loader, device)
        
        # 更新学习率
        if scheduler is not None:
            scheduler.step()
            
        # 记录结果
        results['train_losses'].append(train_loss)
        results['train_maes'].append(train_mae)
        results['val_losses'].append(val_loss)
        results['val_maes'].append(val_mae)
        results['test_maes'].append(test_mae if test_mae else 0)
        
        # 打印进度
        if epoch % 10 == 0 or test_mae is not None:
            msg = f'Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Train MAE: {train_mae:.4f} | Val MAE: {val_mae:.4f}'
            if test_mae is not None:
                msg += f' | Test MAE: {test_mae:.4f}'
            print(msg)
        
        # 保存最佳模型
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_test_mae = test_mae if test_mae else best_test_mae
            best_epoch = epoch
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_mae': best_val_mae,
                'test_mae': best_test_mae,
            }, os.path.join(exp_dir, 'best_model.pt'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch}')
                break
    
    results['best_val_mae'] = best_val_mae
    results['best_test_mae'] = best_test_mae
    results['best_epoch'] = best_epoch
    
    # 保存结果
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(exp_dir, 'training_history.csv'), index=False)
    
    return results


def denorm_mae(pred, true, mean, std):
    """反归一化并计算MAE"""
    pred_denorm = pred * std + mean
    true_denorm = true * std + mean
    return F.l1_loss(pred_denorm.squeeze(-1), true_denorm).item()


def evaluate_test(model, test_loader, device):
    """评估测试集"""
    model.eval()
    test_preds = []
    test_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred, true = model(batch)
            test_preds.append(pred.cpu())
            test_targets.append(true.cpu())
    
    test_preds_tensor = torch.cat(test_preds, dim=0)
    test_targets_tensor = torch.cat(test_targets, dim=0)
    
    return denorm_mae(test_preds_tensor, test_targets_tensor,
                     test_loader.dataset.y_mean, test_loader.dataset.y_std)


def run_ablation_experiments(ablation_types=None, args=None):
    """运行多个消融实验"""
    
    # 设置参数
    if args is None:
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--train_csv', type=str, default='datasets/fluorescence/raw/e_train.csv')
        parser.add_argument('--val_csv', type=str, default='datasets/fluorescence/raw/e_valid.csv')
        parser.add_argument('--test_csv', type=str, default='datasets/fluorescence/raw/e_test.csv')
        parser.add_argument('--batch_size', type=int, default=128)
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--lr', type=float, default=0.0007)
        parser.add_argument('--weight_decay', type=float, default=0)
        parser.add_argument('--warm_steps', type=int, default=6)
        parser.add_argument('--max_epoch', type=int, default=100)
        parser.add_argument('--patience', type=int, default=50)
        parser.add_argument('--device', type=str, default='cuda')
        parser.add_argument('--save_dir', type=str, default='./ablation_results')
        args = parser.parse_args()
    
    # 创建保存目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.save_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    
    # 获取配置列表
    configs = get_ablation_configs(ablation_types)
    
    # 运行实验
    all_results = {}
    for exp_name, config in tqdm(configs, desc="Running experiments"):
        try:
            results = run_single_experiment(config, args, save_dir)
            all_results[exp_name] = {
                'best_val_mae': results['best_val_mae'],
                'best_test_mae': results['best_test_mae'],
                'best_epoch': results['best_epoch']
            }
        except Exception as e:
            print(f"\nError in experiment {exp_name}: {str(e)}")
            all_results[exp_name] = {'error': str(e)}
    
    # 保存汇总结果
    summary_df = pd.DataFrame(all_results).T
    summary_df.to_csv(os.path.join(save_dir, 'ablation_summary.csv'))
    
    print("\n" + "="*60)
    print("Ablation Experiments Completed!")
    print("="*60)
    print(summary_df)
    
    return all_results


if __name__ == "__main__":
    # 运行所有预定义的消融实验
    run_ablation_experiments()
