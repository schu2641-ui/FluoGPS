import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from data import FluorescenceDataset, compute_normalization_params
from torch_geometric.loader import DataLoader

def create_loader( train_dataset_dir, val_dataset_dir, test_dataset_dir,
                  batch_size, num_workers=8, kernel=None):

    y_mean, y_std = compute_normalization_params(train_dataset_dir, target_col='abs')
    print()

    # ⭐ 定义 RWSE transform 函数
    def add_rwse_transform(data):
        """在数据加载时动态添加 RWSE PE"""
        if kernel is not None:
            from data.loader import get_rw_landing_probs
            rwse = get_rw_landing_probs(
                ksteps=kernel,
                edge_index=data.edge_index,
                num_nodes=data.num_nodes
            )
            data.pestat_RWSE = rwse
        return data

    train_dataset = FluorescenceDataset(
        csv_file=train_dataset_dir,
        kernel=kernel,
        y_mean=y_mean,
        y_std=y_std,
        transform=add_rwse_transform  # ⭐ 添加 transform
    )

    print("\nLoading validation dataset...")
    val_dataset = FluorescenceDataset(
        csv_file=val_dataset_dir,
        kernel=kernel,
        y_mean=y_mean,
        y_std=y_std,
        transform=add_rwse_transform  # ⭐ 添加 transform
    )

    print("\nLoading test dataset...")
    test_dataset = FluorescenceDataset(
        csv_file=test_dataset_dir,
        kernel=kernel,
        y_mean=y_mean,
        y_std=y_std,
        transform=add_rwse_transform  # ⭐ 添加 transform
    )


    # 创建 loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


def custom_train(model, loaders, loggers, optimizer, scheduler, max_epoch):
    """
    自定义训练循环

    Args:
        model: 模型
        loaders: [train_loader, val_loader, test_loader]
        loggers: logger 实例
        optimizer: 优化器
        scheduler: 学习率调度器
        max_epoch: 最大训练轮数
    """
    train_loader, val_loader, test_loader = loaders
    device = next(model.parameters()).device

    best_val_loss = float('inf')
    best_val_mae = float('inf')
    patience_counter = 0
    patience = 50

    for epoch in range(max_epoch):
        # ==================== 训练阶段 ====================
        model.train()
        train_loss_sum = 0
        train_preds = []
        train_targets = []

        for batch in train_loader:
            batch = batch.to(device)

            optimizer.zero_grad()
            pred, true = model(batch)  # ⭐ 模型返回 (pred, true) 元组

            # 计算损失 (在归一化尺度上)
            loss = F.l1_loss(pred.squeeze(-1), true)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * batch.num_graphs

            # 收集预测结果用于计算 MAE (在真实尺度上)
            train_preds.append(pred.detach().cpu())
            train_targets.append(true.detach().cpu())

        # 计算训练指标
        train_loss = train_loss_sum / len(train_loader.dataset)

        # 反归一化预测和目标
        train_preds_tensor = torch.cat(train_preds, dim=0)
        train_targets_tensor = torch.cat(train_targets, dim=0)

        # 从数据集获取归一化参数
        train_preds_denorm = denormalize_predictions(
            train_preds_tensor,
            train_loader.dataset.y_mean,
            train_loader.dataset.y_std
        )
        train_targets_denorm = denormalize_predictions(
            train_targets_tensor,
            train_loader.dataset.y_mean,
            train_loader.dataset.y_std
        )

        train_mae = F.l1_loss(train_preds_denorm.squeeze(-1), train_targets_denorm).item()

        # ==================== 验证阶段 ====================
        model.eval()
        val_loss_sum = 0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred, true = model(batch)  # ⭐ 解包元组

                loss = F.l1_loss(pred.squeeze(-1), true)
                val_loss_sum += loss.item() * batch.num_graphs

                val_preds.append(pred.cpu())
                val_targets.append(true.cpu())

        val_loss = val_loss_sum / len(val_loader.dataset)

        # 反归一化验证预测
        val_preds_tensor = torch.cat(val_preds, dim=0)
        val_targets_tensor = torch.cat(val_targets, dim=0)

        val_preds_denorm = denormalize_predictions(
            val_preds_tensor,
            val_loader.dataset.y_mean,
            val_loader.dataset.y_std
        )
        val_targets_denorm = denormalize_predictions(
            val_targets_tensor,
            val_loader.dataset.y_mean,
            val_loader.dataset.y_std
        )

        val_mae = F.l1_loss(val_preds_denorm.squeeze(-1), val_targets_denorm).item()

        # ==================== 测试阶段 ====================
        model.eval()
        test_preds = []
        test_targets = []

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                pred, true = model(batch)  # ⭐ 解包元组

                test_preds.append(pred.cpu())
                test_targets.append(true.cpu())

        # 反归一化测试预测
        test_preds_tensor = torch.cat(test_preds, dim=0)
        test_targets_tensor = torch.cat(test_targets, dim=0)

        test_preds_denorm = denormalize_predictions(
            test_preds_tensor,
            test_loader.dataset.y_mean,
            test_loader.dataset.y_std
        )
        test_targets_denorm = denormalize_predictions(
            test_targets_tensor,
            test_loader.dataset.y_mean,
            test_loader.dataset.y_std
        )

        test_mae = F.l1_loss(test_preds_denorm.squeeze(-1), test_targets_denorm).item()

        # ==================== 日志记录 ====================
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = optimizer.param_groups[0]['lr']

        print(f'Epoch {epoch:03d} | '
              f'Train Loss: {train_loss:.4f} | '
              f'Train MAE: {train_mae:.4f} | '
              f'Val Loss: {val_loss:.4f} | '
              f'Val MAE: {val_mae:.4f} | '
              f'Test MAE: {test_mae:.4f} | '
              f'LR: {current_lr:.6f}')

        # 记录到 logger (手动写入 TensorBoard)
        for logger in loggers:
            logger.tb_writer.add_scalar(f'{logger.name}/loss',
                                        train_loss if logger.name == 'train' else
                                        val_loss if logger.name == 'val' else test_mae, epoch)
            logger.tb_writer.add_scalar(f'{logger.name}/mae',
                                        train_mae if logger.name == 'train' else
                                        val_mae if logger.name == 'val' else test_mae, epoch)
            logger.tb_writer.add_scalar(f'{logger.name}/lr', current_lr, epoch)
            logger.tb_writer.flush()

        # ==================== 早停检查 ====================
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_val_loss = val_loss
            patience_counter = 0

            # 保存最佳模型
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'val_mae': best_val_mae,
            }

            checkpoint_path = os.path.join(loggers[0].tb_writer.log_dir, 'best_model.pt')
            torch.save(checkpoint, checkpoint_path)
            print(f'  ✅ New best model saved! Val MAE: {best_val_mae:.4f}')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'\n Early stopping at epoch {epoch}')
                break

    print('\n' + '='*60)
    print('Training completed!')
    print(f'Best Val MAE: {best_val_mae:.4f}')
    print('='*60)


def denormalize_predictions(pred, mean, std):
    """
    反归一化预测结果

    Args:
        pred: 归一化的预测值
        mean: 原始数据的均值
        std: 原始数据的标准差

    Returns:
        反归一化后的预测值
    """
    return pred * std + mean
