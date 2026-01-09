"""
Simple Logger for Regression Tasks

A minimal logger implementation for regression tasks without GraphGym dependencies.
Supports console logging and TensorBoard integration.
"""

import logging
import time

import numpy as np
import torch
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                               r2_score, root_mean_squared_error)
from scipy.stats import stats
from torch.utils.tensorboard import SummaryWriter


class SimpleLogger:
    """
    Minimal logger for regression tasks.

    Tracks predictions and targets across batches, computes regression metrics,
    outputs to console and TensorBoard.

    Args:
        name: Logger name ('train', 'val', or 'test')
        output_dir: Base directory for TensorBoard logs
        round_precision: Number of decimal places for rounding (default: 8)
    """

    def __init__(self, name, output_dir='./runs', round_precision=8):
        # State tracking
        self._true = []           # List to accumulate true values
        self._pred = []           # List to accumulate predictions
        self._loss = 0.0          # Accumulated loss (sum of batch losses)
        self._size_current = 0    # Total number of samples processed
        self._lr = 0.0            # Current learning rate
        self._time_used = 0.0     # Time used in current epoch

        # Metadata
        self.name = name
        self.round_precision = round_precision
        self.task_type = 'regression'  # Fixed to regression

        # TensorBoard writer
        self.tb_writer = SummaryWriter(log_dir=f"{output_dir}/{name}")

    def update_stats(self, true, pred, loss, lr, time_used, params,
                     dataset_name=None, **kwargs):
        """
        Update statistics with batch data.

        Args:
            true: Ground truth values (Tensor) - detached, on CPU
            pred: Predicted values (Tensor) - detached, on CPU
            loss: Batch loss value (float)
            lr: Current learning rate (float)
            time_used: Time taken for this batch (float)
            params: Number of parameters in model (int) - kept for compatibility
            dataset_name: Dataset name (optional, unused in regression)
            **kwargs: Additional custom metrics (optional)
        """
        # Validate input shapes
        assert true.shape[0] == pred.shape[0], \
            f"Shape mismatch between true ({true.shape[0]}) and pred ({pred.shape[0]})"

        batch_size = true.shape[0]

        # Accumulate predictions and targets
        self._true.append(true)
        self._pred.append(pred)

        # Accumulate loss (weighted by batch size for accurate averaging)
        self._loss += loss * batch_size

        # Update counters
        self._size_current += batch_size
        self._lr = lr
        self._time_used += time_used

        # Note: params and kwargs are accepted for API compatibility but not used
        # in the current minimal implementation

    def _compute_basic_stats(self):
        """Compute basic statistics (loss, lr)."""
        avg_loss = self._loss / self._size_current if self._size_current > 0 else 0.0

        return {
            'loss': round(avg_loss, self.round_precision),
            'lr': round(self._lr, self.round_precision)
        }

    def _compute_spearman(self, y_true, y_pred):
        """
        Compute Spearman Rho averaged across tasks.

        Handles both 1D and multi-dimensional outputs.
        Ignores NaN values in multi-dimensional targets.

        Args:
            y_true: Ground truth values (numpy array)
            y_pred: Predicted values (numpy array)

        Returns:
            Average Spearman correlation coefficient
        """
        res_list = []

        if y_true.ndim == 1:
            # Single output
            if len(y_true) > 1:  # Need at least 2 points for correlation
                res_list.append(stats.spearmanr(y_true, y_pred)[0])
        else:
            # Multi-dimensional output - average across dimensions
            for i in range(y_true.shape[1]):
                # Ignore NaN values
                is_labeled = ~np.isnan(y_true[:, i])
                if is_labeled.sum() > 1:  # Need at least 2 points for correlation
                    res_list.append(stats.spearmanr(
                        y_true[is_labeled, i],
                        y_pred[is_labeled, i]
                    )[0])

        # Average across all dimensions
        return sum(res_list) / len(res_list) if res_list else 0.0

    def _compute_regression_metrics(self):
        """Compute regression-specific metrics."""
        # Concatenate all batches
        true = torch.cat(self._true)
        pred = torch.cat(self._pred)

        # Convert to numpy for sklearn
        true_np = true.numpy()
        pred_np = pred.numpy()

        # Compute metrics
        mae = mean_absolute_error(true_np, pred_np)
        mse = mean_squared_error(true_np, pred_np)
        rmse = root_mean_squared_error(true_np, pred_np)
        r2 = r2_score(true_np, pred_np, multioutput='uniform_average')

        # Handle Spearman correlation (handles multi-dimensional outputs)
        spearmanr = self._compute_spearman(true_np, pred_np)

        return {
            'mae': round(float(mae), self.round_precision),
            'mse': round(float(mse), self.round_precision),
            'rmse': round(float(rmse), self.round_precision),
            'r2': round(float(r2), self.round_precision),
            'spearmanr': round(float(spearmanr), self.round_precision)
        }

    def _write_to_tensorboard(self, stats, epoch):
        """
        Write metrics to TensorBoard.

        Args:
            stats: Dictionary of metrics
            epoch: Current epoch number
        """
        for key, value in stats.items():
            if key != 'epoch':  # Don't log epoch as a scalar
                self.tb_writer.add_scalar(f'{self.name}/{key}', value, epoch)
        self.tb_writer.flush()

    def write_epoch(self, cur_epoch):
        """
        Compute and log metrics for the current epoch.

        Args:
            cur_epoch: Current epoch number (int)

        Returns:
            stats: Dictionary of computed metrics
        """
        start_time = time.perf_counter()

        # Compute all metrics
        basic_stats = self._compute_basic_stats()
        regression_stats = self._compute_regression_metrics()

        # Combine stats
        if self.name == 'train':
            # Only train gets timing info
            stats = {
                'epoch': cur_epoch,
                'time_epoch': round(self._time_used, self.round_precision),
                **basic_stats,
                **regression_stats
            }
        else:
            stats = {
                'epoch': cur_epoch,
                **basic_stats,
                **regression_stats
            }

        # ⭐ 注释掉终端输出,只保留 TensorBoard (训练循环中有更好的输出)
        # logging.info(f'{self.name}: {stats}')

        # Output to TensorBoard
        self._write_to_tensorboard(stats, cur_epoch)

        # Reset state for next epoch
        self.reset()

        # ⭐ 注释掉计算时间日志
        # if cur_epoch < 3:
        #     compute_time = time.perf_counter() - start_time
        #     logging.info(f"...computing epoch stats took: {compute_time:.2f}s")

        return stats

    def reset(self):
        """Reset all epoch-specific state."""
        self._true = []
        self._pred = []
        self._loss = 0.0
        self._size_current = 0
        self._time_used = 0.0

    def close(self):
        """Close TensorBoard writer and cleanup."""
        if hasattr(self, 'tb_writer'):
            self.tb_writer.close()


def create_simple_logger(output_dir='./runs', num_splits=3):
    """
    Create simple loggers for the experiment.

    Args:
        output_dir: Base directory for TensorBoard logs
        num_splits: Number of splits (default: 3 for train/val/test)

    Returns:
        List of SimpleLogger objects [train_logger, val_logger, test_logger]
    """
    loggers = []
    names = ['train', 'val', 'test']

    for i in range(num_splits):
        logger = SimpleLogger(
            name=names[i],
            output_dir=output_dir
        )
        loggers.append(logger)

    return loggers
