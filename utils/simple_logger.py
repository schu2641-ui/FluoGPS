"""Minimal TensorBoard logger used by the training scripts."""

import time

import numpy as np
import torch
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.tensorboard import SummaryWriter


class SimpleLogger:
    def __init__(self, name, output_dir="./runs", round_precision=8):
        self._true = []
        self._pred = []
        self._loss = 0.0
        self._size_current = 0
        self._lr = 0.0
        self._time_used = 0.0
        self.name = name
        self.round_precision = round_precision
        self.task_type = "regression"
        self.tb_writer = SummaryWriter(log_dir=f"{output_dir}/{name}")

    def update_stats(self, true, pred, loss, lr, time_used, params, dataset_name=None, **kwargs):
        assert true.shape[0] == pred.shape[0], (
            f"Shape mismatch between true ({true.shape[0]}) and pred ({pred.shape[0]})"
        )

        batch_size = true.shape[0]
        self._true.append(true)
        self._pred.append(pred)
        self._loss += loss * batch_size
        self._size_current += batch_size
        self._lr = lr
        self._time_used += time_used

    def _compute_basic_stats(self):
        avg_loss = self._loss / self._size_current if self._size_current > 0 else 0.0
        return {
            "loss": round(avg_loss, self.round_precision),
            "lr": round(self._lr, self.round_precision),
        }

    def _compute_spearman(self, y_true, y_pred):
        res_list = []
        if y_true.ndim == 1:
            if len(y_true) > 1:
                res_list.append(stats.spearmanr(y_true, y_pred)[0])
        else:
            for index in range(y_true.shape[1]):
                is_labeled = ~np.isnan(y_true[:, index])
                if is_labeled.sum() > 1:
                    res_list.append(stats.spearmanr(y_true[is_labeled, index], y_pred[is_labeled, index])[0])
        return sum(res_list) / len(res_list) if res_list else 0.0

    def _compute_regression_metrics(self):
        true = torch.cat(self._true)
        pred = torch.cat(self._pred)
        true_np = true.numpy()
        pred_np = pred.numpy()
        mae = mean_absolute_error(true_np, pred_np)
        mse = mean_squared_error(true_np, pred_np)
        rmse = float(np.sqrt(mse))
        r2 = r2_score(true_np, pred_np, multioutput="uniform_average")
        spearmanr = self._compute_spearman(true_np, pred_np)
        return {
            "mae": round(float(mae), self.round_precision),
            "mse": round(float(mse), self.round_precision),
            "rmse": round(rmse, self.round_precision),
            "r2": round(float(r2), self.round_precision),
            "spearmanr": round(float(spearmanr), self.round_precision),
        }

    def _write_to_tensorboard(self, stats_dict, epoch):
        for key, value in stats_dict.items():
            if key != "epoch":
                self.tb_writer.add_scalar(f"{self.name}/{key}", value, epoch)
        self.tb_writer.flush()

    def write_epoch(self, cur_epoch):
        _ = time.perf_counter()
        basic_stats = self._compute_basic_stats()
        regression_stats = self._compute_regression_metrics()
        if self.name == "train":
            stats_dict = {
                "epoch": cur_epoch,
                "time_epoch": round(self._time_used, self.round_precision),
                **basic_stats,
                **regression_stats,
            }
        else:
            stats_dict = {
                "epoch": cur_epoch,
                **basic_stats,
                **regression_stats,
            }

        self._write_to_tensorboard(stats_dict, cur_epoch)
        self.reset()
        return stats_dict

    def reset(self):
        self._true = []
        self._pred = []
        self._loss = 0.0
        self._size_current = 0
        self._time_used = 0.0

    def close(self):
        if hasattr(self, "tb_writer"):
            self.tb_writer.close()


def create_simple_logger(output_dir="./runs", num_splits=3):
    loggers = []
    names = ["train", "val", "test"]
    for index in range(num_splits):
        loggers.append(SimpleLogger(name=names[index], output_dir=output_dir))
    return loggers
