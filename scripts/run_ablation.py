"""Run FluoGPS ablations on selected fluorescence tasks."""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
import random
import sys
from datetime import datetime
from multiprocessing import Process, Queue
from pathlib import Path
from queue import Empty


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

DATA_DIR = ROOT_DIR / "datasets" / "fluorescence" / "raw"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "outputs" / "runs" / "ablation"

RWSE_STEPS = list(range(1, 17))
TASKS = ("abs", "emi")
EXPERIMENTS = {
    "baseline": {"use_rwse": True, "rwse_steps": RWSE_STEPS, "use_local": True, "use_global": True, "use_solvent": True},
    "local": {"use_rwse": True, "rwse_steps": RWSE_STEPS, "use_local": True, "use_global": False, "use_solvent": True},
    "global": {"use_rwse": True, "rwse_steps": RWSE_STEPS, "use_local": False, "use_global": True, "use_solvent": True},
    "norwse": {"use_rwse": False, "rwse_steps": RWSE_STEPS, "use_local": True, "use_global": True, "use_solvent": True},
    "no_solvent": {"use_rwse": True, "rwse_steps": RWSE_STEPS, "use_local": True, "use_global": True, "use_solvent": False},
}


def parse_args():
    parser = argparse.ArgumentParser(description="FluoGPS ablation runner")
    parser.add_argument("--tasks", nargs="+", default=["abs", "emi"], choices=TASKS)
    parser.add_argument("--experiments", nargs="+", default=list(EXPERIMENTS), choices=list(EXPERIMENTS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--gpus", type=int, nargs="+", default=None)
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max_epoch", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--dim_hidden", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=10)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--attn_dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--warm_steps", type=int, default=15)
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


def dataset_paths(task):
    return {
        "train": DATA_DIR / f"{task}_train.csv",
        "val": DATA_DIR / f"{task}_valid.csv",
        "test": DATA_DIR / f"{task}_test.csv",
    }


def set_seed(seed):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def run_one(task, exp_name, cfg, seed, gpu_id, args, output_dir):
    from models.network.gps_model_ablation import GPSModelAblation
    from train.trainer import create_loader, custom_train
    from utils import schedule_with_warmup
    from utils.simple_logger import create_simple_logger
    import torch

    set_seed(seed)
    paths = dataset_paths(task)
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    log_dir = output_dir / "logs" / task / exp_name / f"seed_{seed}"
    log_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader = create_loader(
        train_dataset_dir=str(paths["train"]),
        val_dataset_dir=str(paths["val"]),
        test_dataset_dir=str(paths["test"]),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        kernel=cfg["rwse_steps"],
        use_solvent=cfg["use_solvent"],
    )

    model = GPSModelAblation(
        dim_h=args.dim_hidden,
        num_heads=args.num_heads,
        dropout=args.dropout,
        attn_dropout=args.attn_dropout,
        num_layers=args.num_layers,
        dim_out=1,
        ablation_cfg={key: cfg[key] for key in ("use_rwse", "rwse_steps", "use_local", "use_global")},
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)
    scheduler = schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.warm_steps,
        num_training_steps=args.max_epoch,
    )

    return custom_train(
        model=model,
        loaders=[train_loader, val_loader, test_loader],
        loggers=create_simple_logger(output_dir=str(log_dir)),
        optimizer=optimizer,
        scheduler=scheduler,
        max_epoch=args.max_epoch,
    )


def worker(gpu_id, task_queue, result_queue, args, output_dir):
    while True:
        item = task_queue.get()
        if item is None:
            break

        task_id, task, exp_name, cfg, seed = item
        try:
            result = run_one(task, exp_name, cfg, seed, gpu_id, args, output_dir)
            result_queue.put(
                {
                    "task_id": task_id,
                    "task": task,
                    "experiment": exp_name,
                    "seed": seed,
                    "success": True,
                    "test_mae": result["final_test_mae"],
                }
            )
        except Exception as exc:
            result_queue.put(
                {
                    "task_id": task_id,
                    "task": task,
                    "experiment": exp_name,
                    "seed": seed,
                    "success": False,
                    "error": str(exc),
                }
            )


def make_tasks(args):
    jobs = []
    for task in args.tasks:
        for exp_name in args.experiments:
            for seed in args.seeds:
                jobs.append((len(jobs), task, exp_name, EXPERIMENTS[exp_name], seed))
    return jobs


def save_raw(raw_results, output_dir):
    path = output_dir / "raw_results.json"
    with path.open("w", encoding="utf-8") as file:
        json.dump(raw_results, file, indent=2, ensure_ascii=False)


def summarize(raw_results):
    import numpy as np

    rows = []
    for task, task_results in raw_results.items():
        baseline = np.array(task_results.get("baseline", []), dtype=float)
        baseline_mean = baseline.mean() if len(baseline) else np.nan

        for exp_name, values in task_results.items():
            arr = np.array(values, dtype=float)
            mean = arr.mean() if len(arr) else np.nan
            std = arr.std(ddof=1) if len(arr) > 1 else 0.0 if len(arr) else np.nan
            delta = (mean - baseline_mean) / baseline_mean * 100 if baseline_mean else np.nan
            rows.append(
                {
                    "task": task,
                    "experiment": exp_name,
                    "n": len(arr),
                    "mean_mae": float(mean),
                    "std_mae": float(std),
                    "delta_pct": float(delta),
                    "raw_maes": values,
                }
            )
    return rows


def fmt(value, digits=4, suffix=""):
    import math

    return "NA" if math.isnan(value) else f"{value:.{digits}f}{suffix}"


def save_summary(rows, output_dir):
    csv_path = output_dir / "ablation_results.csv"
    fields = ["task", "experiment", "n", "mean_mae", "std_mae", "delta_pct", "raw_maes"]
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "raw_maes": str(row["raw_maes"])})

    md_path = output_dir / "ablation_report.md"
    lines = [
        "# FluoGPS 消融实验报告",
        "",
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "| 任务 | 实验 | n | MAE 均值 | MAE 标准差 | vs baseline |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['task']} | {row['experiment']} | {row['n']} | "
            f"{fmt(row['mean_mae'])} | {fmt(row['std_mae'])} | {fmt(row['delta_pct'], 2, '%')} |"
        )
    lines.extend(["", "## 原始 MAE", ""])
    for row in rows:
        raw = ", ".join(f"{value:.4f}" for value in row["raw_maes"]) if row["raw_maes"] else "无数据"
        lines.append(f"- {row['task']} / {row['experiment']}: {raw}")

    with md_path.open("w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")

    print(f"CSV: {csv_path}")
    print(f"Markdown: {md_path}")


def resolve_gpus(args):
    import torch

    if not torch.cuda.is_available():
        print("[错误] CUDA 不可用。")
        return None

    device_count = torch.cuda.device_count()
    gpus = list(range(device_count)) if args.gpus is None else args.gpus
    invalid = [gpu_id for gpu_id in gpus if gpu_id < 0 or gpu_id >= device_count]
    if invalid:
        print(f"[错误] GPU 不存在: {invalid}，可用范围: 0-{device_count - 1}")
        return None
    return gpus


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass

    gpus = resolve_gpus(args)
    if not gpus:
        return

    jobs = make_tasks(args)
    raw_results = {task: {exp: [] for exp in args.experiments} for task in args.tasks}
    failures = []

    print(f"任务: {args.tasks}")
    print(f"实验: {args.experiments}")
    print(f"种子: {args.seeds}")
    print(f"GPU: {gpus}")
    print(f"总训练次数: {len(jobs)}")

    task_queue = Queue()
    result_queue = Queue()
    for job in jobs:
        task_queue.put(job)
    for _ in gpus:
        task_queue.put(None)

    workers = [Process(target=worker, args=(gpu, task_queue, result_queue, args, output_dir)) for gpu in gpus]
    for process in workers:
        process.start()

    completed = 0
    while completed < len(jobs):
        try:
            result = result_queue.get(timeout=5)
        except Empty:
            continue

        completed += 1
        prefix = f"[{completed}/{len(jobs)}] {result['task']} / {result['experiment']} / seed={result['seed']}"
        if result["success"]:
            raw_results[result["task"]][result["experiment"]].append(result["test_mae"])
            print(f"{prefix}: MAE={result['test_mae']:.4f}")
        else:
            failures.append(result)
            print(f"{prefix}: 失败 - {result['error']}")

        save_raw(raw_results, output_dir)

    for process in workers:
        process.join()

    rows = summarize(raw_results)
    save_summary(rows, output_dir)

    if failures:
        fail_path = output_dir / "failed_runs.json"
        with fail_path.open("w", encoding="utf-8") as file:
            json.dump(failures, file, indent=2, ensure_ascii=False)
        print(f"失败记录: {fail_path}")
    else:
        fail_path = output_dir / "failed_runs.json"
        if fail_path.exists():
            fail_path.unlink()


if __name__ == "__main__":
    main()
