from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing as mp
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def load_cv_module():
    module_path = ROOT_DIR / "scripts" / "5_cross_validation.py"
    spec = importlib.util.spec_from_file_location("scaffold_cv", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CV = load_cv_module()

DEFAULT_LOG_DIR = ROOT_DIR / "outputs" / "runs" / "scaffold_nested_tuning"
FOLDS = [0, 1, 2, 3, 4]
DEFAULT_SPLIT_COLUMN = "family_scaffold_fold"
DEFAULT_SCAFFOLD_COLUMN = "tag_name"

BASE_MODEL_CONFIG = {
    "batch_size": 128,
    "num_layers": 10,
    "dim_hidden": 128,
    "num_heads": 4,
    "max_epoch": 300,
    "warm_steps": 15,
}

CANDIDATES = {
    "P0": {
        "description": "current scaffold CV defaults",
        "lr": 1e-3,
        "dropout": 0.05,
        "attn_dropout": 0.10,
        "weight_decay": 0.0,
    },
    "P1": {
        "description": "light regularization",
        "lr": 1e-3,
        "dropout": 0.10,
        "attn_dropout": 0.10,
        "weight_decay": 1e-5,
    },
    "P2": {
        "description": "stronger attention regularization",
        "lr": 1e-3,
        "dropout": 0.10,
        "attn_dropout": 0.20,
        "weight_decay": 1e-5,
    },
    "P3": {
        "description": "medium regularization",
        "lr": 1e-3,
        "dropout": 0.20,
        "attn_dropout": 0.20,
        "weight_decay": 1e-5,
    },
    "P4": {
        "description": "stronger weight decay",
        "lr": 1e-3,
        "dropout": 0.10,
        "attn_dropout": 0.20,
        "weight_decay": 1e-4,
    },
    "P5": {
        "description": "lower learning rate",
        "lr": 5e-4,
        "dropout": 0.10,
        "attn_dropout": 0.20,
        "weight_decay": 1e-5,
    },
    "P6": {
        "description": "lower learning rate with medium regularization",
        "lr": 5e-4,
        "dropout": 0.20,
        "attn_dropout": 0.20,
        "weight_decay": 1e-5,
    },
    "P7": {
        "description": "conservative learning rate",
        "lr": 3e-4,
        "dropout": 0.10,
        "attn_dropout": 0.20,
        "weight_decay": 1e-5,
    },
}

CANDIDATE_SETS = {
    "probe": ["P0", "P6"],
    "minimal": ["P0", "P2", "P5", "P6"],
    "full": ["P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7"],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Nested scaffold/group hyperparameter tuning for FluoGPS."
    )
    parser.add_argument(
        "--task",
        choices=["abs", "emi", "both"],
        default="both",
        help="Which prediction task to tune.",
    )
    parser.add_argument("--abs_csv", type=Path, default=CV.TASKS["abs"]["source_csv"])
    parser.add_argument("--emi_csv", type=Path, default=CV.TASKS["emi"]["source_csv"])
    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=FOLDS,
        help="All scaffold fold IDs available in the source CSVs.",
    )
    parser.add_argument(
        "--outer_folds",
        type=int,
        nargs="+",
        default=None,
        help="Outer test folds to run. Defaults to all --folds.",
    )
    parser.add_argument(
        "--inner_folds",
        type=int,
        nargs="+",
        default=None,
        help="Optional subset of folds allowed as inner validation folds.",
    )
    parser.add_argument(
        "--split_column",
        type=str,
        default=DEFAULT_SPLIT_COLUMN,
        help="CSV column that defines fold IDs. Defaults to family_scaffold_fold.",
    )
    parser.add_argument(
        "--scaffold_column",
        type=str,
        default=DEFAULT_SCAFFOLD_COLUMN,
        help="Column checked for group leakage across folds. Defaults to tag_name.",
    )
    parser.add_argument(
        "--candidate_set",
        choices=sorted(CANDIDATE_SETS),
        default="minimal",
        help="Named candidate set to use when --candidates is not set.",
    )
    parser.add_argument(
        "--candidates",
        nargs="+",
        default=None,
        help="Explicit candidate IDs, for example: --candidates P0 P2 P5.",
    )
    parser.add_argument("--batch_size", type=int, default=BASE_MODEL_CONFIG["batch_size"])
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--kernel", type=int, nargs="+", default=list(range(1, 17)))
    parser.add_argument("--num_layers", type=int, default=BASE_MODEL_CONFIG["num_layers"])
    parser.add_argument("--dim_out", type=int, default=1)
    parser.add_argument("--dim_hidden", type=int, default=BASE_MODEL_CONFIG["dim_hidden"])
    parser.add_argument("--num_heads", type=int, default=BASE_MODEL_CONFIG["num_heads"])
    parser.add_argument("--warm_steps", type=float, default=BASE_MODEL_CONFIG["warm_steps"])
    parser.add_argument("--max_epoch", type=int, default=BASE_MODEL_CONFIG["max_epoch"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of seeds to run for each split/candidate pair.",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        nargs="+",
        default=None,
        help="GPU IDs used to run jobs in parallel, for example: --gpus 0 1 2 3.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--raw_dir",
        type=Path,
        default=None,
        help="Directory for generated split CSVs. Defaults to <log_dir>/dataset/raw.",
    )
    parser.add_argument("--log_dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument(
        "--run_outer_eval",
        action="store_true",
        help="After inner tuning, evaluate selected candidates on each outer test fold.",
    )
    parser.add_argument(
        "--require_preprocessed",
        action="store_true",
        help="Fail before training if required PyG processed cache files are missing.",
    )
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Build PyG processed cache files for all planned splits before training.",
    )
    parser.add_argument(
        "--preprocess_only",
        action="store_true",
        help="Write split CSVs, build PyG processed caches, and skip training.",
    )
    parser.add_argument(
        "--force_preprocess",
        action="store_true",
        help="Delete existing processed cache files before preprocessing planned splits.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only write split CSVs and the run plan; do not train.",
    )
    return parser.parse_args()


def selected_tasks(args):
    task_configs = {task: dict(config) for task, config in CV.TASKS.items()}
    task_configs["abs"]["source_csv"] = args.abs_csv
    task_configs["emi"]["source_csv"] = args.emi_csv
    tasks = ["abs", "emi"] if args.task == "both" else [args.task]
    return {task: task_configs[task] for task in tasks}


def resolve_candidate_ids(args):
    candidate_ids = args.candidates if args.candidates is not None else CANDIDATE_SETS[args.candidate_set]
    unknown = sorted(set(candidate_ids) - set(CANDIDATES))
    if unknown:
        raise ValueError(f"Unknown candidate IDs: {unknown}")
    return candidate_ids


def candidate_config(candidate_id, args):
    config = {
        **BASE_MODEL_CONFIG,
        "batch_size": args.batch_size,
        "num_layers": args.num_layers,
        "dim_hidden": args.dim_hidden,
        "num_heads": args.num_heads,
        "max_epoch": args.max_epoch,
        "warm_steps": args.warm_steps,
        **CANDIDATES[candidate_id],
    }
    return config


def validate_folds(all_folds, outer_folds, inner_folds):
    all_fold_set = set(all_folds)
    invalid_outer = sorted(set(outer_folds) - all_fold_set)
    invalid_inner = sorted(set(inner_folds) - all_fold_set) if inner_folds else []
    if invalid_outer:
        raise ValueError(f"Outer folds are not in --folds: {invalid_outer}")
    if invalid_inner:
        raise ValueError(f"Inner folds are not in --folds: {invalid_inner}")


def safe_name(value):
    return "".join(char if char.isalnum() else "_" for char in str(value)).strip("_")


def load_fold_table(path, source_target, target, split_column, scaffold_column):
    df = pd.read_csv(path)
    required = {"smiles", "solvent", "tag_name", split_column, source_target}
    if scaffold_column:
        required.add(scaffold_column)
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")

    columns = ["smiles", "solvent", source_target, "tag_name", split_column]
    if scaffold_column and scaffold_column not in columns:
        columns.append(scaffold_column)
    work = df[columns].copy()
    work = work.rename(columns={source_target: target})
    work["fold"] = pd.to_numeric(work[split_column], errors="coerce")
    work[target] = pd.to_numeric(work[target], errors="coerce")
    work = work.dropna(subset=["smiles", target, "fold"])
    work["fold"] = work["fold"].astype(int)

    if scaffold_column:
        positive_fold_df = work[work["fold"] >= 0]
        scaffold_fold_counts = positive_fold_df.groupby(scaffold_column)["fold"].nunique()
        split_scaffolds = scaffold_fold_counts[scaffold_fold_counts > 1]
        if not split_scaffolds.empty:
            raise ValueError(
                f"These {scaffold_column} values appear in multiple folds: "
                + ", ".join(split_scaffolds.index.astype(str))
            )
    return work


def split_paths(task, source_csv, split_column, outer_fold, inner_valid_fold, raw_dir):
    fingerprint = CV.source_fingerprint(source_csv)
    prefix = f"{task}_nested_{safe_name(split_column)}_{fingerprint}_outer{outer_fold}_valid{inner_valid_fold}"
    return {
        "train": raw_dir / f"{prefix}_train.csv",
        "valid": raw_dir / f"{prefix}_valid.csv",
        "test": raw_dir / f"{prefix}_test.csv",
    }


def write_nested_split_csvs(df, task, source_csv, split_column, all_folds, outer_fold, inner_valid_fold, raw_dir):
    if outer_fold == inner_valid_fold:
        raise ValueError("outer_fold and inner_valid_fold must be different")

    train_folds = [fold for fold in all_folds if fold not in {outer_fold, inner_valid_fold}]
    if not train_folds:
        raise ValueError("No training folds left after choosing outer and inner validation folds")

    paths = split_paths(task, source_csv, split_column, outer_fold, inner_valid_fold, raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    eligible = df[df["fold"].isin(all_folds)].copy()

    columns = ["smiles", "solvent", task]
    train_df = eligible[eligible["fold"].isin(train_folds)][columns]
    valid_df = eligible[eligible["fold"] == inner_valid_fold][columns]
    test_df = eligible[eligible["fold"] == outer_fold][columns]

    if train_df.empty:
        raise ValueError(f"{task} outer {outer_fold} valid {inner_valid_fold}: training split is empty")
    if valid_df.empty:
        raise ValueError(f"{task} outer {outer_fold} valid {inner_valid_fold}: validation split is empty")
    if test_df.empty:
        raise ValueError(f"{task} outer {outer_fold} valid {inner_valid_fold}: test split is empty")

    train_df.to_csv(paths["train"], index=False)
    valid_df.to_csv(paths["valid"], index=False)
    test_df.to_csv(paths["test"], index=False)
    return paths, train_folds, train_df, valid_df, test_df


def summarize_tags(source_df, folds):
    return ";".join(sorted(source_df.loc[source_df["fold"].isin(folds), "tag_name"].dropna().astype(str).unique()))


def make_base_row(
    task,
    split_column,
    scaffold_column,
    outer_fold,
    inner_valid_fold,
    train_folds,
    train_df,
    valid_df,
    test_df,
    source_df,
    paths,
):
    return {
        "task": task,
        "split_column": split_column,
        "scaffold_column": scaffold_column,
        "outer_fold": outer_fold,
        "inner_valid_fold": inner_valid_fold,
        "train_folds": ";".join(str(fold) for fold in train_folds),
        "train_size": len(train_df),
        "valid_size": len(valid_df),
        "test_size": len(test_df),
        "train_tag_names": summarize_tags(source_df, train_folds),
        "valid_tag_names": summarize_tags(source_df, [inner_valid_fold]),
        "test_tag_names": summarize_tags(source_df, [outer_fold]),
        "train_csv": str(paths["train"]),
        "valid_csv": str(paths["valid"]),
        "test_csv": str(paths["test"]),
    }


def make_job(index, stage, base_row, candidate_id, candidate, args, repeat_index=0):
    output_dir = (
        args.log_dir
        / base_row["task"]
        / f"outer_{base_row['outer_fold']}"
        / f"candidate_{candidate_id}"
        / f"inner_valid_{base_row['inner_valid_fold']}"
        / f"seed_{args.seed + repeat_index}"
        / stage
    )
    test_csv = base_row["valid_csv"] if stage == "tune" else base_row["test_csv"]
    return {
        "index": index,
        "stage": stage,
        "task": base_row["task"],
        "outer_fold": base_row["outer_fold"],
        "inner_valid_fold": base_row["inner_valid_fold"],
        "candidate": candidate_id,
        "candidate_description": candidate["description"],
        "repeat_index": repeat_index,
        "train_csv": base_row["train_csv"],
        "valid_csv": base_row["valid_csv"],
        "test_csv": test_csv,
        "output_dir": str(output_dir),
        "seed": args.seed + repeat_index,
        "raw_dir": str(args.raw_dir),
        "device": args.device,
        "num_workers": args.num_workers,
        "kernel": args.kernel,
        "dim_out": args.dim_out,
        **{key: candidate[key] for key in [
            "batch_size",
            "num_layers",
            "dim_hidden",
            "num_heads",
            "dropout",
            "attn_dropout",
            "lr",
            "warm_steps",
            "weight_decay",
            "max_epoch",
        ]},
    }


def require_job_cache(job):
    CV.require_preprocessed_cache(
        raw_dir=Path(job["raw_dir"]),
        csv_paths=[job["train_csv"], job["valid_csv"], job["test_csv"]],
        kernel=job["kernel"],
    )


def delete_processed_cache(root, csv_path, kernel):
    for cache_path in CV.processed_paths(root, csv_path, kernel):
        if cache_path.exists():
            cache_path.unlink()
            print(f"Deleted cache: {cache_path}")


def preprocess_nested_caches(split_rows, args):
    from data import FluorescenceDataset, compute_normalization_params
    from data.loader import create_rwse_transform, normalize_kernel

    root = args.raw_dir.parent
    kernel = normalize_kernel(args.kernel)
    pre_transform = create_rwse_transform(kernel=kernel)

    print(f"Preprocessing nested PyG caches under: {root / 'processed'}")
    for row in split_rows:
        y_mean, y_std = compute_normalization_params(row["train_csv"])
        csv_paths = list(dict.fromkeys([row["train_csv"], row["valid_csv"], row["test_csv"]]))
        print(
            f"[preprocess] {row['task']} outer={row['outer_fold']} "
            f"valid={row['inner_valid_fold']} files={len(csv_paths)}"
        )
        for csv_path in csv_paths:
            if args.force_preprocess:
                delete_processed_cache(root, csv_path, kernel)
            FluorescenceDataset(
                csv_file=csv_path,
                root=str(root),
                kernel=kernel,
                y_mean=y_mean,
                y_std=y_std,
                pre_transform=pre_transform,
            )


def set_seed(seed):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name):
    import torch

    if str(device_name).startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def run_training_job(job):
    if job.get("gpu") is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(job["gpu"])
        job = dict(job)
        job["device"] = "cuda"

    from models.network import GPSModel
    from train.trainer import create_loader, custom_train
    from utils import schedule_with_warmup
    from utils.simple_logger import create_simple_logger
    import torch

    set_seed(job["seed"])
    device = resolve_device(job["device"])
    output_dir = Path(job["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[{job['stage']}] {job['task']} outer={job['outer_fold']} "
        f"valid={job['inner_valid_fold']} candidate={job['candidate']} "
        f"seed={job['seed']}",
        flush=True,
    )

    train_loader, val_loader, test_loader = create_loader(
        train_dataset_dir=job["train_csv"],
        val_dataset_dir=job["valid_csv"],
        test_dataset_dir=job["test_csv"],
        batch_size=job["batch_size"],
        num_workers=job["num_workers"],
        kernel=job["kernel"],
        dataset_root=Path(job["raw_dir"]).parent,
    )

    model = GPSModel(
        dim_h=job["dim_hidden"],
        num_heads=job["num_heads"],
        dropout=job["dropout"],
        attn_dropout=job["attn_dropout"],
        num_layers=job["num_layers"],
        dim_out=job["dim_out"],
        rwse_steps=job["kernel"],
    ).to(device)
    loggers = create_simple_logger(output_dir=str(output_dir))
    optimizer = torch.optim.AdamW(model.parameters(), lr=job["lr"], weight_decay=job["weight_decay"])
    scheduler = schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=job["warm_steps"],
        num_training_steps=job["max_epoch"],
    )

    try:
        result = custom_train(
            model=model,
            loaders=[train_loader, val_loader, test_loader],
            loggers=loggers,
            optimizer=optimizer,
            scheduler=scheduler,
            max_epoch=job["max_epoch"],
        )
    finally:
        for logger in loggers:
            logger.close()

    return {
        "index": job["index"],
        "stage": job["stage"],
        "task": job["task"],
        "outer_fold": job["outer_fold"],
        "inner_valid_fold": job["inner_valid_fold"],
        "candidate": job["candidate"],
        "candidate_description": job["candidate_description"],
        "repeat_index": job["repeat_index"],
        "seed": job["seed"],
        "output_dir": job["output_dir"],
        "train_csv": job["train_csv"],
        "valid_csv": job["valid_csv"],
        "test_csv": job["test_csv"],
        "batch_size": job["batch_size"],
        "num_layers": job["num_layers"],
        "dim_hidden": job["dim_hidden"],
        "num_heads": job["num_heads"],
        "dropout": job["dropout"],
        "attn_dropout": job["attn_dropout"],
        "lr": job["lr"],
        "warm_steps": job["warm_steps"],
        "weight_decay": job["weight_decay"],
        "max_epoch": job["max_epoch"],
        **result,
    }


def shard_jobs_by_gpu(jobs, gpus, group_same_split):
    shards = {gpu_id: [] for gpu_id in gpus}
    if not group_same_split:
        for index, job in enumerate(jobs):
            gpu_id = gpus[index % len(gpus)]
            shards[gpu_id].append({**job, "gpu": gpu_id})
        return shards

    split_to_gpu = {}
    for job in jobs:
        split_key = (
            job["stage"],
            job["task"],
            job["outer_fold"],
            job["inner_valid_fold"],
        )
        if split_key not in split_to_gpu:
            split_to_gpu[split_key] = gpus[len(split_to_gpu) % len(gpus)]
        gpu_id = split_to_gpu[split_key]
        shards[gpu_id].append({**job, "gpu": gpu_id})
    return shards


def run_gpu_worker(gpu_id, jobs):
    completed = []
    for job in jobs:
        print(
            f"[gpu {gpu_id}] assigned {job['stage']} {job['task']} "
            f"outer={job['outer_fold']} valid={job['inner_valid_fold']} "
            f"candidate={job['candidate']}",
            flush=True,
        )
        completed.append(run_training_job(job))
    return completed


def run_jobs(jobs, gpus=None, group_same_split=True):
    if not jobs:
        return []

    if not gpus:
        return [run_training_job(job) for job in jobs]

    CV.validate_gpus(gpus)
    shards = shard_jobs_by_gpu(jobs, gpus, group_same_split=group_same_split)
    non_empty_shards = {gpu_id: gpu_jobs for gpu_id, gpu_jobs in shards.items() if gpu_jobs}
    for gpu_id, gpu_jobs in non_empty_shards.items():
        names = ", ".join(
            f"{job['stage']}:{job['task']}:o{job['outer_fold']}:v{job['inner_valid_fold']}:{job['candidate']}"
            for job in gpu_jobs
        )
        print(f"[gpu {gpu_id}] job queue: {names}")

    completed = []
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=len(non_empty_shards), mp_context=context) as executor:
        futures = [
            executor.submit(run_gpu_worker, gpu_id, gpu_jobs)
            for gpu_id, gpu_jobs in non_empty_shards.items()
        ]
        for future in as_completed(futures):
            completed.extend(future.result())

    return sorted(completed, key=lambda item: item["index"])


def candidate_summary(inner_results):
    if inner_results.empty:
        return inner_results

    summary = (
        inner_results.groupby(["task", "outer_fold", "candidate"], as_index=False)
        .agg(
            mean_inner_val_mae=("best_val_mae", "mean"),
            std_inner_val_mae=("best_val_mae", "std"),
            worst_inner_val_mae=("best_val_mae", "max"),
            mean_best_epoch=("best_epoch", "mean"),
            n_inner_runs=("best_val_mae", "count"),
        )
        .fillna({"std_inner_val_mae": 0.0})
    )
    return summary.sort_values(
        ["task", "outer_fold", "mean_inner_val_mae", "worst_inner_val_mae", "std_inner_val_mae", "candidate"]
    )


def overall_candidate_summary(inner_results):
    if inner_results.empty:
        return inner_results

    summary = (
        inner_results.groupby(["task", "candidate"], as_index=False)
        .agg(
            mean_inner_val_mae=("best_val_mae", "mean"),
            std_inner_val_mae=("best_val_mae", "std"),
            worst_inner_val_mae=("best_val_mae", "max"),
            mean_best_epoch=("best_epoch", "mean"),
            n_inner_runs=("best_val_mae", "count"),
        )
        .fillna({"std_inner_val_mae": 0.0})
    )
    return summary.sort_values(
        ["task", "mean_inner_val_mae", "worst_inner_val_mae", "std_inner_val_mae", "candidate"]
    )


def select_candidates(summary):
    selected = []
    for _, group in summary.groupby(["task", "outer_fold"], sort=True):
        selected.append(group.iloc[0].to_dict())
    return pd.DataFrame(selected)


def outer_eval_summary(eval_results):
    if eval_results.empty:
        return eval_results

    summary = (
        eval_results.groupby(["task", "outer_fold", "candidate"], as_index=False)
        .agg(
            mean_outer_test_mae=("final_test_mae", "mean"),
            std_outer_test_mae=("final_test_mae", "std"),
            worst_outer_test_mae=("final_test_mae", "max"),
            mean_eval_val_mae=("best_val_mae", "mean"),
            n_eval_runs=("final_test_mae", "count"),
        )
        .fillna({"std_outer_test_mae": 0.0})
    )
    return summary.sort_values(["task", "outer_fold"])


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main():
    args = parse_args()
    args.log_dir = args.log_dir.resolve()
    args.raw_dir = (
        args.raw_dir.resolve()
        if args.raw_dir is not None
        else (args.log_dir / "dataset" / "raw").resolve()
    )
    outer_folds = args.outer_folds if args.outer_folds is not None else args.folds
    validate_folds(args.folds, outer_folds, args.inner_folds)
    if args.repeat < 1:
        raise ValueError("--repeat must be at least 1")
    if args.preprocess_only:
        args.preprocess = True
    if args.dry_run and args.preprocess_only:
        raise ValueError("--dry_run and --preprocess_only cannot be used together")
    if args.force_preprocess and not args.preprocess:
        raise ValueError("--force_preprocess requires --preprocess or --preprocess_only")
    candidate_ids = resolve_candidate_ids(args)
    candidates = {candidate_id: candidate_config(candidate_id, args) for candidate_id in candidate_ids}

    args.log_dir.mkdir(parents=True, exist_ok=True)
    split_rows = []
    tune_jobs = []
    split_by_key = {}

    for task, config in selected_tasks(args).items():
        source_csv = Path(config["source_csv"]).resolve()
        source_df = load_fold_table(
            source_csv,
            config["source_target"],
            config["target"],
            split_column=args.split_column,
            scaffold_column=args.scaffold_column,
        )
        target = config["target"]

        for outer_fold in outer_folds:
            inner_valid_folds = [fold for fold in args.folds if fold != outer_fold]
            if args.inner_folds is not None:
                inner_valid_folds = [fold for fold in inner_valid_folds if fold in args.inner_folds]
            if not inner_valid_folds:
                raise ValueError(f"No inner validation folds left for outer fold {outer_fold}")

            for inner_valid_fold in inner_valid_folds:
                paths, train_folds, train_df, valid_df, test_df = write_nested_split_csvs(
                    source_df,
                    task=target,
                    source_csv=source_csv,
                    split_column=args.split_column,
                    all_folds=args.folds,
                    outer_fold=outer_fold,
                    inner_valid_fold=inner_valid_fold,
                    raw_dir=args.raw_dir,
                )
                base_row = make_base_row(
                    target,
                    args.split_column,
                    args.scaffold_column,
                    outer_fold,
                    inner_valid_fold,
                    train_folds,
                    train_df,
                    valid_df,
                    test_df,
                    source_df,
                    paths,
                )
                split_rows.append(base_row)
                split_by_key[(target, outer_fold, inner_valid_fold)] = base_row

                for candidate_id, candidate in candidates.items():
                    for repeat_index in range(args.repeat):
                        tune_jobs.append(
                            make_job(
                                len(tune_jobs),
                                "tune",
                                base_row,
                                candidate_id,
                                candidate,
                                args,
                                repeat_index=repeat_index,
                            )
                        )

    split_path = args.log_dir / "nested_split_summary.csv"
    pd.DataFrame(split_rows).to_csv(split_path, index=False)

    plan_path = args.log_dir / "nested_tuning_plan.csv"
    pd.DataFrame(tune_jobs).to_csv(plan_path, index=False)

    metadata_path = args.log_dir / "nested_tuning_metadata.json"
    write_json(
        metadata_path,
        {
            "task": args.task,
            "split_column": args.split_column,
            "scaffold_column": args.scaffold_column,
            "folds": args.folds,
            "outer_folds": outer_folds,
            "inner_folds": args.inner_folds,
            "candidate_ids": candidate_ids,
            "candidates": candidates,
            "repeat": args.repeat,
            "preprocess": args.preprocess,
            "preprocess_only": args.preprocess_only,
            "force_preprocess": args.force_preprocess,
            "run_outer_eval": args.run_outer_eval,
            "note": "Inner tuning uses only inner validation MAE for candidate selection. Outer test CSVs are used only when --run_outer_eval is set.",
        },
    )

    print(f"Split summary saved to: {split_path}")
    print(f"Tuning plan saved to: {plan_path}")
    print(f"Metadata saved to: {metadata_path}")
    print(f"Tuning jobs: {len(tune_jobs)}")

    if args.dry_run:
        print("Dry run requested; training was skipped.")
        return

    if args.preprocess:
        preprocess_nested_caches(split_rows, args)
        if args.preprocess_only:
            print("Preprocess-only requested; training was skipped.")
            return

    if args.require_preprocessed:
        for job in tune_jobs:
            require_job_cache(job)

    group_same_split = not (args.preprocess or args.require_preprocessed)
    inner_results = pd.DataFrame(run_jobs(tune_jobs, args.gpus, group_same_split=group_same_split))
    inner_results_path = args.log_dir / "inner_tuning_results.csv"
    inner_results.to_csv(inner_results_path, index=False)
    print(f"Inner tuning results saved to: {inner_results_path}")

    by_outer = candidate_summary(inner_results)
    by_outer_path = args.log_dir / "candidate_summary_by_outer.csv"
    by_outer.to_csv(by_outer_path, index=False)
    print(f"Candidate summary by outer fold saved to: {by_outer_path}")

    overall = overall_candidate_summary(inner_results)
    overall_path = args.log_dir / "candidate_summary_overall.csv"
    overall.to_csv(overall_path, index=False)
    print(f"Overall candidate summary saved to: {overall_path}")

    selected = select_candidates(by_outer)
    selected_path = args.log_dir / "selected_candidates_by_outer.csv"
    selected.to_csv(selected_path, index=False)
    print(f"Selected candidates saved to: {selected_path}")

    if not args.run_outer_eval:
        print("Outer evaluation was skipped. Add --run_outer_eval to evaluate selected candidates on held-out outer folds.")
        return

    eval_jobs = []
    for _, selected_row in selected.iterrows():
        task = selected_row["task"]
        outer_fold = int(selected_row["outer_fold"])
        candidate_id = selected_row["candidate"]
        candidate = candidates[candidate_id]
        matching_splits = [
            row for key, row in split_by_key.items()
            if key[0] == task and key[1] == outer_fold
        ]
        for base_row in matching_splits:
            for repeat_index in range(args.repeat):
                eval_jobs.append(
                    make_job(
                        len(eval_jobs),
                        "outer_eval",
                        base_row,
                        candidate_id,
                        candidate,
                        args,
                        repeat_index=repeat_index,
                    )
                )

    eval_plan_path = args.log_dir / "outer_eval_plan.csv"
    pd.DataFrame(eval_jobs).to_csv(eval_plan_path, index=False)
    print(f"Outer eval plan saved to: {eval_plan_path}")

    if args.require_preprocessed:
        for job in eval_jobs:
            require_job_cache(job)

    eval_results = pd.DataFrame(run_jobs(eval_jobs, args.gpus, group_same_split=group_same_split))
    eval_results_path = args.log_dir / "outer_eval_results.csv"
    eval_results.to_csv(eval_results_path, index=False)
    print(f"Outer eval results saved to: {eval_results_path}")

    eval_summary = outer_eval_summary(eval_results)
    eval_summary_path = args.log_dir / "outer_eval_summary.csv"
    eval_summary.to_csv(eval_summary_path, index=False)
    print(f"Outer eval summary saved to: {eval_summary_path}")


if __name__ == "__main__":
    main()
