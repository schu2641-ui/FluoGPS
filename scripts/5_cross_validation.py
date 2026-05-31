from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
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

DEFAULT_RAW_DIR = ROOT_DIR / "datasets" / "fluorescence" / "raw"
DEFAULT_LOG_DIR = ROOT_DIR / "outputs" / "runs" / "dual_fluogps_scaffold_cv"
MODEL_NAME = "Dual-FluoGPS"
USE_DUAL_GRAPH = False
DEFAULT_DUAL_WEIGHT_MODE = "shared"
TASKS = {
    "abs": {
        "source_csv": ROOT_DIR / "FluoDB-Lite_abs.csv",
        "source_target": "absorption/nm",
        "target": "abs",
    },
    "emi": {
        "source_csv": ROOT_DIR / "FluoDB-Lite_emi.csv",
        "source_target": "emission/nm",
        "target": "emi",
    },
}
SPLIT_SCHEMES = {
    "family": {
        "name": "family_scaffold",
        "split_column": "family_scaffold_fold",
        "scaffold_column": "tag_name",
    },
    "murcko": {
        "name": "murcko_scaffold",
        "split_column": "Murcko_scaffold_fold",
        "scaffold_column": "Murcko_scaffold",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=f"Run scaffold 5-fold cross validation for {MODEL_NAME}."
    )
    parser.add_argument(
        "--task",
        choices=["abs", "emi", "both"],
        default="both",
        help="Which prediction task to run.",
    )
    parser.add_argument("--abs_csv", type=Path, default=TASKS["abs"]["source_csv"])
    parser.add_argument("--emi_csv", type=Path, default=TASKS["emi"]["source_csv"])
    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="Held-out fold IDs to run.",
    )
    parser.add_argument(
        "--split_scheme",
        choices=["family", "murcko", "both"],
        default="murcko",
        help="Scaffold split to run. Use 'both' to run family and Murcko folds.",
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--kernel", type=int, nargs="+", default=list(range(1, 17)))
    parser.add_argument(
        "--dual_graph",
        action="store_true",
        default=USE_DUAL_GRAPH,
        help="Use separate solute and solvent graphs.",
    )
    parser.add_argument(
        "--dual_weight_mode",
        choices=["shared", "separate"],
        default=DEFAULT_DUAL_WEIGHT_MODE,
        help="Weight sharing mode for --dual_graph: shared reuses one GPS stack, separate uses one stack per branch.",
    )
    parser.add_argument("--num_layers", type=int, default=10)
    parser.add_argument("--dim_out", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--attn_dropout", type=float, default=0.1)
    parser.add_argument("--dim_hidden", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--warm_steps", type=float, default=15)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--max_epoch", type=int, default=300)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--gpus",
        type=int,
        nargs="+",
        default=None,
        help="GPU IDs used to run folds in parallel, for example: --gpus 0 1 2 3.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--raw_dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--log_dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only write and summarize fold CSVs; do not train.",
    )
    parser.add_argument(
        "--require_preprocessed",
        action="store_true",
        help="Fail before training if scaffold CV processed cache files are missing.",
    )
    return parser.parse_args()


def args_to_worker_dict(args):
    values = vars(args).copy()
    values["abs_csv"] = str(values["abs_csv"])
    values["emi_csv"] = str(values["emi_csv"])
    values["raw_dir"] = str(values["raw_dir"])
    values["log_dir"] = str(values["log_dir"])
    return values


def worker_dict_to_args(values):
    values = dict(values)
    values["abs_csv"] = Path(values["abs_csv"])
    values["emi_csv"] = Path(values["emi_csv"])
    values["raw_dir"] = Path(values["raw_dir"])
    values["log_dir"] = Path(values["log_dir"])
    return argparse.Namespace(**values)


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


def validate_gpus(gpus):
    if not gpus:
        return

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("--gpus was set, but CUDA is not available in this environment.")

    device_count = torch.cuda.device_count()
    invalid = [gpu_id for gpu_id in gpus if gpu_id < 0 or gpu_id >= device_count]
    if invalid:
        raise ValueError(
            f"Invalid GPU IDs {invalid}; available GPU IDs are 0-{device_count - 1}."
        )


def selected_tasks(args):
    task_configs = {task: dict(config) for task, config in TASKS.items()}
    task_configs["abs"]["source_csv"] = args.abs_csv
    task_configs["emi"]["source_csv"] = args.emi_csv

    tasks = ["abs", "emi"] if args.task == "both" else [args.task]
    return {task: task_configs[task] for task in tasks}


def selected_split_schemes(args):
    if args.split_scheme == "both":
        keys = ["family", "murcko"]
    else:
        keys = [args.split_scheme]
    return {key: SPLIT_SCHEMES[key] for key in keys}


def source_fingerprint(path):
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:10]


def load_scaffold_table(path, source_target, target, split_column, scaffold_column):
    df = pd.read_csv(path)
    required = {"smiles", "solvent", "tag_name", scaffold_column, split_column, source_target}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")

    columns = list(
        dict.fromkeys(["smiles", "solvent", source_target, "tag_name", scaffold_column, split_column])
    )
    work = df[columns].copy()
    work = work.rename(columns={source_target: target})
    work["fold"] = pd.to_numeric(work[split_column], errors="coerce")
    work[target] = pd.to_numeric(work[target], errors="coerce")
    work = work.dropna(subset=["smiles", target, "fold"])
    work["fold"] = work["fold"].astype(int)

    positive_fold_df = work[work["fold"] >= 0]
    scaffold_fold_counts = positive_fold_df.groupby(scaffold_column)["fold"].nunique()
    split_scaffolds = scaffold_fold_counts[scaffold_fold_counts > 1]
    if not split_scaffolds.empty:
        raise ValueError(
            f"These {scaffold_column} values appear in multiple folds: "
            + ", ".join(split_scaffolds.index.astype(str))
        )
    return work


def write_fold_csvs(df, task, source_csv, fold, raw_dir, split_scheme):
    raw_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{task}_{split_scheme}_cv_{source_fingerprint(source_csv)}_fold{fold}"
    train_path = raw_dir / f"{prefix}_train.csv"
    valid_path = raw_dir / f"{prefix}_valid.csv"

    eligible = df[df["fold"] >= 0].copy()
    train_df = eligible[eligible["fold"] != fold][["smiles", "solvent", task]]
    valid_df = eligible[eligible["fold"] == fold][["smiles", "solvent", task]]

    if train_df.empty:
        raise ValueError(f"{task} fold {fold}: training split is empty")
    if valid_df.empty:
        raise ValueError(f"{task} fold {fold}: validation split is empty")

    train_df.to_csv(train_path, index=False)
    valid_df.to_csv(valid_path, index=False)
    return train_path, valid_path, train_df, valid_df


def processed_paths(root, csv_path, kernel, use_solvent=True, dual_graph=False):
    base_name = Path(csv_path).stem
    kernel_suffix = f"_k{len(kernel)}" if kernel is not None else ""
    solvent_suffix = "_dual" if dual_graph else "" if use_solvent else "_nosolvent"
    processed_dir = Path(root) / "processed"
    return [
        processed_dir / f"{base_name}{kernel_suffix}{solvent_suffix}_processed.pt",
        processed_dir / f"{base_name}{kernel_suffix}{solvent_suffix}_norm_params.pt",
    ]


def require_preprocessed_cache(raw_dir, csv_paths, kernel, dual_graph=False):
    root = Path(raw_dir).parent
    missing = []
    for csv_path in csv_paths:
        for cache_path in processed_paths(root, csv_path, kernel, dual_graph=dual_graph):
            if not cache_path.exists():
                missing.append(cache_path)
    if missing:
        missing_preview = "\n".join(str(path) for path in missing[:8])
        extra = "" if len(missing) <= 8 else f"\n... and {len(missing) - 8} more"
        dual_hint = " --dual_graph" if dual_graph else ""
        raise FileNotFoundError(
            "Missing processed scaffold CV cache files. Run "
            f"`python scripts/preprocess_scaffold_cv_data.py --task both --split_scheme <family|murcko|both>{dual_hint}` first.\n"
            f"{missing_preview}{extra}"
        )


def summarize_split(task, fold, train_df, valid_df, source_df, split_scheme, split_column, scaffold_column):
    heldout_tags = sorted(
        source_df.loc[source_df["fold"] == fold, "tag_name"].dropna().astype(str).unique()
    )
    train_tags = sorted(
        source_df.loc[
            (source_df["fold"] >= 0) & (source_df["fold"] != fold),
            "tag_name",
        ].dropna().astype(str).unique()
    )
    heldout_scaffolds = source_df.loc[
        source_df["fold"] == fold, scaffold_column
    ].dropna().astype(str).unique()
    train_scaffolds = source_df.loc[
        (source_df["fold"] >= 0) & (source_df["fold"] != fold),
        scaffold_column,
    ].dropna().astype(str).unique()
    return {
        "task": task,
        "split_scheme": split_scheme,
        "split_column": split_column,
        "scaffold_column": scaffold_column,
        "fold": fold,
        "train_size": len(train_df),
        "valid_size": len(valid_df),
        "train_tag_count": len(train_tags),
        "valid_tag_count": len(heldout_tags),
        "train_scaffold_count": len(train_scaffolds),
        "valid_scaffold_count": len(heldout_scaffolds),
        "valid_tag_names": ";".join(heldout_tags),
    }


def run_fold(args, task, fold, train_csv, valid_csv, split_scheme):
    from data import DualFluorescenceDataset
    from models.network import DualGraphGPSModel, GPSModel
    from train.trainer import create_loader, custom_train
    from utils import schedule_with_warmup
    from utils.simple_logger import create_simple_logger
    import torch

    run_seed = args.seed + fold
    set_seed(run_seed)
    device = resolve_device(args.device)
    fold_log_dir = args.log_dir / split_scheme / task / f"fold_{fold}"
    fold_log_dir.mkdir(parents=True, exist_ok=True)
    model_cls = DualGraphGPSModel if args.dual_graph else GPSModel
    dataset_cls = DualFluorescenceDataset if args.dual_graph else None
    follow_batch = ["solute_x", "solvent_x"] if args.dual_graph else None

    train_loader, val_loader, test_loader = create_loader(
        train_dataset_dir=str(train_csv),
        val_dataset_dir=str(valid_csv),
        test_dataset_dir=str(valid_csv),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        kernel=args.kernel,
        dataset_root=args.raw_dir.parent,
        **({"dataset_cls": dataset_cls} if dataset_cls is not None else {}),
        follow_batch=follow_batch,
    )

    model = model_cls(
        dim_h=args.dim_hidden,
        num_heads=args.num_heads,
        dropout=args.dropout,
        attn_dropout=args.attn_dropout,
        num_layers=args.num_layers,
        dim_out=args.dim_out,
        rwse_steps=args.kernel,
        **({"dual_weight_mode": args.dual_weight_mode} if args.dual_graph else {}),
    ).to(device)
    loggers = create_simple_logger(output_dir=str(fold_log_dir))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.warm_steps,
        num_training_steps=args.max_epoch,
    )

    try:
        return custom_train(
            model=model,
            loaders=[train_loader, val_loader, test_loader],
            loggers=loggers,
            optimizer=optimizer,
            scheduler=scheduler,
            max_epoch=args.max_epoch,
        )
    finally:
        for logger in loggers:
            logger.close()


def shard_jobs_by_gpu(jobs, gpus):
    shards = {gpu_id: [] for gpu_id in gpus}
    for index, job in enumerate(jobs):
        gpu_id = gpus[index % len(gpus)]
        shards[gpu_id].append(job)
    return shards


def run_gpu_worker(gpu_id, jobs):
    completed = []
    for job in jobs:
        args = worker_dict_to_args(job["args"])
        args.device = f"cuda:{gpu_id}"
        print(
            f"[gpu {gpu_id}] starting {job['task']} fold {job['fold']} "
            f"train={job['train_csv']} valid={job['valid_csv']}",
            flush=True,
        )
        result = run_fold(
            args,
            task=job["task"],
            fold=job["fold"],
            train_csv=Path(job["train_csv"]),
            valid_csv=Path(job["valid_csv"]),
            split_scheme=job["split_scheme"],
        )
        completed.append(
            {
                "index": job["index"],
                "gpu": gpu_id,
                "result": result,
            }
        )
        print(
            f"[gpu {gpu_id}] finished {job['task']} fold {job['fold']}",
            flush=True,
        )
    return completed


def run_parallel_jobs(jobs, gpus):
    if not jobs:
        return {}

    validate_gpus(gpus)
    shards = shard_jobs_by_gpu(jobs, gpus)
    non_empty_shards = {
        gpu_id: gpu_jobs for gpu_id, gpu_jobs in shards.items() if gpu_jobs
    }
    for gpu_id, gpu_jobs in non_empty_shards.items():
        job_names = ", ".join(f"{job['task']}:fold{job['fold']}" for job in gpu_jobs)
        print(f"[gpu {gpu_id}] assigned jobs: {job_names}")

    results_by_index = {}
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=len(non_empty_shards),
        mp_context=context,
    ) as executor:
        future_to_gpu = {
            executor.submit(run_gpu_worker, gpu_id, gpu_jobs): gpu_id
            for gpu_id, gpu_jobs in non_empty_shards.items()
        }
        for future in as_completed(future_to_gpu):
            for item in future.result():
                results_by_index[item["index"]] = {
                    **item["result"],
                    "gpu": item["gpu"],
                }
    return results_by_index


def main():
    args = parse_args()
    args.raw_dir = args.raw_dir.resolve()
    args.log_dir = args.log_dir.resolve()

    rows = []
    jobs = []
    worker_args = args_to_worker_dict(args)
    split_schemes = selected_split_schemes(args)
    for split_key, split_config in split_schemes.items():
        split_name = split_config["name"]
        split_column = split_config["split_column"]
        scaffold_column = split_config["scaffold_column"]

        for task, config in selected_tasks(args).items():
            source_csv = Path(config["source_csv"]).resolve()
            source_target = config["source_target"]
            target = config["target"]
            source_df = load_scaffold_table(
                source_csv,
                source_target,
                target,
                split_column=split_column,
                scaffold_column=scaffold_column,
            )

            for fold in args.folds:
                train_csv, valid_csv, train_df, valid_df = write_fold_csvs(
                    source_df,
                    task=target,
                    source_csv=source_csv,
                    fold=fold,
                    raw_dir=args.raw_dir,
                    split_scheme=split_name,
                )
                summary = summarize_split(
                    target,
                    fold,
                    train_df,
                    valid_df,
                    source_df,
                    split_scheme=split_name,
                    split_column=split_column,
                    scaffold_column=scaffold_column,
                )
                summary["train_csv"] = str(train_csv)
                summary["valid_csv"] = str(valid_csv)
                summary["dual_graph"] = args.dual_graph
                summary["dual_weight_mode"] = args.dual_weight_mode if args.dual_graph else ""
                print(
                    f"[{split_name} {target} fold {fold}] "
                    f"train={summary['train_size']} valid={summary['valid_size']} "
                    f"valid_tags={summary['valid_tag_names']}"
                )

                if not args.dry_run:
                    if args.require_preprocessed:
                        require_preprocessed_cache(
                            args.raw_dir,
                            [train_csv, valid_csv],
                            args.kernel,
                            dual_graph=args.dual_graph,
                        )
                    if args.gpus:
                        jobs.append(
                            {
                                "index": len(rows),
                                "args": worker_args,
                                "task": target,
                                "fold": fold,
                                "split_scheme": split_name,
                                "train_csv": str(train_csv),
                                "valid_csv": str(valid_csv),
                            }
                        )
                    else:
                        result = run_fold(args, target, fold, train_csv, valid_csv, split_name)
                        summary.update(result)
                rows.append(summary)

    if jobs:
        results_by_index = run_parallel_jobs(jobs, args.gpus)
        for index, result in results_by_index.items():
            rows[index].update(result)

    args.log_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.log_dir / "scaffold_cv_summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    metadata_path = args.log_dir / "scaffold_cv_metadata.json"
    metadata = {
        "model_name": MODEL_NAME,
        "dual_graph": args.dual_graph,
        "dual_weight_mode": args.dual_weight_mode if args.dual_graph else None,
        "split_scheme": args.split_scheme,
        "split_schemes": {
            key: {
                "name": value["name"],
                "split_column": value["split_column"],
                "scaffold_column": value["scaffold_column"],
            }
            for key, value in split_schemes.items()
        },
        "task": args.task,
        "folds": args.folds,
        "gpus": args.gpus,
        "note": "test_csv is the same held-out fold as val_csv for scaffold cross validation.",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\nSummary saved to: {summary_path}")
    print(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()
