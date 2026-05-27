from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

from data import FluorescenceDataset, compute_normalization_params
from data.loader import create_rwse_transform, normalize_kernel


def load_cv_module():
    module_path = ROOT_DIR / "scripts" / "5_cross_validation.py"
    spec = importlib.util.spec_from_file_location("scaffold_cv", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CV = load_cv_module()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess Murcko scaffold CV CSVs into PyG processed caches."
    )
    parser.add_argument(
        "--task",
        choices=["abs", "emi", "both"],
        default="both",
        help="Which prediction task to preprocess.",
    )
    parser.add_argument("--abs_csv", type=Path, default=CV.TASKS["abs"]["source_csv"])
    parser.add_argument("--emi_csv", type=Path, default=CV.TASKS["emi"]["source_csv"])
    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="Held-out fold IDs to preprocess.",
    )
    parser.add_argument("--kernel", type=int, nargs="+", default=list(range(1, 17)))
    parser.add_argument("--raw_dir", type=Path, default=CV.DEFAULT_RAW_DIR)
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="FluorescenceDataset root. Defaults to raw_dir parent.",
    )
    parser.add_argument(
        "--no_solvent",
        action="store_true",
        help="Preprocess caches without solvent graphs.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete existing processed cache files before preprocessing.",
    )
    parser.add_argument(
        "--status_only",
        action="store_true",
        help="Only report cache status; do not preprocess datasets.",
    )
    return parser.parse_args()


def selected_tasks(args):
    task_configs = {task: dict(config) for task, config in CV.TASKS.items()}
    task_configs["abs"]["source_csv"] = args.abs_csv
    task_configs["emi"]["source_csv"] = args.emi_csv
    tasks = ["abs", "emi"] if args.task == "both" else [args.task]
    return {task: task_configs[task] for task in tasks}


def processed_paths(root, csv_path, kernel, use_solvent):
    base_name = Path(csv_path).stem
    kernel_suffix = f"_k{len(kernel)}" if kernel is not None else ""
    solvent_suffix = "" if use_solvent else "_nosolvent"
    processed_dir = Path(root) / "processed"
    return [
        processed_dir / f"{base_name}{kernel_suffix}{solvent_suffix}_processed.pt",
        processed_dir / f"{base_name}{kernel_suffix}{solvent_suffix}_norm_params.pt",
    ]


def cache_ready(root, csv_path, kernel, use_solvent):
    processed_path, norm_path = processed_paths(root, csv_path, kernel, use_solvent)
    return processed_path.exists() and norm_path.exists()


def delete_cache(root, csv_path, kernel, use_solvent):
    for path in processed_paths(root, csv_path, kernel, use_solvent):
        if path.exists():
            path.unlink()
            print(f"Deleted cache: {path}")


def preprocess_dataset(csv_path, root, kernel, pre_transform, y_mean, y_std, use_solvent, force):
    if force:
        delete_cache(root, csv_path, kernel, use_solvent)

    if cache_ready(root, csv_path, kernel, use_solvent):
        print(f"Cache ready, skip: {csv_path}")
        return

    print(f"Preprocessing: {csv_path}")
    FluorescenceDataset(
        csv_file=csv_path,
        root=str(root),
        kernel=kernel,
        y_mean=y_mean,
        y_std=y_std,
        pre_transform=pre_transform,
        use_solvent=use_solvent,
    )

    processed_path, norm_path = processed_paths(root, csv_path, kernel, use_solvent)
    if not processed_path.exists() or not norm_path.exists():
        raise RuntimeError(f"Cache was not created for {csv_path}")
    print(f"Cache saved: {processed_path}")


def main():
    args = parse_args()
    args.raw_dir = args.raw_dir.resolve()
    root = args.root.resolve() if args.root is not None else args.raw_dir.parent.resolve()
    kernel = normalize_kernel(args.kernel)
    pre_transform = create_rwse_transform(kernel=kernel)
    use_solvent = not args.no_solvent

    for task, config in selected_tasks(args).items():
        source_csv = Path(config["source_csv"]).resolve()
        source_target = config["source_target"]
        target = config["target"]
        source_df = CV.load_scaffold_table(source_csv, source_target, target)

        for fold in args.folds:
            train_csv, valid_csv, train_df, valid_df = CV.write_fold_csvs(
                source_df,
                task=target,
                source_csv=source_csv,
                fold=fold,
                raw_dir=args.raw_dir,
            )
            summary = CV.summarize_split(target, fold, train_df, valid_df, source_df)
            print(
                f"\n[{target} fold {fold}] "
                f"train={summary['train_size']} valid={summary['valid_size']} "
                f"valid_tags={summary['valid_tag_names']}"
            )

            if args.status_only:
                for split_name, csv_path in [("train", train_csv), ("valid", valid_csv)]:
                    status = "ready" if cache_ready(root, csv_path, kernel, use_solvent) else "missing"
                    print(f"{split_name} cache: {status} ({csv_path})")
                continue

            y_mean, y_std = compute_normalization_params(train_csv)
            preprocess_dataset(
                csv_path=train_csv,
                root=root,
                kernel=kernel,
                pre_transform=pre_transform,
                y_mean=y_mean,
                y_std=y_std,
                use_solvent=use_solvent,
                force=args.force,
            )
            preprocess_dataset(
                csv_path=valid_csv,
                root=root,
                kernel=kernel,
                pre_transform=pre_transform,
                y_mean=y_mean,
                y_std=y_std,
                use_solvent=use_solvent,
                force=args.force,
            )

    print("\nScaffold CV preprocessing completed.")


if __name__ == "__main__":
    main()
