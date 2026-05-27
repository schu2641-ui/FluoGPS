import argparse
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "datasets" / "fluorescence" / "raw"
TASKS = ("abs", "emi", "plqy", "e")


def default_task_config(task):
    return {
        "checkpoint": str(ROOT_DIR / "checkpoints" / f"{task}.pt"),
        "train_csv": str(DATA_DIR / f"{task}_train.csv"),
        "norm_params": None,
    }


def add_task_args(parser):
    for task in TASKS:
        parser.add_argument(
            f"--{task}_checkpoint",
            type=str,
            default=str(ROOT_DIR / "checkpoints" / f"{task}.pt"),
            help=f"{task} model checkpoint path.",
        )
        parser.add_argument(
            f"--{task}_train_csv",
            type=str,
            default=str(DATA_DIR / f"{task}_train.csv"),
            help=f"{task} training CSV used to recover normalization.",
        )
        parser.add_argument(
            f"--{task}_norm_params",
            type=str,
            default=None,
            help=f"Optional {task} normalization parameter file.",
        )


def normalize_tasks(tasks):
    if "all" in tasks:
        return list(TASKS)
    return tasks


def selected_task_configs(args):
    tasks = normalize_tasks(args.tasks)
    configs = {}
    for task in tasks:
        config = default_task_config(task)
        config["checkpoint"] = getattr(args, f"{task}_checkpoint")
        config["train_csv"] = getattr(args, f"{task}_train_csv")
        config["norm_params"] = getattr(args, f"{task}_norm_params")
        configs[task] = config

    single_task_overrides = {
        "checkpoint": args.checkpoint,
        "train_csv": args.train_csv,
        "norm_params": args.norm_params,
    }
    active_overrides = {key: value for key, value in single_task_overrides.items() if value is not None}
    if active_overrides:
        if len(tasks) != 1:
            raise ValueError("--checkpoint, --train_csv, and --norm_params can only be used with one selected task.")
        configs[tasks[0]].update(active_overrides)

    return configs


def parse_args():
    parser = argparse.ArgumentParser(description="FluoGPS prediction")
    parser.add_argument("--input_csv", type=str, default=str(ROOT_DIR / "examples" / "inputs" / "data.csv"), help="Input CSV path.")
    parser.add_argument(
        "--output_csv",
        type=str,
        default=str(ROOT_DIR / "outputs" / "predictions" / "predictions.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        choices=(*TASKS, "all"),
        default=["abs"],
        help="Prediction tasks to run. Use 'all' for abs emi plqy e.",
    )
    add_task_args(parser)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Backward-compatible checkpoint override for a single selected task.",
    )
    parser.add_argument(
        "--train_csv",
        type=str,
        default=None,
        help="Backward-compatible training CSV override for a single selected task.",
    )
    parser.add_argument("--norm_params", type=str, default=None, help="Backward-compatible normalization override for a single selected task.")
    parser.add_argument("--kernel", type=int, nargs="+", default=list(range(1, 17)), help="RWSE steps.")
    parser.add_argument("--num_layers", type=int, default=10, help="Number of GPS layers.")
    parser.add_argument("--dim_out", type=int, default=1, help="Output dimension.")
    parser.add_argument("--dropout", type=float, default=0.05, help="Dropout rate.")
    parser.add_argument("--attn_dropout", type=float, default=0.1, help="Attention dropout rate.")
    parser.add_argument("--dim_hidden", type=int, default=128, help="Hidden dimension.")
    parser.add_argument("--num_heads", type=int, default=4, help="Attention heads.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader worker count.")
    parser.add_argument("--device", type=str, default="cuda", help="Device name.")
    parser.add_argument("--smiles_col", type=str, default="smiles", help="SMILES column name.")
    parser.add_argument("--solvent_col", type=str, default="solvent", help="Optional solvent column name.")
    parser.add_argument("--default_solvent", type=str, default="CS(=O)C", help="Fallback solvent SMILES.")
    return parser.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")
    task_configs = selected_task_configs(args)

    import pandas as pd

    from utils.inference import (
        build_graph_dataset,
        create_prediction_loader,
        ensure_parent_dir,
        load_model,
        resolve_device,
        resolve_norm_params,
        run_prediction,
    )
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    model_args = {
        "dim_hidden": args.dim_hidden,
        "num_heads": args.num_heads,
        "dropout": args.dropout,
        "attn_dropout": args.attn_dropout,
        "num_layers": args.num_layers,
        "dim_out": args.dim_out,
    }

    print(f"\nLoading input data from: {args.input_csv}")
    df = pd.read_csv(args.input_csv)
    print(f"✓ Loaded {len(df)} molecules")
    if args.smiles_col not in df.columns:
        raise ValueError(f"Column '{args.smiles_col}' not found in CSV. Available columns: {df.columns.tolist()}")

    data_list, valid_indices = build_graph_dataset(
        df=df,
        smiles_col=args.smiles_col,
        solvent_col=args.solvent_col,
        kernel=args.kernel,
        default_solvent=args.default_solvent,
    )
    loader = create_prediction_loader(
        data_list,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )

    results_df = df.iloc[valid_indices].copy()
    results_df.insert(0, "source_index", valid_indices)

    for task_name, config in task_configs.items():
        checkpoint_path = Path(config["checkpoint"])
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"[{task_name}] Checkpoint not found: {config['checkpoint']}")

        y_mean, y_std = resolve_norm_params(
            norm_params_path=config["norm_params"],
            train_csv=config["train_csv"],
            task_name=task_name,
        )
        model = load_model(
            checkpoint_path=config["checkpoint"],
            device=device,
            kernel=args.kernel,
            model_args=model_args,
            task_name=task_name,
        )
        print(f"\nRunning {task_name} predictions...")
        predictions = run_prediction(model, loader, device, y_mean, y_std, task_name)
        results_df[f"predicted_{task_name}"] = predictions

    ensure_parent_dir(args.output_csv)
    results_df.to_csv(args.output_csv, index=False)
    print(f"\n✓ Predictions saved to: {args.output_csv}")


if __name__ == "__main__":
    main()
