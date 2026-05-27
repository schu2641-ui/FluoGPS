import argparse
import random
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
DEFAULT_DATA_DIR = ROOT_DIR / "datasets" / "fluorescence" / "raw"
DEFAULT_LOG_DIR = ROOT_DIR / "outputs" / "runs"
PROPERTY_GROUPS = {
    1: {
        "properties": ["abs", "emi"],
        "dropout": 0.05,
        "attn_dropout": 0.1,
        "lr": 0.001,
        "max_epoch": 300,
    },
    2: {
        "properties": ["plqy", "e"],
        "dropout": 0.05,
        "attn_dropout": 0.1,
        "lr": 0.001,
        "max_epoch": 300,
    },
}

def parse_args():
    parser = argparse.ArgumentParser(description="FluoGPS model training")
    parser.add_argument("--property_group", type=int, default=2, choices=[1, 2], help="Property group to train.")
    parser.add_argument(
        "--properties",
        nargs="+",
        choices=["abs", "emi", "plqy", "e"],
        default=None,
        help="Properties to train. Overrides --property_group when set.",
    )
    parser.add_argument("--train_csv", type=str, default=str(DEFAULT_DATA_DIR / "abs_train.csv"), help="Training CSV.")
    parser.add_argument("--val_csv", type=str, default=str(DEFAULT_DATA_DIR / "abs_valid.csv"), help="Validation CSV.")
    parser.add_argument("--test_csv", type=str, default=str(DEFAULT_DATA_DIR / "abs_test.csv"), help="Test CSV.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader worker count.")
    parser.add_argument("--kernel", type=int, nargs="+", default=list(range(1, 17)), help="RWSE steps.")
    parser.add_argument("--num_layers", type=int, default=10, help="Number of GPS layers.")
    parser.add_argument("--dim_out", type=int, default=1, help="Output dimension.")
    parser.add_argument("--dropout", type=float, default=None, help="Override group dropout rate.")
    parser.add_argument("--attn_dropout", type=float, default=None, help="Override group attention dropout rate.")
    parser.add_argument("--dim_hidden", type=int, default=128, help="Hidden dimension.")
    parser.add_argument("--num_heads", type=int, default=4, help="Attention head count.")
    parser.add_argument("--lr", type=float, default=None, help="Override group learning rate.")
    parser.add_argument("--warm_steps", type=float, default=15, help="Warmup steps.")
    parser.add_argument("--weight_decay", type=float, default=0, help="Weight decay.")
    parser.add_argument("--max_epoch", type=int, default=None, help="Override group maximum training epochs.")
    parser.add_argument("--device", type=str, default="cuda", help="Device name.")
    parser.add_argument("--log_dir", type=str, default=str(DEFAULT_LOG_DIR), help="Base output directory.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument("--repeat", type=int, default=1, help="Number of consecutive seeds to run.")
    return parser.parse_args()


def resolve_group_config(args, group_config):
    return {
        "dropout": args.dropout if args.dropout is not None else group_config["dropout"],
        "attn_dropout": args.attn_dropout if args.attn_dropout is not None else group_config["attn_dropout"],
        "lr": args.lr if args.lr is not None else group_config["lr"],
        "max_epoch": args.max_epoch if args.max_epoch is not None else group_config["max_epoch"],
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


def resolve_device(device_name):
    import torch

    if str(device_name).startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def run_once(args, group_config, run_seed, run_id, property_name):
    from models.network import GPSModel
    from train.trainer import create_loader, custom_train
    from utils import schedule_with_warmup
    from utils.simple_logger import create_simple_logger
    import torch

    set_seed(run_seed)
    device = resolve_device(args.device)
    training_config = resolve_group_config(args, group_config)
    base_log_dir = Path(args.log_dir) / property_name
    run_log_dir = base_log_dir / f"seed_{run_seed}" if args.repeat > 1 else base_log_dir
    run_log_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Training {property_name} | "
        f"dropout={training_config['dropout']} | "
        f"attn_dropout={training_config['attn_dropout']} | "
        f"lr={training_config['lr']} | "
        f"max_epoch={training_config['max_epoch']}"
    )
    train_csv = str(DEFAULT_DATA_DIR / f"{property_name}_train.csv")
    val_csv = str(DEFAULT_DATA_DIR / f"{property_name}_valid.csv")
    test_csv = str(DEFAULT_DATA_DIR / f"{property_name}_test.csv")
    train_loader, val_loader, test_loader = create_loader(
        train_dataset_dir=train_csv,
        val_dataset_dir=val_csv,
        test_dataset_dir=test_csv,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        kernel=args.kernel,
    )

    model = GPSModel(
        dim_h=args.dim_hidden,
        num_heads=args.num_heads,
        dropout=training_config["dropout"],
        attn_dropout=training_config["attn_dropout"],
        num_layers=args.num_layers,
        dim_out=args.dim_out,
        rwse_steps=args.kernel,
    ).to(device)
    loggers = create_simple_logger(output_dir=str(run_log_dir))
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_config["lr"], weight_decay=args.weight_decay)
    scheduler = schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.warm_steps,
        num_training_steps=training_config["max_epoch"],
    )

    result = custom_train(
        model=model,
        loaders=[train_loader, val_loader, test_loader],
        loggers=loggers,
        optimizer=optimizer,
        scheduler=scheduler,
        max_epoch=training_config["max_epoch"],
    )

    print(
        f"[Run {run_id}] seed={run_seed} | "
        f"Best Val MAE={result['best_val_mae']:.4f} | "
        f"Final Test MAE={result['final_test_mae']:.4f}"
    )
    return result


def main(args):
    import numpy as np
    group_config = PROPERTY_GROUPS[args.property_group]
    properties = args.properties if args.properties is not None else group_config["properties"]
    final_results = {}
    for prop in properties:
        results = []
        for run_id in range(args.repeat):
            run_seed = args.seed + run_id
            print(f"\n{'=' * 60}\nStarting {prop} run {run_id + 1}/{args.repeat} with seed={run_seed}\n{'=' * 60}")
            results.append(run_once(args, group_config, run_seed, run_id + 1, property_name=prop))
        final_results[prop] = results
    if len(final_results) > 1:
        for prop, results in final_results.items():
                print(f"\n{'=' * 60}")
                print(f"Summary for property: {prop}")
                val_maes = np.array([result["best_val_mae"] for result in results], dtype=float)
                test_maes = np.array([result["final_test_mae"] for result in results], dtype=float)
                print(f"\n{'=' * 60}")
                print("Multi-seed summary")
                print(f"Seeds: {[args.seed + index for index in range(args.repeat)]}")
                print(f"Best Val MAE: mean={val_maes.mean():.4f}, std={val_maes.std(ddof=0):.4f}")
                print(f"Final Test MAE: mean={test_maes.mean():.4f}, std={test_maes.std(ddof=0):.4f}")
                print(f"{'=' * 60}")


if __name__ == "__main__":
    main(parse_args())
