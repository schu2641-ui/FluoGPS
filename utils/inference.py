from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from ogb.utils import smiles2graph
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from data import compute_normalization_params
from data.loader import create_rwse_transform, normalize_kernel
from models.network import GPSModel


def resolve_device(device_name):
    if str(device_name).startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def ensure_parent_dir(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def resolve_norm_params(norm_params_path, train_csv, task_name):
    if norm_params_path and Path(norm_params_path).exists():
        y_mean, y_std = torch.load(norm_params_path)
        print(f"✓ [{task_name}] Loaded normalization params from: {norm_params_path}")
        print(f"  → mean={y_mean:.4f}, std={y_std:.4f}")
        return y_mean, y_std

    if train_csv and Path(train_csv).exists():
        print(f"✓ [{task_name}] Computing normalization params from: {train_csv}")
        return compute_normalization_params(train_csv)

    print(f"⚠ [{task_name}] No normalization params found. Using mean=0, std=1")
    return 0.0, 1.0


def load_model(checkpoint_path, device, kernel, model_args, task_name):
    model = GPSModel(
        dim_h=model_args["dim_hidden"],
        num_heads=model_args["num_heads"],
        dropout=model_args["dropout"],
        attn_dropout=model_args["attn_dropout"],
        num_layers=model_args["num_layers"],
        dim_out=model_args["dim_out"],
        rwse_steps=normalize_kernel(kernel),
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        epoch = checkpoint.get("epoch", "N/A")
        val_mae = checkpoint.get("val_mae", "N/A")
    else:
        state_dict = checkpoint
        epoch = "N/A"
        val_mae = "N/A"

    model.load_state_dict(state_dict)
    model.eval()
    print(f"✓ [{task_name}] Model loaded from: {checkpoint_path}")
    print(f"  → Best epoch: {epoch}")
    print(f"  → Best Val MAE: {val_mae}")
    return model


def smiles_to_data(smiles, solvent=None, pre_transform=None):
    use_solvent = solvent is not None and not pd.isna(solvent) and str(solvent).strip() != ""
    combined_smiles = smiles if not use_solvent else f"{smiles}.{solvent}"

    try:
        solute_graph = smiles2graph(smiles)
        solvent_graph = smiles2graph(solvent) if use_solvent else None
        graph = smiles2graph(combined_smiles)
    except Exception as exc:
        print(f"Error parsing SMILES '{combined_smiles}': {exc}")
        return None

    solute_num_nodes = int(solute_graph["num_nodes"])
    solvent_num_nodes = int(solvent_graph["num_nodes"]) if solvent_graph is not None else 0
    if int(graph["num_nodes"]) != solute_num_nodes + solvent_num_nodes:
        print(
            "Warning: Combined graph node count "
            f"({graph['num_nodes']}) does not match solute + solvent "
            f"({solute_num_nodes} + {solvent_num_nodes}) for SMILES: {combined_smiles}"
        )

    data = Data(
        x=torch.from_numpy(graph["node_feat"]).to(torch.long),
        edge_index=torch.from_numpy(graph["edge_index"]).to(torch.long),
        edge_attr=torch.from_numpy(graph["edge_feat"]).to(torch.long),
        num_nodes=int(graph["num_nodes"]),
        role=torch.tensor([0] * solute_num_nodes + [1] * solvent_num_nodes, dtype=torch.long),
    )

    if pre_transform is not None:
        data = pre_transform(data)
    return data


def build_graph_dataset(df, smiles_col, solvent_col=None, kernel=None, default_solvent=None):
    pre_transform = create_rwse_transform(kernel=kernel)
    data_list = []
    valid_indices = []

    print("\nConverting SMILES to graphs...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        smiles = row[smiles_col]
        if pd.isna(smiles) or str(smiles).strip() == "":
            print(f"Skipping empty SMILES at row {idx}")
            continue

        if solvent_col and solvent_col in df.columns:
            solvent = row[solvent_col]
            if pd.isna(solvent) or str(solvent).strip() == "":
                solvent = default_solvent
        else:
            solvent = default_solvent

        data = smiles_to_data(str(smiles), solvent, pre_transform)
        if data is None:
            continue

        data_list.append(data)
        valid_indices.append(idx)

    print(f"✓ Successfully converted {len(data_list)}/{len(df)} molecules")
    if not data_list:
        raise ValueError("No valid molecules found in the input CSV.")
    return data_list, valid_indices


def create_prediction_loader(data_list, batch_size, num_workers, device):
    return DataLoader(
        data_list,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )


def run_prediction(model, data_loader, device, y_mean, y_std, task_name):
    all_preds = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Predicting {task_name}"):
            batch = batch.to(device)
            pred, _ = model(batch)
            all_preds.append(pred.cpu())

    preds_tensor = torch.cat(all_preds, dim=0)
    preds_denorm = preds_tensor * y_std + y_mean
    predictions = preds_denorm.squeeze(-1).numpy()
    print(
        f"✓ [{task_name}] range=[{predictions.min():.4f}, {predictions.max():.4f}], "
        f"mean={predictions.mean():.4f}, std={predictions.std():.4f}"
    )
    return predictions


def summarize_regression(y_true, y_pred, task_name):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]

    mae = np.mean(np.abs(y_pred - y_true))
    mse = np.mean((y_pred - y_true) ** 2)
    rmse = np.sqrt(mse)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    print(f"\n[{task_name}] Evaluation Metrics:")
    print(f"  → MAE:  {mae:.4f}")
    print(f"  → MSE:  {mse:.4f}")
    print(f"  → RMSE: {rmse:.4f}")
    print(f"  → R²:   {r2:.4f}")
    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
    }
