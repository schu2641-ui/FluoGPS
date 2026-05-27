from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import torch
from ogb.utils import smiles2graph
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm


def infer_target_column(csv_path):
    return Path(csv_path).stem.split("_")[0]


def transform_target_series(values, target_col):
    labels = pd.to_numeric(values, errors="coerce")
    if target_col == "e":
        labels = labels.where(labels > 0)
        labels = labels.map(lambda value: math.log10(value) if pd.notna(value) else value)
    return labels


def transform_target_value(value, target_col):
    try:
        label = float(value)
    except (TypeError, ValueError):
        return None

    if pd.isna(label):
        return None
    if target_col == "e":
        if label <= 0:
            return None
        return math.log10(label)
    return label


def compute_normalization_params(csv_path):
    df = pd.read_csv(csv_path)
    target_col = infer_target_column(csv_path)
    all_labels = transform_target_series(df[target_col], target_col).dropna().to_list()
    all_labels = torch.tensor(all_labels, dtype=torch.float)
    y_mean = all_labels.mean().item()
    y_std = all_labels.std().item()
    if y_std == 0:
        y_std = 1.0

    print(f"✓ Computed normalization from '{csv_path}':")
    if target_col == "e":
        print("  → target transform: log10(e)")
    print(f"  → mean={y_mean:.4f}, std={y_std:.4f}")
    print(f"  → n_samples={len(all_labels)}")
    return y_mean, y_std


class FluorescenceDataset(InMemoryDataset):
    def __init__(
        self,
        csv_file,
        root="datasets/fluorescence",
        kernel=None,
        transform=None,
        pre_transform=None,
        y_mean=None,
        y_std=None,
        use_solvent=True,
    ):
        self.csv_file = str(csv_file)
        self.kernel = list(kernel) if kernel is not None else None
        self.provided_y_mean = y_mean
        self.provided_y_std = y_std
        self.use_solvent = use_solvent
        super().__init__(root, transform, pre_transform)
        self._ensure_normalization_alignment()
        self.data, self.slices = torch.load(self.processed_paths[0])

        if self.provided_y_mean is not None and self.provided_y_std is not None:
            self.y_mean = self.provided_y_mean
            self.y_std = self.provided_y_std
            print(f"  → Using provided normalization: mean={self.y_mean:.4f}, std={self.y_std:.4f}")
            return

        norm_params_path = Path(self.processed_paths[1])
        if norm_params_path.exists():
            self.y_mean, self.y_std = torch.load(norm_params_path)
            print(f"  → Loaded normalization from file: mean={self.y_mean:.4f}, std={self.y_std:.4f}")
        else:
            self.y_mean, self.y_std = None, None
            print("   Warning: No normalization parameters available!")

    def _ensure_normalization_alignment(self):
        if self.provided_y_mean is None or self.provided_y_std is None:
            return

        norm_params_path = Path(self.processed_paths[1])
        needs_reprocess = True
        if norm_params_path.exists():
            stored_mean, stored_std = torch.load(norm_params_path)
            needs_reprocess = (
                abs(float(stored_mean) - float(self.provided_y_mean)) > 1e-8
                or abs(float(stored_std) - float(self.provided_y_std)) > 1e-8
            )

        if needs_reprocess:
            print("  → Reprocessing dataset to align normalization with training split.")
            self.process()

    @property
    def raw_file_names(self):
        return [Path(self.csv_file).name]

    @property
    def processed_file_names(self):
        base_name = Path(self.csv_file).stem
        kernel_suffix = f"_k{len(self.kernel)}" if self.kernel is not None else ""
        solvent_suffix = "" if self.use_solvent else "_nosolvent"
        return [
            f"{base_name}{kernel_suffix}{solvent_suffix}_processed.pt",
            f"{base_name}{kernel_suffix}{solvent_suffix}_norm_params.pt",
        ]

    def download(self):
        pass

    def process(self):
        df = pd.read_csv(self.raw_paths[0])
        smiles_col = "smiles"
        target_col = infer_target_column(self.csv_file)
        solvent_col = "solvent"
        data_list = []
        print(f"Processing '{self.csv_file}'...")

        y_mean = self.provided_y_mean
        y_std = self.provided_y_std
        if y_mean is not None and y_std is not None:
            print(f"  → Using provided normalization: mean={y_mean:.4f}, std={y_std:.4f}")
        else:
            print("  → Computing normalization from this dataset...")
            all_labels = transform_target_series(df[target_col], target_col).dropna().to_list()
            all_labels = torch.tensor(all_labels, dtype=torch.float)
            y_mean = all_labels.mean().item()
            y_std = all_labels.std().item()
            if y_std == 0:
                y_std = 1.0
            if target_col == "e":
                print("  → target transform: log10(e)")
            print(f"  → Computed normalization: mean={y_mean:.4f}, std={y_std:.4f}")

        torch.save((y_mean, y_std), self.processed_paths[1])

        for _, row in tqdm(df.iterrows(), total=df.shape[0]):
            smiles = row[smiles_col]
            solvent = row[solvent_col] if solvent_col in df.columns else None
            label = transform_target_value(row[target_col], target_col)
            combined_smiles = smiles if (not self.use_solvent or pd.isna(solvent)) else f"{smiles}.{solvent}"
            if label is None:
                print(f"Skipping missing label for SMILES: {smiles}")
                continue

            try:
                solute_graph = smiles2graph(smiles)
                solvent_graph = smiles2graph(solvent) if (self.use_solvent and solvent and not pd.isna(solvent)) else None
                graph = smiles2graph(combined_smiles)

                solute_num_nodes = solute_graph['num_nodes']
                solvent_num_nodes = 0
                if solvent_graph is not None:
                    solvent_num_nodes = solvent_graph['num_nodes']
                    if graph['num_nodes'] != solute_num_nodes + solvent_num_nodes:
                        print(f"Warning: Combined graph node count ({graph['num_nodes']}) does not match sum of solute ({solute_num_nodes}) and solvent ({solvent_num_nodes}) for SMILES: {combined_smiles}")
            except Exception:
                print(f"Skipping invalid SMILES: {combined_smiles}")
                continue
            
            role = torch.tensor([0]*solute_num_nodes + [1]*solvent_num_nodes, dtype=torch.long)
            x = torch.from_numpy(graph["node_feat"]).to(torch.long)
            edge_index = torch.from_numpy(graph["edge_index"]).to(torch.long)
            edge_attr = torch.from_numpy(graph["edge_feat"]).to(torch.long)
            num_nodes = int(graph["num_nodes"])
            normalized_label = (label - y_mean) / y_std
            y = torch.tensor([normalized_label], dtype=torch.float)

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, num_nodes=num_nodes, role=role)
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
