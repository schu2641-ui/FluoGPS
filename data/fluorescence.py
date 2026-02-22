from filecmp import cmpfiles
import os.path as osp
import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset, Data
from ogb.utils import smiles2graph
from tqdm import tqdm


def compute_normalization_params(csv_path, target_col='e'):
    df = pd.read_csv(csv_path)

    all_labels = []
    for idx, row in df.iterrows():
        label = row[target_col]
        if pd.isna(label) or label == '':
            continue
        all_labels.append(label)

    all_labels = torch.tensor(all_labels, dtype=torch.float)
    y_mean = all_labels.mean().item()
    y_std = all_labels.std().item()

    print(f"✓ Computed normalization from '{csv_path}':")
    print(f"  → mean={y_mean:.4f}, std={y_std:.4f}")
    print(f"  → n_samples={len(all_labels)}")

    return y_mean, y_std

class FluorescenceDataset(InMemoryDataset):
    def __init__(self, csv_file, root='datasets/fluorescence', kernel=None,
                 transform=None, pre_transform=None,
                 y_mean=None, y_std=None):
        self.csv_file = csv_file
        self.kernel = kernel
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

        if y_mean is not None and y_std is not None:
            self.y_mean = y_mean
            self.y_std = y_std
            print(f"  → Using provided normalization: mean={y_mean:.4f}, std={y_std:.4f}")
        else:
            norm_params_path = self.processed_paths[1]
            if osp.exists(norm_params_path):
                self.y_mean, self.y_std = torch.load(norm_params_path)
                print(f"  → Loaded normalization from file: mean={self.y_mean:.4f}, std={self.y_std:.4f}")
            else:
                self.y_mean, self.y_std = None, None
                print(f"   Warning: No normalization parameters available!")

    @property
    def raw_file_names(self):
        import os
        return [os.path.basename(self.csv_file)]

    @property
    def processed_file_names(self):
        import os
        csv_basename = os.path.basename(self.csv_file)
        base_name = csv_basename.replace('.csv', '')
        return [f'{base_name}_processed.pt', f'{base_name}_norm_params.pt']

    def download(self):
        pass

    def process(self, y_mean=None, y_std=None):
        df = pd.read_csv(self.raw_paths[0])

        smiles_col = 'smiles' 
        target_col = 'e'  
        solvent_col = 'solvent'
        data_list = []
        print(f"Processing '{self.csv_file}'...")
        if y_mean is not None and y_std is not None:
            print(f"  → Using provided normalization: mean={y_mean:.4f}, std={y_std:.4f}")
        else:
            print("  → Computing normalization from this dataset...")
            all_labels = []
            for idx, row in df.iterrows():
                label = row[target_col]
                if pd.isna(label) or label == '':
                    continue
                all_labels.append(label)

            all_labels = torch.tensor(all_labels, dtype=torch.float)
            y_mean = all_labels.mean().item()
            y_std = all_labels.std().item()
            print(f"  → Computed normalization: mean={y_mean:.4f}, std={y_std:.4f}")

        torch.save((y_mean, y_std), self.processed_paths[1])

        for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
            smiles = row[smiles_col]
            solvent = row[solvent_col]
            label = row[target_col]
            combined_smiles = f"{smiles}.{solvent}"
            if pd.isna(label) or label == '':
                print(f'Skipping missing label for SMILES: {smiles}')
                continue

            try:
                graph = smiles2graph(combined_smiles)
            except:
                print(f"Skipping invalid SMILES: {combined_smiles}")
                continue
            x = torch.from_numpy(graph['node_feat']).to(torch.long)
            edge_index = torch.from_numpy(graph['edge_index']).to(torch.long)
            edge_attr = torch.from_numpy(graph['edge_feat']).to(torch.long)  
            num_nodes = int(graph['num_nodes'])
            normalized_label = (label - y_mean) / y_std
            y = torch.tensor([normalized_label], dtype=torch.float)

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, num_nodes=num_nodes)
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])