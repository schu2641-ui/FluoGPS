from filecmp import cmpfiles
import os.path as osp
import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset, Data
from ogb.utils import smiles2graph
from tqdm import tqdm


def compute_normalization_params(csv_path, target_col='abs'):
    """
    从 CSV 文件计算归一化参数 (均值和标准差)

    Args:
        csv_path: CSV 文件路径
        target_col: 目标列名

    Returns:
        (y_mean, y_std): 归一化均值和标准差
    """
    df = pd.read_csv(csv_path)

    # 收集所有有效的标签值
    all_labels = []
    for idx, row in df.iterrows():
        label = row[target_col]
        if pd.isna(label) or label == '':
            continue
        all_labels.append(label)

    # 计算均值和标准差
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
        """
        Args:
            csv_file: CSV 文件名
            root: 数据集保存的根目录
            kernel: RWSE 计算的 kernel steps
            y_mean: 归一化均值 (如果为 None,则从文件加载或计算)
            y_std: 归一化标准差 (如果为 None,则从文件加载或计算)
        """
        self.csv_file = csv_file
        self.kernel = kernel
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

        # ⭐ 优先使用传入的归一化参数,否则尝试从文件加载
        if y_mean is not None and y_std is not None:
            self.y_mean = y_mean
            self.y_std = y_std
            print(f"  → Using provided normalization: mean={y_mean:.4f}, std={y_std:.4f}")
        else:
            # 尝试从文件加载 (用于向后兼容)
            norm_params_path = self.processed_paths[1]
            if osp.exists(norm_params_path):
                self.y_mean, self.y_std = torch.load(norm_params_path)
                print(f"  → Loaded normalization from file: mean={self.y_mean:.4f}, std={self.y_std:.4f}")
            else:
                self.y_mean, self.y_std = None, None
                print(f"  ⚠️  Warning: No normalization parameters available!")

    @property
    def raw_file_names(self):
        # [MOD] 你的原始 CSV 文件名，请确保该文件在 datasets/fluorescence/raw/ 目录下
        # ⭐ 只提取文件名部分，避免完整路径导致查找错误
        import os
        return [os.path.basename(self.csv_file)]

    @property
    def processed_file_names(self):
        # 使用 csv_file 生成唯一的文件名，避免缓存冲突
        # ⭐ 只提取文件名部分，避免完整路径导致保存位置错误
        import os
        csv_basename = os.path.basename(self.csv_file)
        base_name = csv_basename.replace('.csv', '')
        return [f'{base_name}_processed.pt', f'{base_name}_norm_params.pt']

    def download(self):
        # 如果文件在本地 raw 文件夹，不需要下载逻辑
        pass

    def process(self, y_mean=None, y_std=None):
        """
        Args:
            y_mean: 归一化均值 (如果提供,则使用该值)
            y_std: 归一化标准差 (如果提供,则使用该值)
        """
        # 读取 CSV
        df = pd.read_csv(self.raw_paths[0])

        # [MOD] 修改为你 CSV 中的列名
        smiles_col = 'smiles'  # SMILES 列名
        target_col = 'abs'   # 预测目标（吸收波长）
        solvent_col = 'solvent'
        data_list = []
        print(f"Processing '{self.csv_file}'...")

        # ⭐ 第一步：使用提供的归一化参数,或从数据中计算
        if y_mean is not None and y_std is not None:
            # 使用外部提供的归一化参数 (推荐)
            print(f"  → Using provided normalization: mean={y_mean:.4f}, std={y_std:.4f}")
        else:
            # 从当前数据集计算归一化参数 (向后兼容)
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

        # 保存归一化参数到文件 (用于记录)
        torch.save((y_mean, y_std), self.processed_paths[1])

        # 第二步：处理数据并应用归一化
        for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
            smiles = row[smiles_col]
            solvent = row[solvent_col]
            label = row[target_col]
            combined_smiles = f"{smiles}.{solvent}"
            if pd.isna(label) or label == '':
                print(f'Skipping missing label for SMILES: {smiles}')
                continue

            try:
                # 使用 OGB 强大的特征提取器 (包含原子类型、键类型、共轭情况等)
                graph = smiles2graph(combined_smiles)
            except:
                print(f"Skipping invalid SMILES: {combined_smiles}")
                continue

            # 转换为 PyG Data 对象
            x = torch.from_numpy(graph['node_feat']).to(torch.long)
            edge_index = torch.from_numpy(graph['edge_index']).to(torch.long)
            edge_attr = torch.from_numpy(graph['edge_feat']).to(torch.long)  # 这里包含了单双键信息
            num_nodes = int(graph['num_nodes'])
            # 归一化 y 值： (y - mean) / std
            normalized_label = (label - y_mean) / y_std
            y = torch.tensor([normalized_label], dtype=torch.float)

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, num_nodes=num_nodes)
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # 保存处理后的数据
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def get(self, idx):
        
            # 获取原始数据
        data = super().get(idx) #调用父类的 get 方法获取数据，self.data是所有图的拼接数据。所以这里获取的是idx对应的单个图数据。
        
        # 动态计算 RWSE（如果需要）
        if hasattr(self, 'kernel') and self.kernel is not None:
            from .loader import get_rw_landing_probs
            rwse = get_rw_landing_probs(
                ksteps=self.kernel,
                edge_index=data.edge_index,
                num_nodes=data.num_nodes
            )
            data.pestat_RWSE = rwse
        
        return data