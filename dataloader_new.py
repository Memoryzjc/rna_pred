import torch
from torch import nn
from torch_geometric.data import Data, Dataset, DataLoader
from torch_geometric.utils import dense_to_sparse
import os
from tqdm import tqdm
import numpy as np

class RNAGeometricDataset(Dataset):
    """
    RNA数据集类，使用PyTorch Geometric
    """
    def __init__(self, database_path, split='train', atom_idx=11, transform=None, pre_transform=None):
        """
        Args:
            database_path: str - 数据库路径
            split: str - 数据分割 ('train', 'val', 'test')
            atom_idx: int - 使用的原子索引，默认为11
            transform: callable - 数据变换函数
            pre_transform: callable - 预处理变换函数
        """
        self.database_path = database_path
        self.data_path = os.path.join(database_path, 'prediction_split.pt')
        self.split = split
        self.atom_idx = atom_idx
        
        # RNA碱基到索引的映射
        self.base_to_idx = {'A': 0, 'U': 1, 'G': 2, 'C': 3, 'N': 4}
        
        super().__init__(None, transform, pre_transform)
        
        # 加载数据路径列表
        self.data_paths = self.load_prediction_split_data()
        
    def load_prediction_split_data(self):
        """
        从prediction_split.pt文件加载数据路径列表
        
        Returns:
            List[str]: 数据文件路径列表
        """
        if self.data_path.endswith('.pt'):
            data_set = torch.load(self.data_path, weights_only=False)
        else:
            raise FileNotFoundError(f"不支持的文件格式: {self.data_path}")

        # 如果data是字典，提取对应分割的数据
        if isinstance(data_set, dict) and self.split in data_set:
            data_set = data_set[self.split]
        else:
            raise ValueError(f"数据文件应包含键'{self.split}'的字典")

        # 验证数据格式
        if not isinstance(data_set, list):
            raise ValueError(f"数据集应为列表格式，但得到: {type(data_set)}")

        print(f"成功加载 {self.split} 数据路径: {len(data_set)} 个样本")
        return data_set

    def len(self):
        """返回数据集大小"""
        return len(self.data_paths)

    def get(self, idx):
        """
        获取单个数据样本，返回PyTorch Geometric的Data对象
        
        Args:
            idx: int - 样本索引
            
        Returns:
            Data: PyTorch Geometric数据对象，包含:
                - x: 节点特征 (seq_len, feature_dim)
                - pos: 节点坐标 (seq_len, 3)
                - edge_index: 边索引 (2, num_edges)
                - edge_attr: 边属性 (num_edges, edge_feature_dim)
                - seq: 原始序列字符串
                - seq_len: 序列长度
        """
        # 构建数据文件路径
        data_str = self.data_paths[idx]
        if isinstance(data_str, str):
            data_path = os.path.join(self.database_path, 'prediction_data', f'{data_str}.pt')
            loaded_data = torch.load(data_path, weights_only=False)
        else:
            raise ValueError(f"数据路径应为字符串，但得到: {type(data_str)}")

        # 提取序列和坐标
        seq_string = loaded_data['seq']
        coords = loaded_data['coords']
        
        # 处理序列：字符串 -> 数字索引
        seq_indices = [self.base_to_idx.get(base, 4) for base in seq_string]
        seq_len = len(seq_indices)
        
        # 处理坐标
        if not isinstance(coords, torch.Tensor):
            coords = torch.tensor(coords, dtype=torch.float32)
        
        # 处理坐标中的NaN值
        coords = torch.nan_to_num(coords, nan=0.0)
        
        # 选择指定的原子坐标
        if coords.dim() == 3:
            pos = coords[:, self.atom_idx, :]  # (seq_len, 3)
        elif coords.dim() == 2:
            pos = coords  # 假设已经是 (seq_len, 3)
        else:
            raise ValueError(f"坐标维度错误: {coords.shape}，期望 (seq_len, atom_num, 3) 或 (seq_len, 3)")
        
        # 创建节点特征 (one-hot编码)
        x = torch.zeros(seq_len, 5)  # 5种碱基类型
        for i, base_idx in enumerate(seq_indices):
            x[i, base_idx] = 1.0
        
        # 生成全连接图的边索引
        edge_index = self.get_fully_connected_edges(seq_len)
        
        # 计算边属性（距离）
        edge_attr = self.compute_edge_attributes(pos, edge_index)
        
        # 创建PyTorch Geometric数据对象
        data = Data(
            x=x,
            pos=pos,
            edge_index=edge_index,
            edge_attr=edge_attr,
            seq=seq_string,
            seq_len=torch.tensor(seq_len, dtype=torch.long),
            y=torch.tensor(loaded_data.get('n_contacts', 0), dtype=torch.long)  # 如果有标签
        )
        
        return data
    
    def get_fully_connected_edges(self, n_nodes):
        """
        生成全连接图的边索引
        
        Args:
            n_nodes: int - 节点数量
            
        Returns:
            torch.Tensor: 边索引 (2, num_edges)
        """
        rows, cols = [], []
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:  # 排除自环
                    rows.append(i)
                    cols.append(j)
        
        edge_index = torch.tensor([rows, cols], dtype=torch.long)
        return edge_index
    
    def compute_edge_attributes(self, pos, edge_index):
        """
        计算边属性（欧几里德距离）
        
        Args:
            pos: torch.Tensor - 节点坐标 (n_nodes, 3)
            edge_index: torch.Tensor - 边索引 (2, num_edges)
            
        Returns:
            torch.Tensor: 边属性 (num_edges, 1)
        """
        row, col = edge_index
        diff = pos[row] - pos[col]  # (num_edges, 3)
        dist = torch.norm(diff, dim=1, keepdim=True)  # (num_edges, 1)
        return dist


def create_geometric_dataloader(database_path, split='train', batch_size=32, 
                              shuffle=True, atom_idx=11, num_workers=0):
    """
    创建PyTorch Geometric数据加载器
    
    Args:
        database_path: str - 数据库路径
        split: str - 数据分割
        batch_size: int - 批大小
        shuffle: bool - 是否打乱数据
        atom_idx: int - 原子索引
        num_workers: int - 工作进程数
        
    Returns:
        DataLoader: PyTorch Geometric数据加载器
    """
    dataset = RNAGeometricDataset(
        database_path=database_path,
        split=split,
        atom_idx=atom_idx
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        follow_batch=['x', 'pos']  # 为批处理创建batch索引
    )
    
    return dataset, dataloader


def collate_geometric_batch(batch):
    """
    自定义批处理函数，用于处理可变长度序列
    
    Args:
        batch: List[Data] - 数据对象列表
        
    Returns:
        Data: 批处理后的数据对象
    """
    from torch_geometric.data import Batch
    return Batch.from_data_list(batch)


class RNAGeometricDataModule:
    """
    RNA数据模块，管理训练、验证、测试数据加载器
    """
    def __init__(self, database_path, batch_size=32, atom_idx=11, num_workers=0):
        self.database_path = database_path
        self.batch_size = batch_size
        self.atom_idx = atom_idx
        self.num_workers = num_workers
        
    def train_dataloader(self):
        """返回训练数据加载器"""
        return create_geometric_dataloader(
            self.database_path, 
            split='train',
            batch_size=self.batch_size,
            shuffle=True,
            atom_idx=self.atom_idx,
            num_workers=self.num_workers
        )
    
    def val_dataloader(self):
        """返回验证数据加载器"""
        return create_geometric_dataloader(
            self.database_path,
            split='val', 
            batch_size=self.batch_size,
            shuffle=False,
            atom_idx=self.atom_idx,
            num_workers=self.num_workers
        )
    
    def test_dataloader(self):
        """返回测试数据加载器"""
        return create_geometric_dataloader(
            self.database_path,
            split='test',
            batch_size=self.batch_size, 
            shuffle=False,
            atom_idx=self.atom_idx,
            num_workers=self.num_workers
        )


if __name__ == "__main__":
    # 使用示例
    database_path = "/path/to/your/database"
    
    # 创建数据模块
    data_module = RNAGeometricDataModule(
        database_path=database_path,
        batch_size=16,
        atom_idx=11
    )
    
    # 获取训练数据加载器
    train_loader = data_module.train_dataloader()
    
    # 测试数据加载
    print("测试数据加载...")
    for batch_idx, batch in enumerate(train_loader):
        print(f"Batch {batch_idx}:")
        print(f"  - 节点特征形状: {batch.x.shape}")
        print(f"  - 坐标形状: {batch.pos.shape}")
        print(f"  - 边索引形状: {batch.edge_index.shape}")
        print(f"  - 边属性形状: {batch.edge_attr.shape}")
        print(f"  - 批大小: {batch.batch.max().item() + 1}")
        if batch_idx >= 2:  # 只测试前3个批次
            break