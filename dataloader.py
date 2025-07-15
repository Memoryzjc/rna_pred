import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm

class RNADataset(Dataset):
    """
    RNA数据集类
    """
    def __init__(self, database_path, split='train'):
        """
        Args:
            data_path: str - prediction_split.pt文件的路径
            split: str - 数据分割 ('train', 'val', 'test')
        """
        self.database_path = database_path
        self.data_path = os.path.join(database_path, 'prediction_split.pt')
        self.split = split
        
        # 加载数据
        self.data = self.load_prediction_split_data()
        
    def load_prediction_split_data(self):
        """
        从prediction_split.pt文件加载数据
        
        prediction_split.pt: dict[str, List[str]]
        """
        if self.data_path.endswith('.pt'):
            # 如果是.pt文件，直接加载数据
            data_set = torch.load(self.data_path, weights_only=False)
        else:
            raise FileNotFoundError(f"不支持的文件格式: {self.data_path}")

        # 如果data是字典，提取对应分割的数据
        if isinstance(data_set, dict) and self.split in data_set:
            data_set = data_set[self.split]
        else:
            raise ValueError(f"数据文件应包含键'{self.split}'的字典")

        # 从数据列表中得到数据
        data = []
        if isinstance(data_set, list):
            # 添加进度条显示数据加载进度
            print(f"正在加载 {self.split} 数据...")
            for data_str in tqdm(data_set, desc=f"加载 {self.split} 数据", unit="文件"):
                if isinstance(data_str, str):
                    data_str = os.path.join(self.database_path, 'prediction_data', f'{data_str}.pt')
                    loaded_data = torch.load(data_str, weights_only=False)

                    # 处理坐标中的NaN值, 使用 0 进行填充
                    if 'coords' in loaded_data:
                        coords = loaded_data['coords']
                        if isinstance(coords, torch.Tensor):
                            coords = torch.nan_to_num(coords, nan=0.0)
                else:
                    raise ValueError(f"数据列表中的元素应为字符串路径，但得到: {type(data_str)}")
        else:
            raise ValueError(f"数据集应为列表格式，但得到: {type(data_set)}")


        print(f"成功加载 {self.split} 数据: {len(data)} 个样本")
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        获取单个数据样本
        Returns:
            dict: 包含 'seq', 'coords', 'n_contacts' 的字典
        """
        return self.data[idx]


def get_edges(n_nodes):
    rows, cols = [], []
    for i in range(n_nodes):
        # for j in range(max(0, i-1), min(i+2, n_nodes)):
        for j in range(n_nodes):
            if i != j:  # 排除自环
                rows.append(i)
                cols.append(j)

    edges = [rows, cols]
    return edges


def get_edges_batch(n_nodes: int, batch_size: int) -> tuple:
    """
    EGNN批处理图的边索引生成函数\\
    为批处理数据生成边索引
    Args:
        n_nodes: 每个图的节点数
        batch_size: 批大小
    Returns:
        edges: 批处理后的边索引 -> ( )
        edge_attr: 边属性（全1张量）
    """
    edges = get_edges(n_nodes)  # edges: ()
    edge_attr = torch.ones(len(edges[0]) * batch_size, 1)  # 所有边权重为1
    edges = [torch.LongTensor(edges[0]), torch.LongTensor(edges[1])]
    
    if batch_size == 1:
        return edges, edge_attr
    elif batch_size > 1:
        rows, cols = [], []
        # 为每个图添加偏移量，确保不同图的节点索引不重叠
        for i in range(batch_size):
            rows.append(edges[0] + n_nodes * i)  # 源节点偏移
            cols.append(edges[1] + n_nodes * i)  # 目标节点偏移
        edges = [torch.cat(rows), torch.cat(cols)]
    return edges, edge_attr

def preprocess_data(data_list, atom_idx=11, device='cpu'):
    """
    预处理dict格式的RNA数据

    Args:
    - data_list: List[dict] - batch_size 个字典，每个字典包含:
        - 'seq': str - RNA序列字符串 'AUGC...' (可变长度)
        - 'coords': torch.tensor -> 坐标张量 (seq_len, atom_num, 3)
        - 'n_contacts': int -> 忽略
    - atom_idx: int - 使用的原子索引，默认为11

    Returns:
    - seq_batch: torch.LongTensor -> (batch_size, max_seq_len) 序列索引张量
    - coords_batch: torch.Tensor -> (batch_size, max_seq_len, 3) 坐标张量
    - edges: List[torch.Tensor] -> 边索引张量 [row_indices, col_indices]
    - edge_attr: torch.Tensor -> (num_edges, 1) 边属性张量
    - seq_lengths: torch.Tensor -> (batch_size,) 序列长度张量
    - coord_mask: torch.Tensor -> (batch_size, max_seq_len) 坐标掩码张量
    """
    # RNA碱基到索引的映射
    base_to_idx = {'A': 0, 'U': 1, 'G': 2, 'C': 3, 'N': 4}
    
    batch_size = len(data_list)
    sequences = []
    coordinates = []
    seq_lengths = []
    coord_masks = [] 

    # 首先获取所有序列长度，找出最大长度
    seq_lengths = [len(data['seq']) for data in data_list]
    max_seq_len = max(seq_lengths)

    # 处理每个样本
    for i, data in enumerate(data_list):
        # 1. 处理序列：字符串 -> 数字索引，并填充到最大长度
        seq_string = data['seq']
        seq_indices = [base_to_idx.get(base, 4) for base in seq_string]  # 未知碱基映射为N(4)

        # 填充序列到最大长度（使用N的索引4进行填充）
        while len(seq_indices) < max_seq_len:
            seq_indices.append(4)  # 用N填充

        sequences.append(seq_indices)
        
        # 2. 处理坐标
        coords = data['coords']  # shape: (seq_len, atom_num, 3)
        
        # 确保coords是tensor
        if not isinstance(coords, torch.Tensor):
            coords = torch.tensor(coords, dtype=torch.float32)
        
        # 选择指定的原子坐标
        if coords.dim() == 3:
            selected_coords = coords[:, atom_idx, :]  # 选择第atom_idx个原子 (seq_len, 3)
        elif coords.dim() == 2:
            selected_coords = coords  # 假设已经是 (seq_len, 3)
        else:
            raise ValueError(f"坐标维度错误: {coords.shape}，期望 (seq_len, atom_num, 3) 或 (seq_len, 3)")
        
        # 创建坐标掩码（标识有效坐标位置）
        current_len = selected_coords.size(0)
        coord_mask = torch.zeros(max_seq_len, dtype=torch.bool)
        coord_mask[:current_len] = True  # 前current_len个位置为有效
        coord_masks.append(coord_mask) 

        # 填充坐标到最大长度（使用零向量填充）
        if current_len < max_seq_len:
            padding = torch.zeros((max_seq_len - current_len, 3))  # 使用零填充
            selected_coords = torch.cat([selected_coords, padding], dim=0)

        coordinates.append(selected_coords)
    
    # 3. 转换为张量
    seq_batch = torch.tensor(sequences, dtype=torch.long)  # (batch_size, max_seq_len)
    coords_batch_tensor = torch.stack(coordinates)  # (batch_size, max_seq_len, 3)
    coord_mask_batch = torch.stack(coord_masks)  # (batch_size, max_seq_len)
    seq_lengths = torch.tensor(seq_lengths, dtype=torch.long)  # (batch_size,)

    # 4. 展平坐标以匹配EGNN输入格式
    coords_batch = coords_batch_tensor.view(-1, 3)  # (batch_size * max_seq_len, 3)
    coord_mask_flat = coord_mask_batch.view(-1)  # (batch_size * max_seq_len,)

    # 5. 生成边索引（基于最大序列长度）
    edges, edge_attr = get_edges_batch(max_seq_len, batch_size)

    # 6. 将数据放到device上
    seq_batch = seq_batch.to(device)
    coords_batch = coords_batch.to(device)
    edges = [edge.to(device) for edge in edges]
    edge_attr = edge_attr.to(device)
    seq_lengths = seq_lengths.to(device)
    coord_mask_flat = coord_mask_flat.to(device)

    # return 处理后的数据
    return seq_batch, coords_batch, edges, edge_attr, seq_lengths, coord_mask_flat


if __name__ == "__main__":
    data_path = './rna_database'

    RNADataLoader = create_rna_dataloader(data_path)
