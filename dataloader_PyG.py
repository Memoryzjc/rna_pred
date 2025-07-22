import torch
import os

from torch_geometric.data import Data, Dataset
from tqdm import tqdm

class RNAPyGDataset(Dataset):
    def __init__(self, root, split='train', transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.root = root
        self.split = split
        self.atom_idx = 11  # The index of the atom to use, e.g., 11 for C1'
        self.data_list = self.load_data()

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]

    def load_data(self):
        # Load split names
        split_path = os.path.join(self.root, 'prediction_split.pt')
        split_dict = torch.load(split_path)
        sample_names = split_dict[self.split]

        data_list = []
        for name in tqdm(sample_names, desc=f"Loading {self.split} set"):
            data_path = os.path.join(self.root, 'prediction_data', f"{name}.pt")
            item = torch.load(data_path)

            seq = item['seq']
            coords = item['coords'][:, self.atom_idx, :]  # Extract the coordinates for the specified atom
            coords = torch.tensor(coords, dtype=torch.float) if not isinstance(coords, torch.Tensor) else coords

            # Create valid mask (True for valid positions, False for NaN)
            valid_mask = ~torch.isnan(coords).any(dim=1)  # [seq_len]

            # Convert sequence to indices
            x = torch.tensor([self.base_to_idx(ch) for ch in seq], dtype=torch.long).unsqueeze(-1)  # [seq_len, 1]

            # Build fully connected edges only between valid nodes
            edge_index = self.get_fully_connected_edges_with_mask(len(seq), valid_mask)

            # all 1s for edge attributes
            edge_attr = torch.ones((edge_index.size(1), 1), dtype=torch.float)

            data = Data(
                x=x, 
                pos=coords, 
                edge_index=edge_index, 
                edge_attr=edge_attr, 
                seq=seq, 
                valid_mask=valid_mask,  # 有效位置的mask
                original_pos=coords    # 保存原始坐标（包含NaN）用于验证
            )
            data_list.append(data)
        return data_list

    @staticmethod
    def base_to_idx(ch):
        return {'A': 0, 'U': 1, 'G': 2, 'C': 3, 'N': 4}.get(ch, 4)

    @staticmethod
    def get_fully_connected_edges(n):
        row, col = zip(*[(i, j) for i in range(n) for j in range(n) if i != j])
        return torch.tensor([row, col], dtype=torch.long)
    
    @staticmethod
    def get_fully_connected_edges_with_mask(n, valid_mask):
        """
        Create fully connected edges only between valid (non-NaN) nodes
        Args:
            n: total number of nodes
            valid_mask: boolean tensor [n] where True indicates valid nodes
        Returns:
            edge_index: [2, num_edges] tensor of edge indices
        """
        valid_indices = torch.where(valid_mask)[0]
        
        if len(valid_indices) < 2:
            # Return empty edges if less than 2 valid nodes
            return torch.zeros((2, 0), dtype=torch.long)
        
        row, col = [], []
        for i in valid_indices:
            for j in valid_indices:
                if i != j:
                    row.append(i.item())
                    col.append(j.item())
        
        return torch.tensor([row, col], dtype=torch.long)