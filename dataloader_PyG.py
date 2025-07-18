import torch
import os

from torch_geometric.data import Data, Dataset
from tqdm import tqdm

class RNAPyGDataset(Dataset):
    def __init__(self, root, split='train', transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.root = root
        self.split = split
        self.data_list = self.load_data()
        self.atom_idx = 11  # The index of the atom to use, e.g., 11 for C1'

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
            coords = item['coords']
            coord = torch.nan_to_num(coords[:, self.atom_idx, :], nan=0.0)  # [seq_len, 3]

            # Convert sequence to indices
            x = torch.tensor([self.base_to_idx(ch) for ch in seq], dtype=torch.long).unsqueeze(-1)  # [seq_len, 1]

            # Build fully connected edges
            edge_index = self.get_fully_connected_edges(len(seq))

            # Optionally add edge_attr as 1s
            edge_attr = torch.ones((edge_index.size(1), 1), dtype=torch.float)

            data = Data(x=x, pos=coord, edge_index=edge_index, edge_attr=edge_attr, seq=seq)
            data_list.append(data)
        return data_list

    @staticmethod
    def base_to_idx(ch):
        return {'A': 0, 'U': 1, 'G': 2, 'C': 3, 'N': 4}.get(ch, 4)

    @staticmethod
    def get_fully_connected_edges(n):
        row, col = zip(*[(i, j) for i in range(n) for j in range(n) if i != j])
        return torch.tensor([row, col], dtype=torch.long)