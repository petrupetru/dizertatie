import torch
from torch.utils.data import Dataset


class TreeEmbedDataset(Dataset):
    def __init__(self, leaf_indices, labels):
        self.leaf_indices = torch.tensor(leaf_indices, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.leaf_indices[idx], self.labels[idx]
