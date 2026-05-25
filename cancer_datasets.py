
import torch
from torch.utils.data import Dataset
import torch.nn as nn

class BreastCancerDataset(Dataset):
    def __init__(self, df):
        self.X = torch.tensor(df.drop(columns=["ID", "Diagnosis"]).values.astype(float), dtype=torch.float32)
        self.y = torch.tensor(df["Diagnosis"].apply(lambda x: 1 if x == "M" else 0).values.astype(float), dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    

    
class TreeEmbedDataset(Dataset):
    def __init__(self, leaf_indices, labels):
        self.leaf_indices = torch.tensor(leaf_indices, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.leaf_indices[idx], self.labels[idx]
    
