import os
import random

import pandas as pd
import numpy as np
import torch

from constants import DATA_FILE_BREAST_CANCER, COLUMN_NAMES, FEATURE_NAMES, SEED


def read_data_to_df(file, columns):
    """
    Reads the breast cancer data from the specified file.
    Create dataframe with the data and return it.
    """
    
    with open(file, "r") as f:
        data = f.readlines()
    # Process each line to create a list of lists (rows)
    data = [line.strip().split(",") for line in data]
    df = pd.DataFrame(data, columns=columns)
    #write data to csv file
    df.to_csv("data\\breast_cancer_data.csv", index=False)
    return df

def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def create_sparsity(df, feature_columns, sparsity_level, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    df_sparse = df.copy()
    features = df_sparse[feature_columns]
    
    n_rows, n_cols = features.shape
    total_elements = n_rows * n_cols
    
    n_to_remove = int(total_elements * sparsity_level)
    
    if n_to_remove == 0:
        return df_sparse

    flat_indices = np.random.choice(total_elements, size=n_to_remove, replace=False)
    row_indices, col_indices = np.unravel_index(flat_indices, features.shape)
    for row, col in zip(row_indices, col_indices):
        features.iat[row, col] = np.nan
    
    df_sparse[feature_columns] = features
    
    return df_sparse



def _assert_finite_tensor(name, tensor):
    if not torch.isfinite(tensor).all():
        nan_count = torch.isnan(tensor).sum().item()
        inf_count = torch.isinf(tensor).sum().item()
        raise ValueError(f"{name} has {nan_count} NaNs and {inf_count} Infs")

if __name__ == "__main__":

    df = read_data_to_df(DATA_FILE_BREAST_CANCER, COLUMN_NAMES)
    df_sparse_20 = create_sparsity(df, FEATURE_NAMES, sparsity_level=0.2, random_state=SEED).fillna(df[FEATURE_NAMES].mean())
    # Save the sparse datasets to CSV files
    df_sparse_20.to_csv("data\\breast_cancer_data_sparse_20.csv", index=False)
