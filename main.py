import torch

from constants import DATA_FILE_BREAST_CANCER, COLUMN_NAMES, FEATURE_NAMES, SEED
import net
from utils import read_data_to_df, create_sparsity
from net import (
    DNN, BreastCancerDataset,
    train_val_test_split, train_val_test_datasets, train_val_test_dataloaders,
    train, test
)
import torch.optim as optim
import torch.nn as nn
import pandas as pd



if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = read_data_to_df(DATA_FILE_BREAST_CANCER, COLUMN_NAMES)
    df_sparse_20 = create_sparsity(df, FEATURE_NAMES, sparsity_level=0.2, random_state=SEED)
    df_sparse_40 = create_sparsity(df, FEATURE_NAMES, sparsity_level=0.4, random_state=SEED)

    dnn = DNN()

    ### select the df to use for training
    df_to_use = df

    df_to_use[FEATURE_NAMES] = df[FEATURE_NAMES].apply(pd.to_numeric, errors="coerce")

    train_df, val_df, test_df = train_val_test_split(df_to_use)

    train_means = train_df[FEATURE_NAMES].mean()
    train_df[FEATURE_NAMES] = train_df[FEATURE_NAMES].fillna(train_means)
    val_df[FEATURE_NAMES] = val_df[FEATURE_NAMES].fillna(train_means)
    test_df[FEATURE_NAMES] = test_df[FEATURE_NAMES].fillna(train_means)

    #write the sparse datasets to csv files
    train_df.to_csv("data\\breast_cancer_data_sparse_20_train.csv", index=False)

    train_dataset, val_dataset, test_dataset = train_val_test_datasets(train_df, val_df, test_df)
    train_loader, val_loader, test_loader = train_val_test_dataloaders(train_dataset, val_dataset, test_dataset)

    train(dnn, train_loader, val_loader, num_epochs=20, lr=1e-3, device=device)

    print('Finished Training')

    test(dnn, test_loader, device=device)
    