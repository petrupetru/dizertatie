import torch
import pandas as pd
import numpy as np
import xgboost as xgb

from constants import DATA_FILE_BREAST_CANCER, COLUMN_NAMES, FEATURE_NAMES, LABEL_NAME, SEED
from utils import read_data_to_df, create_sparsity
from net import DNN, train_kfold
from TN_net import TN_DNN, train_kfold as train_kfold_tn
from tree_embeddings import create_tree_embeddings


# test imports
from sklearn.metrics import roc_auc_score, average_precision_score


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = read_data_to_df(DATA_FILE_BREAST_CANCER, COLUMN_NAMES)
    df_sparse_20 = create_sparsity(df, FEATURE_NAMES, sparsity_level=0.2, random_state=SEED)
    df_sparse_40 = create_sparsity(df, FEATURE_NAMES, sparsity_level=0.4, random_state=SEED)

    # Select the dataset to use: df, df_sparse_20, or df_sparse_40
    df_to_use = df
    df_to_use[FEATURE_NAMES] = df_to_use[FEATURE_NAMES].apply(pd.to_numeric, errors="coerce")

    cv_results_tn = train_kfold_tn(
        df_to_use,
        TN_DNN,
        n_splits=5,
        num_epochs=25,
        lr=1e-3,
        device=device,
    )
    print(f"TN-DNN CV AUROC: {cv_results_tn['mean_auroc']:.4f} +/- {cv_results_tn['std_auroc']:.4f}")
    print(f"TN-DNN CV AUPRC: {cv_results_tn['mean_auprc']:.4f} +/- {cv_results_tn['std_auprc']:.4f}")

    # # test the tree-based embeddings and evaluate the gbr performance
    # gbr, tree_embeddings = create_tree_embeddings(df_to_use)
    
    # # Encode labels: B (benign) = 0, M (malignant) = 1
    # y_encoded = np.where(df_to_use[LABEL_NAME].values == 'M', 1, 0)
    
    # # Use DMatrix for predictions with XGBoost
    # dtest = xgb.DMatrix(df_to_use[FEATURE_NAMES].values)
    # gbr_predictions = gbr.predict(dtest)
    # gbr_auroc = roc_auc_score(y_encoded, gbr_predictions)
    # gbr_auprc = average_precision_score(y_encoded, gbr_predictions)
    # print(f"GBR AUROC: {gbr_auroc:.4f}")
    # print(f"GBR AUPRC: {gbr_auprc:.4f}")

    


