import torch
import pandas as pd

from constants import DATA_FILE_BREAST_CANCER, COLUMN_NAMES, FEATURE_NAMES, SEED
from utils import read_data_to_df, create_sparsity, set_seed
from TN_net import TN_DNN, train_kfold_TNT
from dGBT_generated import train_kfold_tnt_dgbdt

MODEL = "dgbdt" # "dgbdt" or "soft_tree"

if __name__ == "__main__":
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = read_data_to_df(DATA_FILE_BREAST_CANCER, COLUMN_NAMES)
    df_sparse_20 = create_sparsity(df, FEATURE_NAMES, sparsity_level=0.2, random_state=SEED)
    df_sparse_40 = create_sparsity(df, FEATURE_NAMES, sparsity_level=0.4, random_state=SEED)

    # Select the dataset to use: df, df_sparse_20, or df_sparse_40
    df_to_use = df_sparse_40
    df_to_use[FEATURE_NAMES] = df_to_use[FEATURE_NAMES].apply(pd.to_numeric, errors="coerce")
    if MODEL == "dgbdt":
        cv_results = train_kfold_tnt_dgbdt(
            df_to_use,
            TN_DNN,
            n_splits=5,
            num_epochs=25,
            num_epochs_dgbdt=35,
            lr=1e-3,
            lr_dgbdt=1e-3,
            device=device,
            verbose=True,
            alpha=0.75,
            temperature=1.0,
            dgbdt_num_trees=5,
            dgbdt_depth=3,
            dgbdt_lamda=1e-3,
        )
        print(f"TN-DNN CV AUROC: {cv_results['mean_teacher_auroc']:.4f} +/- {cv_results['std_teacher_auroc']:.4f}")
        print(f"TN-DNN CV AUPRC: {cv_results['mean_teacher_auprc']:.4f} +/- {cv_results['std_teacher_auprc']:.4f}")
        print(f"dGBDT  CV AUROC: {cv_results['mean_dgbdt_auroc']:.4f} +/- {cv_results['std_dgbdt_auroc']:.4f}")
        print(f"dGBDT  CV AUPRC: {cv_results['mean_dgbdt_auprc']:.4f} +/- {cv_results['std_dgbdt_auprc']:.4f}")
    elif MODEL == "soft_tree":
        cv_results = train_kfold_TNT(
            df_to_use,
            TN_DNN,
            n_splits=5,
            num_epochs=25,
            lr=1e-3,
            device=device,
            verbose=True,
            distilation=True,
        )
        print(f"TN-DNN CV AUROC: {cv_results['mean_teacher_auroc']:.4f} +/- {cv_results['std_teacher_auroc']:.4f}")
        print(f"TN-DNN CV AUPRC: {cv_results['mean_teacher_auprc']:.4f} +/- {cv_results['std_teacher_auprc']:.4f}")
        print(f"Soft Tree CV AUROC: {cv_results['mean_soft_tree_auroc']:.4f} +/- {cv_results['std_soft_tree_auroc']:.4f}")
        print(f"Soft Tree CV AUPRC: {cv_results['mean_soft_tree_auprc']:.4f} +/- {cv_results['std_soft_tree_auprc']:.4f}")

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

    


