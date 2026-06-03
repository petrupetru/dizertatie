import torch
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from constants import DATA_FILE_BREAST_CANCER, COLUMN_NAMES, FEATURE_NAMES, LABEL_NAME, SEED
from utils import read_data_to_df, create_sparsity, set_seed, seed_worker
from TN_net import TN_DNN, train_kfold as train_kfold_tn, train_kfold_TNT
from dGBT_generated import train_kfold_tnt_dgbdt, train_kfold_dnn_dgbdt
from net import DNN, train_kfold as train_kfold_dnn
from SDT import SDT
from cancer_datasets import BreastCancerDataset

MODEL = "TNT-dgbdt" # "XGBoost", "dnn", "soft_tree", "tn_dnn",  "DNN-dGBDT", "TNT-soft_tree", "TNT-dgbdt"or "all"

def evaluate_gbdt_kfold(df, n_splits=5, num_trees=80, max_depth=9, verbose=False):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    labels = df[LABEL_NAME].values
    fold_aurocs, fold_auprcs = [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, labels), start=1):
        fold_train_df = df.iloc[train_idx].copy()
        fold_val_df = df.iloc[val_idx].copy()

        X_train = fold_train_df[FEATURE_NAMES].values
        y_train = np.where(fold_train_df[LABEL_NAME].values == "M", 1, 0)
        dtrain = xgb.DMatrix(X_train, label=y_train)

        params = {"objective": "reg:squarederror", "max_depth": max_depth, "seed": SEED, "verbosity": 0}
        model = xgb.train(params, dtrain, num_boost_round=num_trees)

        X_val = fold_val_df[FEATURE_NAMES].values
        y_val = np.where(fold_val_df[LABEL_NAME].values == "M", 1, 0)
        dval = xgb.DMatrix(X_val)
        preds = model.predict(dval)

        fold_aurocs.append(roc_auc_score(y_val, preds))
        fold_auprcs.append(average_precision_score(y_val, preds))

    return {"mean_auroc": np.mean(fold_aurocs), "std_auroc": np.std(fold_aurocs),
            "mean_auprc": np.mean(fold_auprcs), "std_auprc": np.std(fold_auprcs)}

def evaluate_sdt_kfold(df, n_splits=5, num_epochs=75, lr=1e-3, batch_size=32, device="cpu", verbose=True):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    labels = df[LABEL_NAME].values
    fold_aurocs, fold_auprcs = [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, labels), start=1):
        fold_train_df = df.iloc[train_idx].copy()
        fold_val_df = df.iloc[val_idx].copy()
        train_ds = BreastCancerDataset(fold_train_df)
        val_ds = BreastCancerDataset(fold_val_df)

        g = torch.Generator(); g.manual_seed(SEED + fold)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g, worker_init_fn=seed_worker)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, generator=g, worker_init_fn=seed_worker)

        model = SDT(input_dim=len(FEATURE_NAMES), output_dim=1, depth=10, use_cuda=(device=="cuda")).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                preds, penalty = model(xb, is_training_data=True)
                loss = torch.nn.functional.mse_loss(torch.sigmoid(preds), yb) + penalty
                loss.backward(); optimizer.step()
                train_loss += loss.item() * xb.size(0)
            train_loss /= len(train_loader.dataset)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    preds, penalty = model(xb, is_training_data=True)
                    loss = torch.nn.functional.mse_loss(torch.sigmoid(preds), yb) + penalty
                    val_loss += loss.item() * xb.size(0)
            val_loss /= len(val_loader.dataset)

            if verbose:
                print(f"Fold {fold} Epoch {epoch+1:3d}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

        model.eval()
        all_p, all_y = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                all_p.append(torch.sigmoid(model(xb)).cpu())
                all_y.append(yb)
        all_p = torch.cat(all_p).numpy().flatten()
        all_y = torch.cat(all_y).numpy().flatten()
        fold_aurocs.append(roc_auc_score(all_y, all_p))
        fold_auprcs.append(average_precision_score(all_y, all_p))

    return {"mean_auroc": np.mean(fold_aurocs), "std_auroc": np.std(fold_aurocs),
            "mean_auprc": np.mean(fold_auprcs), "std_auprc": np.std(fold_auprcs)}

if __name__ == "__main__":
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = read_data_to_df(DATA_FILE_BREAST_CANCER, COLUMN_NAMES)
    df_sparse_20 = create_sparsity(df, FEATURE_NAMES, sparsity_level=0.2, random_state=SEED)
    df_sparse_40 = create_sparsity(df, FEATURE_NAMES, sparsity_level=0.4, random_state=SEED)

    # Select the dataset to use: df, df_sparse_20, or df_sparse_40
    df_to_use = df
    df_to_use[FEATURE_NAMES] = df_to_use[FEATURE_NAMES].apply(pd.to_numeric, errors="coerce")

    if MODEL == "XGBoost" or MODEL == "all":
        print("Running Baseline Comparisons...")
        res_xgb = evaluate_gbdt_kfold(df_to_use)
        print(f"XGBoost  CV AUROC: {res_xgb['mean_auroc']:.4f} ± {res_xgb['std_auroc']:.4f}")
        print(f"XGBoost  CV AUPRC: {res_xgb['mean_auprc']:.4f} ± {res_xgb['std_auprc']:.4f}")

    if MODEL == "soft_tree" or MODEL == "all":
        res_sdt = evaluate_sdt_kfold(df_to_use, device=device)
        print(f"SDT      CV AUROC: {res_sdt['mean_auroc']:.4f} ± {res_sdt['std_auroc']:.4f}")
        print(f"SDT      CV AUPRC: {res_sdt['mean_auprc']:.4f} ± {res_sdt['std_auprc']:.4f}")

    
    if MODEL == "dnn" or MODEL == "all":
        res_dnn = train_kfold_dnn(df_to_use, DNN, n_splits=5, num_epochs=25, lr=1e-3, device=device, verbose=True)
        print(f"DNN      CV AUROC: {res_dnn['mean_auroc']:.4f} ± {res_dnn['std_auroc']:.4f}")
        print(f"DNN      CV AUPRC: {res_dnn['mean_auprc']:.4f} ± {res_dnn['std_auprc']:.4f}")

    if MODEL == "tn_dnn" or MODEL == "all":
        res_tn = train_kfold_tn(df_to_use, TN_DNN, n_splits=5, num_epochs=25, lr=1e-3, device=device, verbose=True)
        print(f"XGB-DNN  CV AUROC: {res_tn['mean_auroc']:.4f} ± {res_tn['std_auroc']:.4f}")
        print(f"XGB-DNN  CV AUPRC: {res_tn['mean_auprc']:.4f} ± {res_tn['std_auprc']:.4f}")

    if MODEL == "TNT-dgbdt" or MODEL == "all":
        print("\nRunning dGBDT (TNT) Training...")
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
            dgbdt_depth=6,
            dgbdt_lamda=1e-3,
        )
        print(f"dGBDT  CV AUROC: {cv_results['mean_dgbdt_auroc']:.4f} +/- {cv_results['std_dgbdt_auroc']:.4f}")
        print(f"dGBDT  CV AUPRC: {cv_results['mean_dgbdt_auprc']:.4f} +/- {cv_results['std_dgbdt_auprc']:.4f}")

    if MODEL == "DNN-dGBDT" or MODEL == "all":
        print("\nRunning dGBDT (Distilled from DNN) Training...")
        cv_results = train_kfold_dnn_dgbdt(
            df_to_use,
            n_splits=5,
            num_epochs_teacher=25,
            num_epochs_dgbdt=35,
            lr=1e-3,
            lr_dgbdt=1e-3,
            device=device,
            verbose=True,
            alpha=0.75,
            temperature=1.0,
            dgbdt_num_trees=5,
            dgbdt_depth=6,
            dgbdt_lamda=1e-3,
        )
        print(f"Teacher DNN CV AUROC: {cv_results['mean_teacher_auroc']:.4f} +/- {cv_results['std_teacher_auroc']:.4f}")
        print(f"dGBDT Student CV AUROC: {cv_results['mean_dgbdt_auroc']:.4f} +/- {cv_results['std_dgbdt_auroc']:.4f}")
        print(f"dGBDT Student CV AUPRC: {cv_results['mean_dgbdt_auprc']:.4f} +/- {cv_results['std_dgbdt_auprc']:.4f}")

    if MODEL == "TNT-soft_tree" or MODEL == "all":
        print("\nRunning Soft Decision Tree Distillation...")
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
        print(f"Soft Tree CV AUROC: {cv_results['mean_soft_tree_auroc']:.4f} +/- {cv_results['std_soft_tree_auroc']:.4f}")
        print(f"Soft Tree CV AUPRC: {cv_results['mean_soft_tree_auprc']:.4f} +/- {cv_results['std_soft_tree_auprc']:.4f}")
    
    # # Encode labels: B (benign) = 0, M (malignant) = 1
    # y_encoded = np.where(df_to_use[LABEL_NAME].values == 'M', 1, 0)
    
    # # Use DMatrix for predictions with XGBoost
    # dtest = xgb.DMatrix(df_to_use[FEATURE_NAMES].values)
    # gbr_predictions = gbr.predict(dtest)
    # gbr_auroc = roc_auc_score(y_encoded, gbr_predictions)
    # gbr_auprc = average_precision_score(y_encoded, gbr_predictions)
    # print(f"GBR AUROC: {gbr_auroc:.4f}")
    # print(f"GBR AUPRC: {gbr_auprc:.4f}")

    
