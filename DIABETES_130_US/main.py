import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

from constants import DATA_FILE, SEED, FEATURE_NAMES, LABEL_NAME
from diabetes_dataset import preprocess_diabetes
from utils import make_run_key, try_load_results, save_results
from net import DNN, train_kfold as train_kfold_dnn
from TN_net import TN_DNN, train_kfold_TNT
from dGBT_generated import train_kfold_tnt_dgbdt


def train_kfold_xgb_baseline(df, n_splits=5, save_dir=None, force_retrain=False):
    key = make_run_key(model="xgb", splits=n_splits)
    if not force_retrain:
        cached = try_load_results(save_dir, key)
        if cached is not None:
            print(f"[XGBoost] Loaded cached results ({key})")
            return cached

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    labels = df[LABEL_NAME].values

    fold_aurocs = []
    fold_auprcs = []

    for fold, (train_idx, val_idx) in enumerate(tqdm(skf.split(df, labels), total=n_splits, desc="XGBoost folds"), start=1):
        fold_train_df = df.iloc[train_idx]
        fold_val_df   = df.iloc[val_idx]

        X_train = fold_train_df[FEATURE_NAMES].values
        y_train = fold_train_df[LABEL_NAME].values.astype(int)
        X_val   = fold_val_df[FEATURE_NAMES].values
        y_val   = fold_val_df[LABEL_NAME].values.astype(int)

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval   = xgb.DMatrix(X_val)

        params = {
            "objective": "binary:logistic",
            "max_depth": 6,
            "verbosity": 0,
            "seed": SEED,
        }
        model = xgb.train(params, dtrain, num_boost_round=100)

        if save_dir is not None:
            import os; os.makedirs(save_dir, exist_ok=True)
            model.save_model(os.path.join(save_dir, f"{key}__fold{fold}.json"))

        probs = model.predict(dval)

        fold_aurocs.append(roc_auc_score(y_val, probs))
        fold_auprcs.append(average_precision_score(y_val, probs))
        print(f"Fold {fold} — AUROC: {fold_aurocs[-1]:.4f}  AUPRC: {fold_auprcs[-1]:.4f}")

    results = {
        "mean_auroc": float(np.mean(fold_aurocs)),
        "std_auroc":  float(np.std(fold_aurocs)),
        "mean_auprc": float(np.mean(fold_auprcs)),
        "std_auprc":  float(np.std(fold_auprcs)),
    }

    if save_dir is not None:
        save_results(save_dir, key, results)

    return results


def print_results(name, results):
    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    if "mean_auroc" in results:
        print(f"  AUROC: {results['mean_auroc']:.4f} ± {results['std_auroc']:.4f}")
        print(f"  AUPRC: {results['mean_auprc']:.4f} ± {results['std_auprc']:.4f}")
    if "mean_teacher_auroc" in results:
        print(f"  Teacher AUROC: {results['mean_teacher_auroc']:.4f} ± {results['std_teacher_auroc']:.4f}")
        print(f"  Teacher AUPRC: {results['mean_teacher_auprc']:.4f} ± {results['std_teacher_auprc']:.4f}")
    if "mean_soft_tree_auroc" in results:
        print(f"  SDT     AUROC: {results['mean_soft_tree_auroc']:.4f} ± {results['std_soft_tree_auroc']:.4f}")
        print(f"  SDT     AUPRC: {results['mean_soft_tree_auprc']:.4f} ± {results['std_soft_tree_auprc']:.4f}")
    if "mean_dgbdt_auroc" in results:
        print(f"  dGBDT   AUROC: {results['mean_dgbdt_auroc']:.4f} ± {results['std_dgbdt_auroc']:.4f}")
        print(f"  dGBDT   AUPRC: {results['mean_dgbdt_auprc']:.4f} ± {results['std_dgbdt_auprc']:.4f}")
    print(f"{'='*55}")


if __name__ == "__main__":
    DATA_FRACTION = 0.02  # e.g. 0.1 to use 10% of data; None to use all
    MODELS_DIR    = "saved_models"  # directory to save/load results; None to disable
    FORCE_RETRAIN = False           # set True to ignore cached results and retrain

    print("Loading and preprocessing data...")
    df = preprocess_diabetes(DATA_FILE)
    if DATA_FRACTION is not None:
        sss = StratifiedShuffleSplit(n_splits=1, train_size=DATA_FRACTION, random_state=SEED)
        idx, _ = next(sss.split(df, df[LABEL_NAME]))
        df = df.iloc[idx].reset_index(drop=True)
    print(f"Dataset shape after preprocessing: {df.shape}")
    print(f"Positive rate: {df[LABEL_NAME].mean():.4f}")

    DEVICE    = "cpu"
    N_EPOCHS  = 50
    BATCH     = 256
    LR        = 1e-3

    # ── 1. XGBoost baseline ──────────────────────────────────────────────────
    print("\n\n>>> Running XGBoost baseline...")
    xgb_results = train_kfold_xgb_baseline(df, save_dir=MODELS_DIR, force_retrain=FORCE_RETRAIN)
    print_results("XGBoost Baseline", xgb_results)

    # ── 2. Plain DNN ─────────────────────────────────────────────────────────
    print("\n\n>>> Running DNN...")
    dnn_results = train_kfold_dnn(
        df, DNN,
        num_epochs=N_EPOCHS,
        batch_size=BATCH,
        lr=LR,
        device=DEVICE,
        verbose=False,
        save_dir=MODELS_DIR,
        force_retrain=FORCE_RETRAIN,
    )
    print_results("DNN (plain, raw features)", dnn_results)

    # ── 3. TNT + Soft Decision Tree ──────────────────────────────────────────
    print("\n\n>>> Running TNT + Soft Tree...")
    tnt_sdt_results = train_kfold_TNT(
        df, TN_DNN,
        num_epochs=N_EPOCHS,
        batch_size=BATCH,
        lr=LR,
        device=DEVICE,
        distilation=True,
        verbose=False,
        save_dir=MODELS_DIR,
        force_retrain=FORCE_RETRAIN,
    )
    print_results("TNT + Soft Decision Tree", tnt_sdt_results)

    # ── 4. TNT + dGBDT ───────────────────────────────────────────────────────
    print("\n\n>>> Running TNT + dGBDT...")
    tnt_dgbdt_results = train_kfold_tnt_dgbdt(
        df, TN_DNN,
        num_epochs=N_EPOCHS,
        num_epochs_dgbdt=N_EPOCHS,
        batch_size=BATCH,
        lr=LR,
        lr_dgbdt=LR,
        device=DEVICE,
        alpha=0.5,
        temperature=1.0,
        dgbdt_num_trees=5,
        dgbdt_depth=5,
        dgbdt_lamda=1e-3,
        verbose=False,
        save_dir=MODELS_DIR,
        force_retrain=FORCE_RETRAIN,
    )
    print_results("TNT + dGBDT", tnt_dgbdt_results)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n\n" + "="*55)
    print("  FINAL SUMMARY")
    print("="*55)
    print(f"  XGBoost Baseline   AUROC: {xgb_results['mean_auroc']:.4f} ± {xgb_results['std_auroc']:.4f}")
    print(f"  DNN                AUROC: {dnn_results['mean_auroc']:.4f} ± {dnn_results['std_auroc']:.4f}")
    print(f"  TNT Teacher        AUROC: {tnt_sdt_results['mean_teacher_auroc']:.4f} ± {tnt_sdt_results['std_teacher_auroc']:.4f}")
    print(f"  TNT + Soft Tree    AUROC: {tnt_sdt_results['mean_soft_tree_auroc']:.4f} ± {tnt_sdt_results['std_soft_tree_auroc']:.4f}")
    print(f"  TNT + dGBDT        AUROC: {tnt_dgbdt_results['mean_dgbdt_auroc']:.4f} ± {tnt_dgbdt_results['std_dgbdt_auroc']:.4f}")
    print()
    print(f"  XGBoost Baseline   AUPRC: {xgb_results['mean_auprc']:.4f} ± {xgb_results['std_auprc']:.4f}")
    print(f"  DNN                AUPRC: {dnn_results['mean_auprc']:.4f} ± {dnn_results['std_auprc']:.4f}")
    print(f"  TNT Teacher        AUPRC: {tnt_sdt_results['mean_teacher_auprc']:.4f} ± {tnt_sdt_results['std_teacher_auprc']:.4f}")
    print(f"  TNT + Soft Tree    AUPRC: {tnt_sdt_results['mean_soft_tree_auprc']:.4f} ± {tnt_sdt_results['std_soft_tree_auprc']:.4f}")
    print(f"  TNT + dGBDT        AUPRC: {tnt_dgbdt_results['mean_dgbdt_auprc']:.4f} ± {tnt_dgbdt_results['std_dgbdt_auprc']:.4f}")
    print("="*55)
