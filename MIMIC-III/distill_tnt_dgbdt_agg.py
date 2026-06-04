"""
Knowledge distillation: TNT teacher -> dGBDT student on aggregated features.

Student input (instead of n_predictors * 48 flat features):
  - Static features  (AGE, SURGICAL, ELECTIVE, AIDS, METASTATIC_CANCER, LYMPHOMA):
      keep the single scalar value
  - Temporal features (all other predictors):
      replace the 48-hour series with [min, max, mean, var]  (4 values each)

Teacher pipeline is identical to distill_tnt_dgbdt.py.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score

from preprocess import build_dataset
from main_tnt import get_leaf_indices
from tnt_layers import TreeEmbeddingLayer
from dGBT import dGBDT
from utils import set_seed, seed_worker

STATIC_FEATURES = {'AGE', 'SURGICAL', 'ELECTIVE', 'AIDS', 'METASTATIC_CANCER', 'LYMPHOMA'}


def aggregate_features(X_norm, static_mask):
    """
    X_norm : (n_patients, n_predictors, 48)
    Returns : (n_patients, n_static + n_temporal * 4)
    """
    temporal_mask = ~static_mask
    X_static   = X_norm[:, static_mask, 0]                   # (n, n_static)
    X_temporal = X_norm[:, temporal_mask, :]                  # (n, n_temporal, 48)
    return np.concatenate([
        X_static,
        X_temporal.min(axis=2),
        X_temporal.max(axis=2),
        X_temporal.mean(axis=2),
        X_temporal.var(axis=2),
    ], axis=1)


def load_teacher(fold_idx):
    m   = np.load(f'models/tnt/fold{fold_idx}_mean.npy')
    s   = np.load(f'models/tnt/fold{fold_idx}_std.npy')
    bst = xgb.Booster()
    bst.load_model(f'models/tnt/fold{fold_idx}_xgb_teacher.json')
    tnt_teacher = tf.keras.models.load_model(
        f'models/tnt/fold{fold_idx}_best.keras',
        custom_objects={'TreeEmbeddingLayer': TreeEmbeddingLayer}
    )
    return m, s, bst, tnt_teacher


def run_distillation():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── load dataset ──────────────────────────────────────────────────────────
    input_df, labels_df = build_dataset()
    n_patients   = input_df.index.get_level_values('subject_id').nunique()
    meas_names   = input_df.index.get_level_values('measurement_name').unique().sort_values().tolist()
    n_predictors = len(meas_names)

    static_mask  = np.array([n in STATIC_FEATURES for n in meas_names])
    n_static     = static_mask.sum()
    n_temporal   = n_predictors - n_static
    student_dim  = n_static + n_temporal * 4

    X_raw = input_df.values.reshape((n_patients, n_predictors, 48))
    y     = labels_df['mort_icu'].values.astype(np.float32)

    print(f"Predictors: {n_predictors}  |  static: {n_static}  temporal: {n_temporal}")
    print(f"Student input dim: {student_dim}  (vs flat: {n_predictors * 48})")

    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_idx = list(cv.split(X=np.zeros(n_patients), y=y))

    all_aurocs, all_auprcs = [], []

    for i, (train_idx, val_idx) in enumerate(cv_idx):
        print(f"\n--- Fold {i} Distillation (aggregated features) ---")

        # ── teacher pipeline (unchanged) ──────────────────────────────────────
        m, s, bst, tnt_teacher = load_teacher(i)

        X_tr_norm = np.nan_to_num((X_raw[train_idx] - m) / (s + 1e-8))
        X_vl_norm = np.nan_to_num((X_raw[val_idx]   - m) / (s + 1e-8))

        print("Generating teacher soft labels...")
        L_tr = get_leaf_indices(bst, X_tr_norm)
        L_vl = get_leaf_indices(bst, X_vl_norm)

        y_teacher_tr = tnt_teacher.predict(L_tr, verbose=0).flatten()
        y_teacher_vl = tnt_teacher.predict(L_vl, verbose=0).flatten()

        # ── student inputs: aggregated ────────────────────────────────────────
        X_tr_student = aggregate_features(X_tr_norm, static_mask)
        X_vl_student = aggregate_features(X_vl_norm, static_mask)

        # ── dataloaders ───────────────────────────────────────────────────────
        train_ds = TensorDataset(
            torch.tensor(X_tr_student, dtype=torch.float32),
            torch.tensor(y[train_idx],   dtype=torch.float32).unsqueeze(1),
            torch.tensor(y_teacher_tr,   dtype=torch.float32).unsqueeze(1),
        )
        val_ds = TensorDataset(
            torch.tensor(X_vl_student, dtype=torch.float32),
            torch.tensor(y[val_idx],     dtype=torch.float32).unsqueeze(1),
            torch.tensor(y_teacher_vl,   dtype=torch.float32).unsqueeze(1),
        )

        train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, worker_init_fn=seed_worker)
        val_loader   = DataLoader(val_ds,   batch_size=128, shuffle=False)

        # ── student model ─────────────────────────────────────────────────────
        student = dGBDT(
            input_dim=student_dim,
            num_trees=5,
            tree_depth=5,
            lamda=1e-3,
            use_cuda=(device == "cuda"),
        ).to(device)

        optimizer = optim.Adam(student.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
        mse   = nn.MSELoss()
        alpha = 0.5
        temp  = 1.0

        best_auc, best_prc, best_state = 0.0, 0.0, None
        best_val_loss = float("inf")
        patience, epochs_no_improve   = 10, 0

        print("Training student dGBDT...")
        for epoch in range(1, 101):
            student.train()
            running_loss = 0.0
            for xb, yb, ytb in train_loader:
                xb, yb, ytb = xb.to(device), yb.to(device), ytb.to(device)
                optimizer.zero_grad()

                tree_outputs, penalties, _ = student(xb, is_training_data=True)

                pred_so_far = torch.zeros_like(yb)
                total_loss  = torch.tensor(0.0, device=device)
                for m_idx in range(student.num_trees):
                    residual = (
                        alpha * (temp**2) * (ytb - torch.sigmoid(pred_so_far.detach())) +
                        (1.0 - alpha)     * (yb  - torch.sigmoid(pred_so_far.detach()))
                    )
                    total_loss  = total_loss + mse(tree_outputs[m_idx], residual) + penalties[m_idx]
                    pred_so_far = pred_so_far + tree_outputs[m_idx].detach()

                total_loss.backward()
                optimizer.step()
                running_loss += total_loss.item() * xb.size(0)

            avg_train_loss = running_loss / len(train_idx)

            student.eval()
            val_probs, val_labels = [], []
            val_running_loss = 0.0
            with torch.no_grad():
                for vxb, vyb, vytb in val_loader:
                    vxb, vyb, vytb = vxb.to(device), vyb.to(device), vytb.to(device)
                    v_tree_outputs, v_penalties, v_ensemble_prob = student(vxb, is_training_data=True)
                    val_probs.append(v_ensemble_prob.cpu())
                    val_labels.append(vyb.cpu())

                    v_pred_so_far = torch.zeros_like(vyb)
                    v_total_loss  = torch.tensor(0.0, device=device)
                    for m_idx in range(student.num_trees):
                        v_residual = (
                            alpha * (temp**2) * (vytb - torch.sigmoid(v_pred_so_far)) +
                            (1.0 - alpha)     * (vyb  - torch.sigmoid(v_pred_so_far))
                        )
                        v_total_loss  += mse(v_tree_outputs[m_idx], v_residual) + v_penalties[m_idx]
                        v_pred_so_far  = v_pred_so_far + v_tree_outputs[m_idx]
                    val_running_loss += v_total_loss.item() * vxb.size(0)

            avg_val_loss    = val_running_loss / len(val_idx)
            val_probs_cat   = torch.cat(val_probs).numpy().flatten()
            val_labels_cat  = torch.cat(val_labels).numpy().flatten()
            epoch_auc       = roc_auc_score(val_labels_cat, val_probs_cat)
            epoch_prc       = average_precision_score(val_labels_cat, val_probs_cat)

            scheduler.step(epoch_prc)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epoch_prc > best_prc:
                best_auc          = epoch_auc
                best_prc          = epoch_prc
                best_state        = {k: v.clone() for k, v in student.state_dict().items()}

            print(
                f"  Epoch [{epoch:3d}/100]  "
                f"Train Loss: {avg_train_loss:.6f}  Val Loss: {avg_val_loss:.6f}  "
                f"Val AUROC: {epoch_auc:.4f}  Val AUPRC: {epoch_prc:.4f}"
                + ("  *" if epochs_no_improve == 0 else "")
            )

            if epochs_no_improve >= patience:
                print(f"  Early stopping at epoch {epoch}.")
                break

        student.load_state_dict(best_state)

        os.makedirs('models/distill_agg', exist_ok=True)
        torch.save(student.state_dict(), f'models/distill_agg/dgbdt_student_fold_{i}.pt')

        print(f"\nFold {i} Final Result -> AUROC: {best_auc:.4f}, AUPRC: {best_prc:.4f}")
        all_aurocs.append(best_auc)
        all_auprcs.append(best_prc)

    print(f"\nDistillation Complete.")
    print(f"Mean AUROC: {np.mean(all_aurocs):.4f} ± {np.std(all_aurocs):.4f}")
    print(f"Mean AUPRC: {np.mean(all_auprcs):.4f} ± {np.std(all_auprcs):.4f}")


if __name__ == '__main__':
    run_distillation()
