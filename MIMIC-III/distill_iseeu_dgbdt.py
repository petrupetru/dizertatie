import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score

from preprocess import build_dataset
from dGBT import dGBDT
from utils import set_seed, seed_worker

def load_teacher(fold_idx):
    """Loads the ISeeU teacher model for a specific fold."""
    model_path = f'models/iseeu/kfold{fold_idx}_best.keras'
    teacher = tf.keras.models.load_model(model_path)
    return teacher

def run_distillation():
    """Performs knowledge distillation from ISeeU ConvNet to dGBDT."""
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Dataset
    input_df, labels_df = build_dataset()
    n_patients = input_df.index.get_level_values('subject_id').nunique()
    n_predictors = input_df.index.get_level_values('measurement_name').nunique()
    
    X_raw = input_df.values.reshape((n_patients, n_predictors, 48))
    y = labels_df['mort_icu'].values.astype(np.float32)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_idx = list(cv.split(X=np.zeros(n_patients), y=y))
    
    # Calculate fold-specific mean and std on the fly matching main.py exactly
    subject_ids = input_df.index.get_level_values('subject_id').unique()
    fold_mean, fold_std = [], []
    for train_idx, _ in cv_idx:
        train_subjects = subject_ids[train_idx]
        fold_data = input_df.loc[train_subjects]
        m = fold_data.groupby(level='measurement_name').mean().mean(axis=1).values[:, None]
        s = fold_data.groupby(level='measurement_name').std().mean(axis=1).values[:, None]
        fold_mean.append(m)
        fold_std.append(s)
    
    all_aurocs, all_auprcs = [], []
    
    for i, (train_idx, val_idx) in enumerate(cv_idx):
        print(f"\n--- Fold {i} Distillation (ISeeU Teacher) ---")
        
        # 2. Load Teacher Model
        teacher = load_teacher(i)
        
        # 3. Data Preprocessing & Normalization matching main.py
        # Reshape to (N, 1, n_predictors, 48) for the ConvNet teacher
        X_tr_4d = X_raw[train_idx].reshape((-1, 1, n_predictors, 48))
        X_vl_4d = X_raw[val_idx].reshape((-1, 1, n_predictors, 48))
        
        X_tr_norm_teacher = np.nan_to_num((X_tr_4d - fold_mean[i]) / (fold_std[i] + 1e-8))
        X_vl_norm_teacher = np.nan_to_num((X_vl_4d - fold_mean[i]) / (fold_std[i] + 1e-8))
        
        # Generate Teacher soft labels
        print("Generating teacher soft labels...")
        y_teacher_tr = teacher.predict(X_tr_norm_teacher, batch_size=256, verbose=0).flatten()
        y_teacher_vl = teacher.predict(X_vl_norm_teacher, batch_size=256, verbose=0).flatten()
        
        # Prepare Student Inputs (Flattened 1D features: 22 predictors * 48 hours = 1056)
        X_tr_student = X_tr_norm_teacher.reshape(len(train_idx), -1)
        X_vl_student = X_vl_norm_teacher.reshape(len(val_idx), -1)
        
        # 4. Create PyTorch Loaders
        train_ds = TensorDataset(
            torch.tensor(X_tr_student, dtype=torch.float32),
            torch.tensor(y[train_idx], dtype=torch.float32).unsqueeze(1),
            torch.tensor(y_teacher_tr, dtype=torch.float32).unsqueeze(1)
        )
        val_ds = TensorDataset(
            torch.tensor(X_vl_student, dtype=torch.float32),
            torch.tensor(y[val_idx], dtype=torch.float32).unsqueeze(1),
            torch.tensor(y_teacher_vl, dtype=torch.float32).unsqueeze(1)
        )
        
        train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, worker_init_fn=seed_worker)
        val_loader   = DataLoader(val_ds,   batch_size=128, shuffle=False)
        
        # 5. Distillation Training
        student = dGBDT(
            input_dim=n_predictors * 48, 
            num_trees=6, 
            tree_depth=3, 
            lamda=1e-3, 
            use_cuda=(device == "cuda")
        ).to(device)
        
        optimizer = optim.Adam(student.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
        mse = nn.MSELoss()
        
        alpha = 0.8  # Distillation mixing parameter
        temp  = 2.0  # Temperature softening (standard for probability-based distillation)

        best_auc = 0.0
        best_val_loss = float("inf")
        best_prc = 0.0
        best_state = None
        patience = 10
        epochs_no_improve = 0
        
        print("Training student dGBDT...")
        for epoch in range(1, 101):
            student.train()
            running_loss = 0.0
            for xb, yb, ytb in train_loader:
                xb, yb, ytb = xb.to(device), yb.to(device), ytb.to(device)
                optimizer.zero_grad()
                
                # Forward pass: get logit per tree and regularization penalties
                tree_outputs, penalties, _ = student(xb, is_training_data=True)
                
                # Stage-wise residual boosting distillation
                pred_so_far = torch.zeros_like(yb)  # raw logit accumulator
                total_loss = torch.tensor(0.0, device=device)
                
                for m_idx in range(student.num_trees):
                    # Negative gradient of the distillation loss w.r.t s_{m-1}
                    residual = (
                        alpha * (temp**2) * (ytb - torch.sigmoid(pred_so_far.detach())) +
                        (1.0 - alpha) * (yb - torch.sigmoid(pred_so_far.detach()))
                    )
                    
                    total_loss = total_loss + mse(tree_outputs[m_idx], residual) + penalties[m_idx]
                    pred_so_far = pred_so_far + tree_outputs[m_idx].detach()
                
                total_loss.backward()
                optimizer.step()
                running_loss += total_loss.item() * xb.size(0)
            
            avg_train_loss = running_loss / len(train_idx)

            # --- Validation Phase ---
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
                    v_total_loss = torch.tensor(0.0, device=device)
                    for m_idx in range(student.num_trees):
                        v_residual = (
                            alpha * (temp**2) * (vytb - torch.sigmoid(v_pred_so_far)) +
                            (1.0 - alpha) * (vyb - torch.sigmoid(v_pred_so_far))
                        )
                        v_total_loss += mse(v_tree_outputs[m_idx], v_residual) + v_penalties[m_idx]
                        v_pred_so_far = v_pred_so_far + v_tree_outputs[m_idx]
                    val_running_loss += v_total_loss.item() * vxb.size(0)
            
            avg_val_loss = val_running_loss / len(val_idx)
            val_probs_cat = torch.cat(val_probs).numpy().flatten()
            val_labels_cat = torch.cat(val_labels).numpy().flatten()
            epoch_auc = roc_auc_score(val_labels_cat, val_probs_cat)
            epoch_prc = average_precision_score(val_labels_cat, val_probs_cat)

            scheduler.step(epoch_prc)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epoch_prc > best_prc:
                best_auc = epoch_auc
                best_prc = epoch_prc
                best_state = {k: v.clone() for k, v in student.state_dict().items()}

            print(
                f"  Epoch [{epoch:3d}/100]  "
                f"Train Loss: {avg_train_loss:.6f}  Val Loss: {avg_val_loss:.6f}  "
                f"Val AUROC: {epoch_auc:.4f}  Val AUPRC: {epoch_prc:.4f}"
                + ("  *" if epochs_no_improve == 0 else "")
            )

            if epochs_no_improve >= patience:
                print(f"  Early stopping at epoch {epoch}.")
                break

        # 6. Cross-Validation Evaluation
        student.load_state_dict(best_state)
        
        os.makedirs('models/distill_iseeu', exist_ok=True)
        torch.save(student.state_dict(), f'models/distill_iseeu/dgbdt_student_fold_{i}.pt')

        fold_auc = best_auc
        fold_prc = best_prc
        
        print(f"\nFold {i} Final Result -> AUROC: {fold_auc:.4f}, AUPRC: {fold_prc:.4f}")
        all_aurocs.append(fold_auc)
        all_auprcs.append(fold_prc)

    print(f"\nDistillation Complete for all folds.")
    print(f"Mean AUROC: {np.mean(all_aurocs):.4f} ± {np.std(all_aurocs):.4f}")
    print(f"Mean AUPRC: {np.mean(all_auprcs):.4f} ± {np.std(all_auprcs):.4f}")

if __name__ == '__main__':
    run_distillation()