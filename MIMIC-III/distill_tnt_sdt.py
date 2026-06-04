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
from SDT import SDT
from utils import set_seed, seed_worker

def load_teacher(fold_idx):
    """Loads all components of the teacher pipeline for a specific fold."""
    # Load normalization stats saved during main_tnt execution
    m = np.load(f'models/tnt/fold{fold_idx}_mean.npy')
    s = np.load(f'models/tnt/fold{fold_idx}_std.npy')
    
    # Load XGBoost (used for feature engineering / leaf index extraction)
    bst = xgb.Booster()
    bst.load_model(f'models/tnt/fold{fold_idx}_xgb_teacher.json')
    
    # Load TNT Model (Neural Component)
    model_path = f'models/tnt/fold{fold_idx}_best.keras'
    # custom_objects is required to correctly instantiate the TreeEmbeddingLayer
    tnt_teacher = tf.keras.models.load_model(
        model_path, 
        custom_objects={'TreeEmbeddingLayer': TreeEmbeddingLayer}
    )
    
    return m, s, bst, tnt_teacher

def run_distillation():
    """Performs knowledge distillation from TNT ConvNet to a single SDT student."""
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Dataset with sparse=True to preserve NaNs for SDT's internal masking
    input_df, labels_df = build_dataset(sparse=True)
    n_patients = input_df.index.get_level_values('subject_id').nunique()
    n_predictors = input_df.index.get_level_values('measurement_name').nunique()
    
    X_raw = input_df.values.reshape((n_patients, n_predictors, 48))
    y = labels_df['mort_icu'].values.astype(np.float32)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_idx = list(cv.split(X=np.zeros(n_patients), y=y))
    
    all_aurocs, all_auprcs = [], []
    
    for i, (train_idx, val_idx) in enumerate(cv_idx):
        print(f"\n--- Fold {i} Distillation (TNT Teacher -> Single SDT Student) ---")
        
        # 2. Load Teacher Pipeline
        m, s, bst, tnt_teacher = load_teacher(i)
        
        # 3. Data Preprocessing
        # For the teacher components (XGBoost/TNT-ConvNet), we zero-fill NaNs
        X_tr_norm_teacher = np.nan_to_num((X_raw[train_idx] - m) / (s + 1e-8))
        X_vl_norm_teacher = np.nan_to_num((X_raw[val_idx]   - m) / (s + 1e-8))
        
        # Generate Teacher soft labels
        print("Generating teacher soft labels...")
        L_tr = get_leaf_indices(bst, X_tr_norm_teacher)
        L_vl = get_leaf_indices(bst, X_vl_norm_teacher)
        
        # The teacher provides soft probabilities for distillation
        y_teacher_tr = tnt_teacher.predict(L_tr, verbose=0).flatten()
        y_teacher_vl = tnt_teacher.predict(L_vl, verbose=0).flatten()
        
        # For the SDT student, we standardize but KEEP NaNs intact (so SDT can build its binary mask)
        X_tr_student = ((X_raw[train_idx] - m) / (s + 1e-8)).reshape(len(train_idx), -1)
        X_vl_student = ((X_raw[val_idx]   - m) / (s + 1e-8)).reshape(len(val_idx), -1)
        
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
        
        # 5. Distillation Training for Single Soft Decision Tree (SDT)
        student = SDT(
            input_dim=n_predictors * 48, 
            output_dim=1, 
            depth=5, 
            lamda=1e-3, 
            use_cuda=(device == "cuda")
        ).to(device)
        
        optimizer = optim.Adam(student.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
        mse = nn.MSELoss()
        
        alpha = 0.8  # Distillation mixing parameter
        temp  = 2.0  # Temperature softening scaling

        best_auc = 0.0
        best_val_loss = float("inf")
        best_prc = 0.0
        best_state = None
        patience = 10
        epochs_no_improve = 0
        
        print("Training student SDT...")
        for epoch in range(1, 101):
            student.train()
            running_loss = 0.0
            for xb, yb, ytb in train_loader:
                xb, yb, ytb = xb.to(device), yb.to(device), ytb.to(device)
                optimizer.zero_grad()
                
                # Forward pass: get logit output and regularization penalty
                logits, penalty = student(xb, is_training_data=True)
                preds = torch.sigmoid(logits)
                
                # Classical Distillation Loss matching SDT structure:
                # soft labels (MSE with temperature) + hard labels (MSE) + path penalty
                loss = alpha * (temp**2) * mse(preds, ytb) + (1.0 - alpha) * mse(preds, yb) + penalty
                
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * xb.size(0)
            
            avg_train_loss = running_loss / len(train_idx)

            # --- Validation Phase ---
            student.eval()
            val_probs, val_labels = [], []
            val_running_loss = 0.0
            with torch.no_grad():
                for vxb, vyb, vytb in val_loader:
                    vxb, vyb, vytb = vxb.to(device), vyb.to(device), vytb.to(device)
                    v_logits, v_penalty = student(vxb, is_training_data=True)
                    v_preds = torch.sigmoid(v_logits)
                    
                    val_probs.append(v_preds.cpu())
                    val_labels.append(vyb.cpu())

                    v_loss = alpha * (temp**2) * mse(v_preds, vytb) + (1.0 - alpha) * mse(v_preds, vyb) + v_penalty
                    val_running_loss += v_loss.item() * vxb.size(0)
            
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
        
        os.makedirs('models/distill_sdt', exist_ok=True)
        torch.save(student.state_dict(), f'models/distill_sdt/sdt_student_fold_{i}.pt')

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