"""
Standalone single Soft Decision Tree (SDT) training on MIMIC-III.
Loads data, performs 5-fold cross-validation, and evaluates AUROC & AUPRC.
Supports the native sparsity-aware routing mechanism with sparse data.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score

from preprocess import build_dataset
from SDT import SDT
from utils import set_seed, seed_worker

def train_sdt_fold(
    fold_idx,
    x_train,
    x_val,
    y_train,
    y_val,
    device,
    batch_size=128,
    num_epochs=100,
    lr=1e-3,
    depth=5,
    lamda=1e-3,
    patience=10,
):
    # Convert labels to tensors
    y_tr_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    y_vl_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    # Convert inputs to tensors (keeping NaNs for the SDT's internal masking)
    x_tr_tensor = torch.tensor(x_train, dtype=torch.float32)
    x_vl_tensor = torch.tensor(x_val, dtype=torch.float32)

    train_ds = TensorDataset(x_tr_tensor, y_tr_tensor)
    val_ds = TensorDataset(x_vl_tensor, y_vl_tensor)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = SDT(
        input_dim=x_train.shape[1],
        output_dim=1,
        depth=depth,
        lamda=lamda,
        use_cuda=(device == "cuda")
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
    criterion = nn.BCEWithLogitsLoss()

    best_auprc = 0.0
    best_auroc = 0.0
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_state = None

    print(f"Training SDT student (depth={depth}, lamda={lamda})...")
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()

            logits, penalty = model(xb, is_training_data=True)
            loss = criterion(logits, yb) + penalty

            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)

        avg_train_loss = running_loss / len(x_train)

        # Validation phase
        model.eval()
        val_probs, val_labels = [], []
        val_running_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits, penalty = model(xb, is_training_data=True)
                loss = criterion(logits, yb) + penalty
                val_running_loss += loss.item() * xb.size(0)
                
                # Sigmoid to convert logits to probabilities
                probs = torch.sigmoid(logits)
                val_probs.append(probs.cpu())
                val_labels.append(yb.cpu())

        avg_val_loss = val_running_loss / len(x_val)
        val_probs_cat = torch.cat(val_probs).numpy().flatten()
        val_labels_cat = torch.cat(val_labels).numpy().flatten()

        epoch_auroc = roc_auc_score(val_labels_cat, val_probs_cat)
        epoch_auprc = average_precision_score(val_labels_cat, val_probs_cat)

        scheduler.step(epoch_auprc)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch_auprc > best_auprc:
            best_auprc = epoch_auprc
            best_auroc = epoch_auroc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        print(
            f"  Epoch [{epoch:3d}/{num_epochs}]  "
            f"Train Loss: {avg_train_loss:.6f}  Val Loss: {avg_val_loss:.6f}  "
            f"Val AUROC: {epoch_auroc:.4f}  Val AUPRC: {epoch_auprc:.4f}"
            + ("  *" if epochs_no_improve == 0 else "")
        )

        if epochs_no_improve >= patience:
            print(f"  Early stopping at epoch {epoch}.")
            break

    model.load_state_dict(best_state)
    os.makedirs('models/sdt', exist_ok=True)
    torch.save(model.state_dict(), f'models/sdt/sdt_fold_{fold_idx}.pt')

    return best_auroc, best_auprc


def main():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Use sparse=True to let the SDT leverage its built-in binary missingness mask
    input_df, labels_df = build_dataset(sparse=True)

    n_patients = input_df.index.get_level_values('subject_id').nunique()
    n_predictors = input_df.index.get_level_values('measurement_name').nunique()
    print(f"Patients: {n_patients}, Predictors: {n_predictors}, Hours: {input_df.shape[1]}")

    X = input_df.values.reshape((n_patients, n_predictors, 48)).astype(np.float32)
    X_flat = X.reshape(n_patients, -1)   # (N, 22*48) = (N, 1056)
    y = labels_df['mort_icu'].values.astype(np.float32)
    print(f"X_flat shape: {X_flat.shape}, y shape: {y.shape}, mortality rate: {y.mean():.3f}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_idx = list(cv.split(X=np.zeros(n_patients), y=y))

    all_aurocs, all_auprcs = [], []

    for i, (train_idx, val_idx) in enumerate(cv_idx):
        print(f"\n--- Fold {i} Standalone SDT ---")

        x_tr, x_vl = X_flat[train_idx], X_flat[val_idx]
        y_tr, y_vl = y[train_idx], y[val_idx]

        # Per-fold standardization (handling NaNs gracefully using nanmean / nanstd)
        mean = np.nanmean(x_tr, axis=0, keepdims=True)
        std = np.nanstd(x_tr, axis=0, keepdims=True)
        
        # Standardize training and validation folds using training fold statistics
        x_tr_norm = (x_tr - mean) / (std + 1e-8)
        x_vl_norm = (x_vl - mean) / (std + 1e-8)

        fold_auroc, fold_auprc = train_sdt_fold(
            fold_idx=i,
            x_train=x_tr_norm,
            x_val=x_vl_norm,
            y_train=y_tr,
            y_val=y_vl,
            device=device,
            batch_size=128,
            num_epochs=100,
            lr=1e-3,
            depth=5,
            lamda=1e-3,
            patience=10,
        )

        print(f"Fold {i} Final Result -> AUROC: {fold_auroc:.4f}, AUPRC: {fold_auprc:.4f}")
        all_aurocs.append(fold_auroc)
        all_auprcs.append(fold_auprc)

    print(f"\nStandalone SDT Complete.")
    print(f"Mean AUROC: {np.mean(all_aurocs):.4f} ± {np.std(all_aurocs):.4f}")
    print(f"Mean AUPRC: {np.mean(all_auprcs):.4f} ± {np.std(all_auprcs):.4f}")


if __name__ == '__main__':
    main()