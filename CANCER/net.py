import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
import numpy as np
from constants import SEED, FEATURE_NAMES, LABEL_NAME
from utils import set_seed, seed_worker
from cancer_datasets import BreastCancerDataset


class DNN(nn.Module):

    def __init__(self):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(30, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 16)
        self.fc4 = nn.Linear(16, 8)
        self.fc5 = nn.Linear(8, 8)
        self.fc6 = nn.Linear(8, 1)

    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = torch.sigmoid(self.fc6(x))
        return x
    


def train(model, train_loader, val_loader, num_epochs=50, lr=1e-3, device="cpu", verbose=True):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, num_epochs + 1):
        # --- Training phase ---
        model.train()
        train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)

        # --- Validation phase ---
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)

        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)

        if verbose:
            print(
                f"Epoch [{epoch:3d}/{num_epochs}] "
                f"Train MSE Loss: {train_loss:.4f}  "
                f"Val MSE Loss: {val_loss:.4f}"
            )


def test(model, test_loader, device="cpu", verbose=True):
    model.to(device)
    model.eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            probs = model(X_batch).cpu()
            all_probs.append(probs)
            all_labels.append(y_batch)

    all_probs = torch.cat(all_probs).numpy().flatten()
    all_labels = torch.cat(all_labels).numpy().flatten()
    
    auroc = roc_auc_score(all_labels, all_probs)
    auprc = average_precision_score(all_labels, all_probs)

    if verbose:
        print(f"Test AUROC:    {auroc:.4f}")
        print(f"Test AUPRC:    {auprc:.4f}")

    return {"auroc": auroc, "auprc": auprc}

def train_kfold(df, model_class, n_splits=5, num_epochs=50, lr=1e-3, batch_size=32, device="cpu", verbose=True):
    """
    Performs k-fold cross-validation on the full dataset.

    For each fold:
      - Imputes NaNs using only the training fold's means (no data leakage)
      - Trains a fresh model on the training folds
      - Evaluates AUROC and AUPRC on the validation fold

    After all folds, prints mean ± std of each metric.

    Args:
        df:           Full DataFrame (all samples).
        model_class:  Model class to instantiate (e.g. DNN). Fresh instance per fold.
        n_splits:     Number of folds (default 5, giving 80/20 splits).
        num_epochs:   Epochs per fold.
        lr:           Learning rate.
        batch_size:   Batch size.
        device:       "cpu" or "cuda".
        verbose:      Whether to print training progress and results.
    Returns:
        cv_results:  Dict with per-fold metrics and their mean/std.
    """

    set_seed(SEED)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    labels = df[LABEL_NAME].values

    fold_aurocs = []
    fold_auprcs = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, labels), start=1):
        if verbose:
            print(f"\n{'='*50}")
            print(f"  Fold {fold}/{n_splits}")
            print(f"{'='*50}")

        fold_train_df = df.iloc[train_idx].copy()
        fold_val_df   = df.iloc[val_idx].copy()

        # Impute using training fold statistics only
        train_means = fold_train_df[FEATURE_NAMES].mean()
        fold_train_df[FEATURE_NAMES] = fold_train_df[FEATURE_NAMES].fillna(train_means)
        fold_val_df[FEATURE_NAMES]   = fold_val_df[FEATURE_NAMES].fillna(train_means)

        train_dataset = BreastCancerDataset(fold_train_df)
        val_dataset   = BreastCancerDataset(fold_val_df)

        fold_seed = SEED + fold
        g = torch.Generator()
        g.manual_seed(fold_seed)

        train_loader  = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=g,
            worker_init_fn=seed_worker,
        )
        val_loader    = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            generator=g,
            worker_init_fn=seed_worker,
        )

        model = model_class().to(device)
        train(model, train_loader, val_loader, num_epochs=num_epochs, lr=lr, device=device, verbose=verbose)

        metrics = test(model, val_loader, device=device, verbose=verbose)
        fold_aurocs.append(metrics["auroc"])
        fold_auprcs.append(metrics["auprc"])

        if verbose:
            print(f"Fold {fold} — AUROC: {metrics['auroc']:.4f}  AUPRC: {metrics['auprc']:.4f}")

    mean_auroc = np.mean(fold_aurocs)
    std_auroc  = np.std(fold_aurocs)
    mean_auprc = np.mean(fold_auprcs)
    std_auprc  = np.std(fold_auprcs)

    if verbose:
        print(f"\n{'='*50}")
        print(f"  Cross-Validation Results ({n_splits} folds)")
        print(f"  AUROC: {mean_auroc:.4f} ± {std_auroc:.4f}")
        print(f"  AUPRC: {mean_auprc:.4f} ± {std_auprc:.4f}")
        print(f"{'='*50}")

    cv_results = {
        "fold_aurocs": fold_aurocs,
        "fold_auprcs": fold_auprcs,
        "mean_auroc": mean_auroc,
        "std_auroc":  std_auroc,
        "mean_auprc": mean_auprc,
        "std_auprc":  std_auprc,
    }

    return cv_results


if __name__ == "__main__":
    dnn = DNN()
    print(dnn)