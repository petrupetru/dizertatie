"""
dGBDT (distillable Gradient Boosted Decision Trees) for MIMIC-III.

Input: flattened time-series — 22 features × 48 hours = 1056 dimensions.
Training: standalone (no distillation), directly against ground-truth labels
using the stage-wise residual boosting loss from Algorithm 2 of the TNT paper.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, roc_auc_score

from SDT import SDT
from utils import seed_worker, set_seed
import os

# 22 clinical features × 48 hourly time-steps
MIMIC_INPUT_DIM = 22 * 48  # 1056


class dGBDT(nn.Module):
    """Distillable Gradient Boosted Decision Trees with SDT base learners.

    Additive ensemble:  F(x) = sigmoid( sum_m h_m(x) )
    where h_m is a Soft Decision Tree whose raw logit output is summed
    stage-wise (Algorithm 2, TNT paper).  No learned tree weights — each
    tree directly corrects the residual of the previous cumulative logit.

    input_dim should be the flattened feature size (default: 22 * 48 = 1056).
    """

    def __init__(
        self,
        input_dim=MIMIC_INPUT_DIM,
        num_trees=5,
        tree_depth=5,
        lamda=1e-3,
        use_cuda=False,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_trees = num_trees
        self.tree_depth = tree_depth
        self.lamda = lamda
        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.trees = nn.ModuleList(
            [
                SDT(
                    input_dim=self.input_dim,
                    output_dim=1,
                    depth=self.tree_depth,
                    lamda=self.lamda,
                    use_cuda=use_cuda,
                )
                for _ in range(self.num_trees)
            ]
        )

    def forward(self, x, is_training_data=False):
        tree_outputs = []
        penalties = []

        for tree in self.trees:
            if is_training_data:
                out, pen = tree(x, is_training_data=True)
                penalties.append(pen)
            else:
                out = tree(x, is_training_data=False)
            tree_outputs.append(out)

        # Sum raw logits from each tree, then apply sigmoid for probability.
        # This allows later trees to subtract from the cumulative prediction
        # (fix the positivity trap from beta * sigmoid(logit) formulation).
        ensemble_logit = sum(tree_outputs)          # unbounded real value
        ensemble_prob  = torch.sigmoid(ensemble_logit)  # ∈ (0, 1)

        if is_training_data:
            return tree_outputs, penalties, ensemble_prob
        return ensemble_prob


def train_dgbdt_fold(
    x_train: np.ndarray,
    x_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    device: str = "cpu",
    batch_size: int = 512,
    num_epochs: int = 100,
    lr: float = 1e-3,
    seed: int = 42,
    num_trees: int = 5,
    depth: int = 5,
    lamda: float = 1e-3,
    verbose: bool = True,
    fold_idx: int = 0,
    save_dir: str = "models/dgbdt",
) -> dict:
    """Train dGBDT directly against ground-truth labels (no distillation).

    x_train / x_val : float32 arrays of shape (N, 1056) — already flattened and
                       normalised by the caller.
    y_train / y_val : float32 binary label arrays of shape (N,).

    Returns a dict with keys: model, auroc, auprc, train_losses.
    """
    x_tr = torch.tensor(x_train, dtype=torch.float32)
    x_vl = torch.tensor(x_val,   dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    y_vl = torch.tensor(y_val,   dtype=torch.float32).unsqueeze(1)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        TensorDataset(x_tr, y_tr),
        batch_size=batch_size,
        shuffle=True,
        generator=g,
        worker_init_fn=seed_worker,
    )
    val_loader = DataLoader(
        TensorDataset(x_vl, y_vl),
        batch_size=batch_size,
        shuffle=False,
    )

    model = dGBDT(
        input_dim=x_train.shape[1],
        num_trees=num_trees,
        tree_depth=depth,
        lamda=lamda,
        use_cuda=(device == "cuda"),
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    epoch_losses = []
    best_val_loss = float("inf")
    best_auprc = 0.0
    best_state = None
    patience = 10
    epochs_no_improve = 0
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            tree_outputs, penalties, _ = model(xb, is_training_data=True)

            # Stage-wise residual boosting (Algorithm 2, TNT paper).
            # pred_so_far accumulates raw logits; residual is computed in
            # probability space: target_prob - sigmoid(cumulative_logit).
            # This allows later trees to output negative values and correct
            # over-predictions (fixes the positivity trap).
            pred_so_far = torch.zeros_like(yb)  # raw logit accumulator
            total_loss = torch.tensor(0.0, device=device)

            for m in range(model.num_trees):
                residual = yb - torch.sigmoid(pred_so_far.detach())
                total_loss = total_loss + mse(tree_outputs[m], residual) + penalties[m]
                pred_so_far = pred_so_far + tree_outputs[m].detach()

            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item() * xb.size(0)

        avg_train_loss = running_loss / len(train_loader.dataset)
        epoch_losses.append(avg_train_loss)

        # Validation loss
        model.eval()
        val_running_loss = 0.0
        val_probs, val_labels = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                tree_outputs_v, penalties_v, ensemble_prob_v = model(xb, is_training_data=True)
                
                val_probs.append(ensemble_prob_v.cpu())
                val_labels.append(yb.cpu())
                
                pred_so_far_v = torch.zeros_like(yb)
                val_loss = torch.tensor(0.0, device=device)
                for m in range(model.num_trees):
                    residual_v = yb - torch.sigmoid(pred_so_far_v)
                    val_loss = val_loss + mse(tree_outputs_v[m], residual_v) + penalties_v[m]
                    pred_so_far_v = pred_so_far_v + tree_outputs_v[m]
                val_running_loss += val_loss.item() * xb.size(0)
        model.train()

        avg_val_loss = val_running_loss / len(val_loader.dataset)

        val_probs = torch.cat(val_probs, dim=0).numpy().flatten()
        val_labels = torch.cat(val_labels, dim=0).numpy().flatten()
        epoch_auroc = roc_auc_score(val_labels, val_probs)
        epoch_auprc = average_precision_score(val_labels, val_probs)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch_auprc > best_auprc:
            best_auprc = epoch_auprc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if verbose:
            print(
                f"dGBDT Epoch [{epoch:3d}/{num_epochs}]  "
                f"Train Loss: {avg_train_loss:.6f}  Val Loss: {avg_val_loss:.6f}  "
                f"Val AUROC: {epoch_auroc:.4f}  Val AUPRC: {epoch_auprc:.4f}"
                + ("  *" if epochs_no_improve == 0 else "")
            )

        if epochs_no_improve >= patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch}.")
            break

    if best_state:
        model.load_state_dict(best_state)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_dir, f"dgbdt_standalone_fold_{fold_idx}.pt"))

    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            probs.append(model(xb.to(device)).cpu())
            labels.append(yb)

    probs  = torch.cat(probs,  dim=0).numpy().flatten()
    labels = torch.cat(labels, dim=0).numpy().flatten()

    auroc = roc_auc_score(labels, probs)
    auprc = average_precision_score(labels, probs)

    if verbose:
        print(f"dGBDT  AUROC: {auroc:.4f}  AUPRC: {auprc:.4f}")

    return {
        "model":        model,
        "auroc":        auroc,
        "auprc":        auprc,
        "train_losses": epoch_losses,
    }


def train_kfold_dgbdt(
    X_flat: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    num_epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 512,
    device: str = "cpu",
    num_trees: int = 10,
    depth: int = 4,
    lamda: float = 1e-3,
    seed: int = 42,
    verbose: bool = True,
    save_dir: str = "models/dgbdt",
) -> dict:
    """K-fold cross-validation for the standalone dGBDT on MIMIC-III.

    X_flat : float32 array of shape (N, 1056) — flattened, normalised input.
    y      : float32 binary label array of shape (N,).

    Returns a dict with per-fold results and mean AUROC / AUPRC.
    """
    set_seed(seed)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_flat, y)):
        print(f"\n--- Fold {fold + 1}/{n_splits} ---")
        x_tr, x_vl = X_flat[train_idx], X_flat[val_idx]
        y_tr, y_vl = y[train_idx],      y[val_idx]

        # Per-fold standardisation (mean/std from train split only)
        mean = x_tr.mean(axis=0, keepdims=True)
        std  = x_tr.std(axis=0,  keepdims=True)
        x_tr = np.nan_to_num((x_tr - mean) / (std + 1e-8)).astype(np.float32)
        x_vl = np.nan_to_num((x_vl - mean) / (std + 1e-8)).astype(np.float32)

        result = train_dgbdt_fold(
            x_train=x_tr,
            x_val=x_vl,
            y_train=y_tr,
            y_val=y_vl,
            device=device,
            batch_size=batch_size,
            num_epochs=num_epochs,
            lr=lr,
            seed=seed + fold,
            num_trees=num_trees,
            depth=depth,
            lamda=lamda,
            verbose=verbose,
            fold_idx=fold,
            save_dir=save_dir,
        )
        fold_results.append(result)

    aurocs = [r["auroc"] for r in fold_results]
    auprcs = [r["auprc"] for r in fold_results]
    print(
        f"\ndGBDT {n_splits}-fold  "
        f"AUROC: {np.mean(aurocs):.4f} ± {np.std(aurocs):.4f}  "
        f"AUPRC: {np.mean(auprcs):.4f} ± {np.std(auprcs):.4f}"
    )

    return {
        "folds":      fold_results,
        "mean_auroc": np.mean(aurocs),
        "std_auroc":  np.std(aurocs),
        "mean_auprc": np.mean(auprcs),
        "std_auprc":  np.std(auprcs),
    }