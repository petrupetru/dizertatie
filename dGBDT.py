from sklearn.metrics import average_precision_score, roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from constants import SEED, FEATURE_NAMES, LABEL_NAME
from SDT import SDT
from utils import _assert_finite_tensor
from utils import set_seed, seed_worker
from sklearn.model_selection import StratifiedKFold
import numpy as np
from datasets import TreeEmbedDataset
from tree_embeddings import train_xgb
import xgboost as xgb

from TN_net import TN_DNN, train, test
import torch.nn.utils.parametrize as parametrize

class PositiveParam(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return torch.nn.functional.softplus(x) + self.eps

class dGBDT(nn.Module):
    def __init__(self, input_dim, num_trees=5, tree_depth=3, lamda=1e-3, use_cuda=False):
        super(dGBDT, self).__init__()

        self.input_dim = input_dim
        self.num_trees = num_trees
        self.tree_depth = tree_depth
        self.lamda = lamda
        self.device = torch.device("cuda" if use_cuda else "cpu")

        # Initialize a list of SDT modules to represent the trees in the ensemble.
        self.trees = nn.ModuleList([
            SDT(input_dim=self.input_dim, output_dim=1, depth=self.tree_depth, lamda=self.lamda, use_cuda=use_cuda)
            for _ in range(self.num_trees)
        ])

        self.beta = self.beta(torch.zeros(self.num_trees))
        parametrize.register_parametrization(self, "beta", PositiveParam(eps=1e-6))


    def forward(self, X, is_training_data=False, return_tree_outputs=False):
        # Forward the input through each tree and sum their outputs.
        tree_outputs = []
        penalties = []
        for tree in self.trees:
            if is_training_data:
                output, penalty = tree(X, is_training_data=True)
                penalties.append(penalty)
            else:
                output = tree(X, is_training_data=False)
            tree_outputs.append(output)

        beta = self.beta.view(-1, 1, 1)
        stacked = torch.stack(tree_outputs, dim=0)
        ensemble_raw = torch.sum(beta * stacked, dim=0)

   
        if is_training_data:
            return tree_outputs, penalties, ensemble_raw
        if return_tree_outputs:
            return tree_outputs, ensemble_raw
        return ensemble_raw
    
def distill_dGBDT_fold(
    dgbt_model,
    teacher_model,
    leaf_indices_train,
    leaf_indices_val,
    fold_train_df,
    fold_val_df,
    labels_train,
    labels_val,
    device,
    batch_size,
    num_epochs,
    lr,
    seed,
    alpha=1.0,
    depth=5,
    lamda=1e-3,
    verbose=True,
):
    teacher_model.eval()
    with torch.no_grad():
        X_teacher_train = torch.tensor(leaf_indices_train, dtype=torch.long, device=device)
        X_teacher_val = torch.tensor(leaf_indices_val, dtype=torch.long, device=device)
        y_teacher_train = teacher_model(X_teacher_train).cpu()
        y_teacher_val = teacher_model(X_teacher_val).cpu()

    _assert_finite_tensor("y_teacher_train", y_teacher_train)
    _assert_finite_tensor("y_teacher_val", y_teacher_val)

    X_train_raw = torch.tensor(
        fold_train_df[FEATURE_NAMES].values, dtype=torch.float32
    )
    X_val_raw = torch.tensor(
        fold_val_df[FEATURE_NAMES].values, dtype=torch.float32
    )

    _assert_finite_tensor("X_train_raw", X_train_raw)
    _assert_finite_tensor("X_val_raw", X_val_raw)
    y_train_true = torch.tensor(labels_train, dtype=torch.float32).unsqueeze(1)
    y_val_true = torch.tensor(labels_val, dtype=torch.float32).unsqueeze(1)

    dgbdt_train_ds = TensorDataset(X_train_raw, y_train_true, y_teacher_train)
    dgbdt_val_ds = TensorDataset(X_val_raw, y_val_true)

    g = None
    if seed is not None:
        g = torch.Generator()
        g.manual_seed(seed)

    train_loader = DataLoader(
        dgbdt_train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=g,
        worker_init_fn=seed_worker,
    )
    val_loader = DataLoader(
        dgbdt_val_ds,
        batch_size=batch_size,
        shuffle=False,
        generator=g,
        worker_init_fn=seed_worker,
    )
    
    optimizer = optim.Adam(dgbt_model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(1, num_epochs + 1):
        # --- Training phase ---
        dgbt_model.train()
        train_loss = 0.0

        cumulative_tree_loss = 0.0

        for X_batch, y_batch, y_t_batch in train_loader:
            X_batch, y_batch, y_t_batch = X_batch.to(device), y_batch.to(device), y_t_batch.to(device)
            optimizer.zero_grad()
            tree_outputs, penalties, _ = dgbt_model(X_batch, is_training_data=True)

            # cumulative output s
            s = 0
            for m in range(dgbt_model.num_trees):
                if m != 0:
                    s += tree_outputs[m-1]
                residual = alpha * criterion(s, y_t_batch) + (1 - alpha) * criterion(s, y_batch)
                tree_loss = criterion(residual, tree_outputs[m]) + penalties[m]
                cumulative_tree_loss += tree_loss
        
            cumulative_tree_loss.backward()
            optimizer.step()
            train_loss += cumulative_tree_loss.item()
        avg_train_loss = train_loss / len(train_loader)

        # --- Validation phase ---
        dgbt_model.eval()

        with torch.no_grad():
            dgbt_probs = []
            dgbt_labels = []
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                tree_outputs = dgbt_model(X_batch, is_training_data=False)
                s = sum(tree_outputs)
                dgbt_probs.append(s)
                dgbt_labels.append(y_batch)

            dgbt_probs = torch.cat(dgbt_probs, dim=0).numpy().flatten()
            dgbt_labels = torch.cat(dgbt_labels, dim=0).numpy().flatten()

            dgbt_auroc = roc_auc_score(dgbt_labels, dgbt_probs)
            dgbt_auprc = average_precision_score(dgbt_labels, dgbt_probs)

            if verbose:
                print(f"dGBDT AUROC: {dgbt_auroc:.4f}  AUPRC: {dgbt_auprc:.4f}")
            return {"auroc": dgbt_auroc, "auprc": dgbt_auprc}
        

def train_kfold_TNT(df, model_class, n_splits=5, num_epochs=50, lr=1e-3, batch_size=32, device="cpu", verbose=True, distilation=False):
    set_seed(SEED)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    labels = df[LABEL_NAME].values

    fold_aurocs = []
    fold_auprcs = []
    fold_sdt_aurocs = []
    fold_sdt_auprcs = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, labels), start=1):
        if verbose:
            print(f"\n{'='*50}")
            print(f"  Fold {fold}/{n_splits}")
            print(f"{'='*50}")

        fold_train_df = df.iloc[train_idx].copy()
        fold_val_df   = df.iloc[val_idx].copy()

        # Convert to tree embeddings
        xgb_model, num_trees = train_xgb(fold_train_df)
        X = fold_train_df[FEATURE_NAMES].values
        dtrain = xgb.DMatrix(X)
        leaf_indices_train = xgb_model.predict(dtrain, pred_leaf=True)
        labels_train = np.where(fold_train_df[LABEL_NAME].values == 'M', 1, 0)
        dval = xgb.DMatrix(fold_val_df[FEATURE_NAMES].values)
        leaf_indices_val = xgb_model.predict(dval, pred_leaf=True)
        labels_val = np.where(fold_val_df[LABEL_NAME].values == 'M', 1, 0)
        
        train_dataset = TreeEmbedDataset(leaf_indices_train, labels_train)
        val_dataset   = TreeEmbedDataset(leaf_indices_val, labels_val)

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

        tree_df = xgb_model.trees_to_dataframe()
        leaf_nodes = tree_df[tree_df["Feature"] == "Leaf"]
        max_leaf_index = int(leaf_nodes["Node"].max()) if not leaf_nodes.empty else 0
        num_groups = 10
        max_leaves_per_tree = max(1, max_leaf_index + 1)

        model = model_class(
            num_trees=num_trees,
            num_groups=num_groups,
            max_leaves_per_tree=max_leaves_per_tree,
            embedding_dim=8,
        ).to(device)
        train(model, train_loader, val_loader, num_epochs=num_epochs, lr=lr, device=device, verbose=verbose)

        metrics = test(model, val_loader, device=device, verbose=verbose)
        fold_aurocs.append(metrics["auroc"])
        fold_auprcs.append(metrics["auprc"])

        if verbose:
            print(f"Fold {fold} — AUROC: {metrics['auroc']:.4f}  AUPRC: {metrics['auprc']:.4f}")
        if distilation:
            sdt_seed = SEED + 1000 + fold
            sdt_metrics = distill_dGBDT_fold(
                teacher_model=model,
                leaf_indices_train=leaf_indices_train,
                leaf_indices_val=leaf_indices_val,
                fold_train_df=fold_train_df,
                fold_val_df=fold_val_df,
                labels_train=labels_train,
                labels_val=labels_val,
                device=device,
                batch_size=batch_size,
                num_epochs=num_epochs,
                lr=lr,
                alpha=1.0,
                depth=5,
                lamda=1e-3,
                seed=sdt_seed,
                verbose=verbose,
            )
            fold_sdt_aurocs.append(sdt_metrics["auroc"])
            fold_sdt_auprcs.append(sdt_metrics["auprc"])

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

    if distilation:
        cv_results["fold_sdt_aurocs"] = fold_sdt_aurocs
        cv_results["fold_sdt_auprcs"] = fold_sdt_auprcs
        cv_results["mean_sdt_auroc"] = float(np.mean(fold_sdt_aurocs))
        cv_results["std_sdt_auroc"] = float(np.std(fold_sdt_aurocs))
        cv_results["mean_sdt_auprc"] = float(np.mean(fold_sdt_auprcs))
        cv_results["std_sdt_auprc"] = float(np.std(fold_sdt_auprcs))

    return cv_results


