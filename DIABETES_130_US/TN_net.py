import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import numpy as np
import xgboost as xgb
from constants import SEED, FEATURE_NAMES, LABEL_NAME
from utils import set_seed, seed_worker, _assert_finite_tensor, make_run_key, try_load_results, save_results
from SDT import SDT
from tree_embeddings import train_xgb
from diabetes_dataset import TreeEmbedDataset


class TN_DNN(nn.Module):
    """
    input is Xe -> vector of leaf indexes
    """
    def __init__(self, num_trees, num_groups, max_leaves_per_tree, embedding_dim):
        super(TN_DNN, self).__init__()
        if num_trees % num_groups != 0:
            raise ValueError("num_trees must be divisible by num_groups")

        self.num_groups = num_groups
        self.trees_per_group = num_trees // num_groups
        self.max_leaves_per_tree = max_leaves_per_tree
        self.leaves_per_group = self.trees_per_group * max_leaves_per_tree
        self.register_buffer(
            "tree_offsets",
            torch.arange(self.trees_per_group, dtype=torch.long).unsqueeze(0) * self.max_leaves_per_tree,
        )

        self.group_embeddings = nn.ModuleList([
            nn.Embedding(self.leaves_per_group, embedding_dim) for _ in range(num_groups)
        ])
        group_input_dim = self.trees_per_group * embedding_dim

        g = torch.Generator()
        g.manual_seed(SEED)
        self.register_buffer("tree_perm", torch.randperm(num_trees, generator=g))

        self.fc1 = nn.Linear(group_input_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 16)
        self.fc4 = nn.Linear(16, 8)
        self.fc5 = nn.Linear(8, 8)
        self.fc6 = nn.Linear(8, 1)

    def forward(self, input, return_logits=False):
        batch_size = input.size(0)

        input = input[:, self.tree_perm]

        L_T_X = input.reshape(batch_size, self.num_groups, self.trees_per_group)

        group_dnn_outputs = []

        for i in range(self.num_groups):
            group_indices = L_T_X[:, i, :]
            group_indices = group_indices + self.tree_offsets

            group_embeddings = self.group_embeddings[i](group_indices)
            group_embeddings = group_embeddings.view(batch_size, -1)

            group_output = F.relu(self.fc1(group_embeddings))
            group_output = F.relu(self.fc2(group_output))
            group_output = F.relu(self.fc3(group_output))
            group_output = F.relu(self.fc4(group_output))
            group_output = F.relu(self.fc5(group_output))
            group_output = self.fc6(group_output)

            group_dnn_outputs.append(group_output)

        final_output = torch.cat(group_dnn_outputs, dim=1)
        logit = final_output.mean(dim=1, keepdim=True)

        if return_logits:
            return logit
        return torch.sigmoid(logit)


def train(model, train_loader, val_loader, num_epochs=50, lr=1e-3, device="cpu", verbose=True, patience=10):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(1, num_epochs + 1):
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

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)


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


def distill_sdt_fold(
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
    alpha=1.0,
    depth=5,
    lamda=1e-3,
    seed=None,
    verbose=True,
    save_path=None,
    patience=5,
):
    teacher_model.eval()
    with torch.no_grad():
        X_teacher_train = torch.tensor(leaf_indices_train, dtype=torch.long, device=device)
        X_teacher_val = torch.tensor(leaf_indices_val, dtype=torch.long, device=device)
        y_teacher_train = teacher_model(X_teacher_train).cpu()
        y_teacher_val = teacher_model(X_teacher_val).cpu()

    X_train_raw = torch.tensor(
        fold_train_df[FEATURE_NAMES].values, dtype=torch.float32
    )
    X_val_raw = torch.tensor(
        fold_val_df[FEATURE_NAMES].values, dtype=torch.float32
    )

    y_train_true = torch.tensor(labels_train, dtype=torch.float32).unsqueeze(1)
    y_val_true = torch.tensor(labels_val, dtype=torch.float32).unsqueeze(1)

    sdt_train_ds = TensorDataset(X_train_raw, y_train_true, y_teacher_train)
    sdt_val_ds = TensorDataset(X_val_raw, y_val_true, y_teacher_val)

    g = None
    if seed is not None:
        g = torch.Generator()
        g.manual_seed(seed)

    sdt_train_loader = DataLoader(
        sdt_train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=g,
        worker_init_fn=seed_worker,
        pin_memory=(device != "cpu"),
    )
    sdt_val_loader = DataLoader(
        sdt_val_ds,
        batch_size=batch_size,
        shuffle=False,
        generator=g,
        worker_init_fn=seed_worker,
        pin_memory=(device != "cpu"),
    )

    sdt = SDT(
        input_dim=X_train_raw.shape[1],
        output_dim=1,
        depth=depth,
        lamda=lamda,
        use_cuda=(device == "cuda"),
    ).to(device)

    sdt_opt = optim.Adam(sdt.parameters(), lr=lr)
    mse = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(1, num_epochs + 1):
        sdt.train()
        train_running_loss = 0.0
        for xb, yb, ytb in sdt_train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            ytb = ytb.to(device)

            preds, penalty = sdt(xb, is_training_data=True)
            preds = torch.sigmoid(preds)

            _assert_finite_tensor("sdt_preds", preds)
            if not torch.isfinite(penalty).item():
                raise ValueError("sdt penalty is not finite")

            loss = alpha * mse(preds, ytb) + (1 - alpha) * mse(preds, yb) + penalty
            sdt_opt.zero_grad()
            loss.backward()
            sdt_opt.step()
            train_running_loss += loss.item() * xb.size(0)

        avg_train_loss = train_running_loss / len(sdt_train_loader.dataset)

        sdt.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for xb, yb, ytb in sdt_val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                ytb = ytb.to(device)
                preds_v, penalty_v = sdt(xb, is_training_data=True)
                preds_v = torch.sigmoid(preds_v)
                val_loss = alpha * mse(preds_v, ytb) + (1 - alpha) * mse(preds_v, yb) + penalty_v
                val_running_loss += val_loss.item() * xb.size(0)
        sdt.train()

        avg_val_loss = val_running_loss / len(sdt_val_loader.dataset)

        if verbose:
            print(f"SDT Epoch [{epoch:3d}/{num_epochs}] Train Loss: {avg_train_loss:.6f}  Val Loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {k: v.clone() for k, v in sdt.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        sdt.load_state_dict(best_state)

    sdt.eval()
    sdt_probs = []
    sdt_labels = []
    with torch.no_grad():
        for xb, yb, _ytb in sdt_val_loader:
            xb = xb.to(device)
            preds = torch.sigmoid(sdt(xb)).cpu()
            sdt_probs.append(preds)
            sdt_labels.append(yb)

    sdt_probs = torch.cat(sdt_probs).numpy().flatten()
    sdt_labels = torch.cat(sdt_labels).numpy().flatten()

    if not np.isfinite(sdt_probs).all():
        raise ValueError("sdt_probs contains non-finite values")
    if not np.isfinite(sdt_labels).all():
        raise ValueError("sdt_labels contains non-finite values")

    sdt_auroc = roc_auc_score(sdt_labels, sdt_probs)
    sdt_auprc = average_precision_score(sdt_labels, sdt_probs)

    if verbose:
        print(f"SDT AUROC: {sdt_auroc:.4f}  AUPRC: {sdt_auprc:.4f}")

    if save_path is not None:
        torch.save(sdt.state_dict(), save_path)

    return {"auroc": sdt_auroc, "auprc": sdt_auprc}


def train_kfold_TNT(df, model_class, n_splits=5, num_epochs=50, lr=1e-3, batch_size=32, device="cpu", verbose=True, distilation=False, save_dir=None, force_retrain=False):
    key = make_run_key(model="tnt_sdt", splits=n_splits, epochs=num_epochs, lr=lr, batch=batch_size, distil=int(distilation))
    if not force_retrain:
        cached = try_load_results(save_dir, key)
        if cached is not None:
            print(f"[TNT+SDT] Loaded cached results ({key})")
            return cached

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

        scaler = StandardScaler()
        fold_train_df[FEATURE_NAMES] = scaler.fit_transform(fold_train_df[FEATURE_NAMES])
        fold_val_df[FEATURE_NAMES]   = scaler.transform(fold_val_df[FEATURE_NAMES])

        xgb_model, num_trees = train_xgb(fold_train_df)
        dtrain = xgb.DMatrix(fold_train_df[FEATURE_NAMES].values)
        leaf_indices_train = xgb_model.predict(dtrain, pred_leaf=True)
        labels_train = fold_train_df[LABEL_NAME].values.astype(int)
        dval = xgb.DMatrix(fold_val_df[FEATURE_NAMES].values)
        leaf_indices_val = xgb_model.predict(dval, pred_leaf=True)
        labels_val = fold_val_df[LABEL_NAME].values.astype(int)

        train_dataset = TreeEmbedDataset(leaf_indices_train, labels_train)
        val_dataset   = TreeEmbedDataset(leaf_indices_val, labels_val)

        fold_seed = SEED + fold
        g = torch.Generator()
        g.manual_seed(fold_seed)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=g,
            worker_init_fn=seed_worker,
            pin_memory=(device != "cpu"),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            generator=g,
            worker_init_fn=seed_worker,
            pin_memory=(device != "cpu"),
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

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_dir, f"{key}__fold{fold}_teacher.pt"))

        metrics = test(model, val_loader, device=device, verbose=verbose)
        fold_aurocs.append(metrics["auroc"])
        fold_auprcs.append(metrics["auprc"])

        if verbose:
            print(f"Fold {fold} — AUROC: {metrics['auroc']:.4f}  AUPRC: {metrics['auprc']:.4f}")

        if distilation:
            sdt_seed = SEED + 1000 + fold
            sdt_save_path = os.path.join(save_dir, f"{key}__fold{fold}_sdt.pt") if save_dir is not None else None
            sdt_metrics = distill_sdt_fold(
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
                alpha=0.5,
                depth=6,
                lamda=1e-3,
                seed=sdt_seed,
                verbose=verbose,
                save_path=sdt_save_path,
            )
            fold_sdt_aurocs.append(sdt_metrics["auroc"])
            fold_sdt_auprcs.append(sdt_metrics["auprc"])

    mean_teacher_auroc = np.mean(fold_aurocs)
    std_teacher_auroc  = np.std(fold_aurocs)
    mean_teacher_auprc = np.mean(fold_auprcs)
    std_teacher_auprc  = np.std(fold_auprcs)

    if verbose:
        print(f"\n{'='*50}")
        print(f"  Cross-Validation Results ({n_splits} folds)")
        print(f"  AUROC: {mean_teacher_auroc:.4f} ± {std_teacher_auroc:.4f}")
        print(f"  AUPRC: {mean_teacher_auprc:.4f} ± {std_teacher_auprc:.4f}")
        print(f"{'='*50}")

    cv_results = {
        "fold_aurocs": fold_aurocs,
        "fold_auprcs": fold_auprcs,
        "mean_teacher_auroc": mean_teacher_auroc,
        "std_teacher_auroc":  std_teacher_auroc,
        "mean_teacher_auprc": mean_teacher_auprc,
        "std_teacher_auprc":  std_teacher_auprc,
    }

    if distilation:
        cv_results["fold_soft_tree_aurocs"] = fold_sdt_aurocs
        cv_results["fold_soft_tree_auprcs"] = fold_sdt_auprcs
        cv_results["mean_soft_tree_auroc"] = float(np.mean(fold_sdt_aurocs))
        cv_results["std_soft_tree_auroc"]  = float(np.std(fold_sdt_aurocs))
        cv_results["mean_soft_tree_auprc"] = float(np.mean(fold_sdt_auprcs))
        cv_results["std_soft_tree_auprc"]  = float(np.std(fold_sdt_auprcs))

    if save_dir is not None:
        save_results(save_dir, key, cv_results)

    return cv_results
