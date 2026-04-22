from sklearn.metrics import average_precision_score, roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.parametrize as parametrize
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
import numpy as np
import xgboost as xgb

from constants import SEED, FEATURE_NAMES, LABEL_NAME
from SDT import SDT
from utils import _assert_finite_tensor, seed_worker, set_seed
from datasets import TreeEmbedDataset
from tree_embeddings import train_xgb
from TN_net import train, test


class PositiveParam(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return torch.nn.functional.softplus(x) + self.eps


class dGBDT(nn.Module):
    """Distillable Gradient Boosted Decision Trees with SDT base learners.

    This implementation follows the paper's additive ensemble view:
    F(x) = sum_m beta_m * h_m(x)
    where h_m is an SDT and beta_m is a tree weight.
    """

    def __init__(
        self,
        input_dim,
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

        self.beta = nn.Parameter(torch.zeros(self.num_trees))
        parametrize.register_parametrization(self, "beta", PositiveParam(eps=1e-6))

    def forward(self, x, is_training_data=False, return_tree_outputs=False):
        tree_outputs = []
        penalties = []

        for tree in self.trees:
            if is_training_data:
                out, pen = tree(x, is_training_data=True)
                penalties.append(pen)
            else:
                out = tree(x, is_training_data=False)
            tree_outputs.append(out)

        beta = self.beta.view(-1, 1, 1)
        stacked = torch.stack(tree_outputs, dim=0)
        ensemble_raw = torch.sum(beta * stacked, dim=0)

        if is_training_data:
            return tree_outputs, penalties, ensemble_raw
        if return_tree_outputs:
            return ensemble_raw, tree_outputs
        return ensemble_raw



def distill_dgbdt_fold(
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
    alpha=0.5,
    temperature=1.0,
    num_trees=5,
    depth=5,
    lamda=1e-3,
    verbose=True,
):
    """Train dGBDT student from teacher soft labels for one CV fold.

    The per-tree boosting target uses negative gradient style residuals for
    the mixed distillation loss:
      L = alpha*T^2*MSE(O_s, O_t) + (1-alpha)*MSE(O_s, y)
    """

    teacher_model.eval()
    with torch.no_grad():
        x_teacher_train = torch.tensor(leaf_indices_train, dtype=torch.long, device=device)
        x_teacher_val = torch.tensor(leaf_indices_val, dtype=torch.long, device=device)
        y_teacher_train = teacher_model(x_teacher_train).cpu()
        y_teacher_val = teacher_model(x_teacher_val).cpu()

    _assert_finite_tensor("y_teacher_train", y_teacher_train)
    _assert_finite_tensor("y_teacher_val", y_teacher_val)

    x_train_raw = torch.tensor(fold_train_df[FEATURE_NAMES].values, dtype=torch.float32)
    x_val_raw = torch.tensor(fold_val_df[FEATURE_NAMES].values, dtype=torch.float32)
    y_train_true = torch.tensor(labels_train, dtype=torch.float32).unsqueeze(1)
    y_val_true = torch.tensor(labels_val, dtype=torch.float32).unsqueeze(1)

    _assert_finite_tensor("x_train_raw", x_train_raw)
    _assert_finite_tensor("x_val_raw", x_val_raw)

    train_ds = TensorDataset(x_train_raw, y_train_true, y_teacher_train)
    val_ds = TensorDataset(x_val_raw, y_val_true)

    g = None
    if seed is not None:
        g = torch.Generator()
        g.manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=g,
        worker_init_fn=seed_worker,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        generator=g,
        worker_init_fn=seed_worker,
    )

    dgbdt = dGBDT(
        input_dim=x_train_raw.shape[1],
        num_trees=num_trees,
        tree_depth=depth,
        lamda=lamda,
        use_cuda=(device == "cuda"),
    ).to(device)

    optimizer = optim.Adam(dgbdt.parameters(), lr=lr)
    mse = nn.MSELoss()

    w_teacher = alpha * (temperature ** 2)
    w_true = 1.0 - alpha

    epoch_losses = []
    for epoch in range(1, num_epochs + 1):
        dgbdt.train()
        running_loss = 0.0

        for xb, yb, ytb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            ytb = ytb.to(device)

            optimizer.zero_grad()
            tree_outputs, penalties, _ = dgbdt(xb, is_training_data=True)

            # Stage-wise residual fitting over the additive ensemble.
            pred_so_far = torch.zeros_like(yb)
            total_tree_loss = torch.tensor(0.0, device=device)
            beta = dgbdt.beta

            for m in range(dgbdt.num_trees):
                residual_target = w_teacher * (ytb - pred_so_far.detach()) + w_true * (yb - pred_so_far.detach())
                tree_loss = mse(tree_outputs[m], residual_target) + penalties[m]
                total_tree_loss = total_tree_loss + tree_loss
                pred_so_far = pred_so_far + beta[m] * tree_outputs[m]

            total_tree_loss.backward()
            optimizer.step()
            running_loss += total_tree_loss.item() * xb.size(0)

        avg_train_loss = running_loss / len(train_loader.dataset)
        epoch_losses.append(avg_train_loss)
        if verbose:
            print(f"dGBDT Epoch [{epoch:3d}/{num_epochs}] Train Loss: {avg_train_loss:.6f}")

    dgbdt.eval()
    probs = []
    labels = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            pred_raw = dgbdt(xb)
            pred_prob = torch.sigmoid(pred_raw).cpu()
            probs.append(pred_prob)
            labels.append(yb)

    probs = torch.cat(probs, dim=0).numpy().flatten()
    labels = torch.cat(labels, dim=0).numpy().flatten()

    if not np.isfinite(probs).all():
        raise ValueError("dGBDT probabilities contain non-finite values")
    if not np.isfinite(labels).all():
        raise ValueError("dGBDT labels contain non-finite values")

    auroc = roc_auc_score(labels, probs)
    auprc = average_precision_score(labels, probs)

    if verbose:
        print(f"dGBDT AUROC: {auroc:.4f}  AUPRC: {auprc:.4f}")

    return {
        "model": dgbdt,
        "auroc": auroc,
        "auprc": auprc,
        "train_losses": epoch_losses,
        "beta": dgbdt.beta.detach().cpu().numpy(),
    }


def train_kfold_tnt_dgbdt(
    df,
    model_class,
    n_splits=5,
    num_epochs=50,
    lr=1e-3,
    batch_size=32,
    device="cpu",
    alpha=0.5,
    temperature=1.0,
    dgbdt_num_trees=5,
    dgbdt_depth=5,
    dgbdt_lamda=1e-3,
    verbose=True,
):
    """Run TNT with TN-DNN teacher and dGBDT student under k-fold CV."""
    set_seed(SEED)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    labels = df[LABEL_NAME].values

    fold_teacher_aurocs = []
    fold_teacher_auprcs = []
    fold_dgbdt_aurocs = []
    fold_dgbdt_auprcs = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, labels), start=1):
        if verbose:
            print(f"\n{'='*50}")
            print(f"  Fold {fold}/{n_splits}")
            print(f"{'='*50}")

        fold_train_df = df.iloc[train_idx].copy()
        fold_val_df = df.iloc[val_idx].copy()

        xgb_model, num_trees = train_xgb(fold_train_df)
        dtrain = xgb.DMatrix(fold_train_df[FEATURE_NAMES].values)
        dval = xgb.DMatrix(fold_val_df[FEATURE_NAMES].values)

        leaf_indices_train = xgb_model.predict(dtrain, pred_leaf=True)
        leaf_indices_val = xgb_model.predict(dval, pred_leaf=True)
        labels_train = np.where(fold_train_df[LABEL_NAME].values == "M", 1, 0)
        labels_val = np.where(fold_val_df[LABEL_NAME].values == "M", 1, 0)

        train_dataset = TreeEmbedDataset(leaf_indices_train, labels_train)
        val_dataset = TreeEmbedDataset(leaf_indices_val, labels_val)

        fold_seed = SEED + fold
        g = torch.Generator()
        g.manual_seed(fold_seed)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=g,
            worker_init_fn=seed_worker,
        )
        val_loader = DataLoader(
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

        teacher_model = model_class(
            num_trees=num_trees,
            num_groups=num_groups,
            max_leaves_per_tree=max_leaves_per_tree,
            embedding_dim=8,
        ).to(device)

        train(
            teacher_model,
            train_loader,
            val_loader,
            num_epochs=num_epochs,
            lr=lr,
            device=device,
            verbose=verbose,
        )

        teacher_metrics = test(teacher_model, val_loader, device=device, verbose=verbose)
        fold_teacher_aurocs.append(teacher_metrics["auroc"])
        fold_teacher_auprcs.append(teacher_metrics["auprc"])

        dgbdt_seed = SEED + 1000 + fold
        dgbdt_metrics = distill_dgbdt_fold(
            teacher_model=teacher_model,
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
            seed=dgbdt_seed,
            alpha=alpha,
            temperature=temperature,
            num_trees=dgbdt_num_trees,
            depth=dgbdt_depth,
            lamda=dgbdt_lamda,
            verbose=verbose,
        )
        fold_dgbdt_aurocs.append(dgbdt_metrics["auroc"])
        fold_dgbdt_auprcs.append(dgbdt_metrics["auprc"])

        if verbose:
            print(
                f"Fold {fold} - Teacher AUROC: {teacher_metrics['auroc']:.4f} AUPRC: {teacher_metrics['auprc']:.4f}"
            )
            print(
                f"Fold {fold} - dGBDT  AUROC: {dgbdt_metrics['auroc']:.4f} AUPRC: {dgbdt_metrics['auprc']:.4f}"
            )

    results = {
        "fold_teacher_aurocs": fold_teacher_aurocs,
        "fold_teacher_auprcs": fold_teacher_auprcs,
        "mean_teacher_auroc": float(np.mean(fold_teacher_aurocs)),
        "std_teacher_auroc": float(np.std(fold_teacher_aurocs)),
        "mean_teacher_auprc": float(np.mean(fold_teacher_auprcs)),
        "std_teacher_auprc": float(np.std(fold_teacher_auprcs)),
        "fold_dgbdt_aurocs": fold_dgbdt_aurocs,
        "fold_dgbdt_auprcs": fold_dgbdt_auprcs,
        "mean_dgbdt_auroc": float(np.mean(fold_dgbdt_aurocs)),
        "std_dgbdt_auroc": float(np.std(fold_dgbdt_aurocs)),
        "mean_dgbdt_auprc": float(np.mean(fold_dgbdt_auprcs)),
        "std_dgbdt_auprc": float(np.std(fold_dgbdt_auprcs)),
    }

    if verbose:
        print(f"\n{'='*50}")
        print(f"Teacher AUROC: {results['mean_teacher_auroc']:.4f} +/- {results['std_teacher_auroc']:.4f}")
        print(f"Teacher AUPRC: {results['mean_teacher_auprc']:.4f} +/- {results['std_teacher_auprc']:.4f}")
        print(f"dGBDT  AUROC: {results['mean_dgbdt_auroc']:.4f} +/- {results['std_dgbdt_auroc']:.4f}")
        print(f"dGBDT  AUPRC: {results['mean_dgbdt_auprc']:.4f} +/- {results['std_dgbdt_auprc']:.4f}")
        print(f"{'='*50}")

    return results
