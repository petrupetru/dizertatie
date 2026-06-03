"""
Visualize internal node weights of a distilled dGBDT as a feature heatmap.

For normal distillation  (models/distill)   : 22 × 48 (feature × hour)
For aggregated distillation (models/distil_agg): 22 × 4  (feature × stat)
  - static features are broadcast (same weight across all 4 columns)
  - temporal features use their [min, max, mean, var] weights per column

Usage:
  python visualize_node.py --model distill    --fold 0 --tree 0 --node 0
  python visualize_node.py --model distil_agg --fold 0 --tree 2 --node 5
"""

import argparse
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

# ensure imports resolve regardless of cwd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dGBT import dGBDT

# ── feature metadata ──────────────────────────────────────────────────────────

FEATURE_NAMES = [
    'AGE', 'AIDS', 'BICARBONATE', 'BILIRUBIN', 'BUN', 'DIASTOLIC BP',
    'ELECTIVE', 'FiO2', 'GCSEyes', 'GCSMotor', 'GCSVerbal', 'HEART RATE',
    'LYMPHOMA', 'METASTATIC_CANCER', 'PO2', 'POTASSIUM', 'SODIUM',
    'SURGICAL', 'SYSTOLIC BP', 'TEMPERATURE', 'URINE OUTPUT', 'WBC',
]

STATIC_SET = {'AGE', 'AIDS', 'ELECTIVE', 'LYMPHOMA', 'METASTATIC_CANCER', 'SURGICAL'}

# positions of static / temporal features in the sorted full-feature list
static_indices   = [i for i, n in enumerate(FEATURE_NAMES) if n in STATIC_SET]
temporal_indices = [i for i, n in enumerate(FEATURE_NAMES) if n not in STATIC_SET]

N_FEATURES = 22
N_HOURS    = 48
N_STATIC   = len(static_indices)    # 6
N_TEMPORAL = len(temporal_indices)  # 16
TREE_DEPTH = 5


# ── model helpers ─────────────────────────────────────────────────────────────

def load_model(model_type: str, fold_idx: int) -> dGBDT:
    if model_type == 'distill':
        input_dim = N_FEATURES * N_HOURS
        path = f'models/distill/dgbdt_student_fold_{fold_idx}.pt'
    else:
        input_dim = N_STATIC + N_TEMPORAL * 4
        path = f'models/distil_agg/dgbdt_student_fold_{fold_idx}.pt'

    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    model = dGBDT(input_dim=input_dim, num_trees=5, tree_depth=TREE_DEPTH,
                  lamda=1e-3, use_cuda=False)
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    return model


def extract_node_weights(model: dGBDT, tree_idx: int, node_idx: int,
                         input_dim: int) -> np.ndarray:
    """
    Return the feature-weight slice for one internal node.

    The SDT augmented input layout (after _data_augment) is:
      [bias(1), features(input_dim), missing_mask(input_dim)]
    inner_nodes[0].weight shape: (n_internal_nodes, 2*input_dim + 1)
    """
    sdt = model.trees[tree_idx]
    w = sdt.inner_nodes[0].weight[node_idx].detach().cpu().numpy()
    # index 0 = bias, indices 1..input_dim = feature weights
    return w[1: input_dim + 1]


# ── heatmap builders ──────────────────────────────────────────────────────────

def build_heatmap_temporal(feature_w: np.ndarray) -> np.ndarray:
    """22 × 48  (feature axis first, matches X_raw layout)."""
    return feature_w.reshape(N_FEATURES, N_HOURS)


def build_heatmap_agg(feature_w: np.ndarray) -> np.ndarray:
    """
    22 × 4  columns = [min, max, mean, var].

    Aggregated weight vector layout (mirrors aggregate_features):
      feature_w[:N_STATIC]                          → static weights
      feature_w[N_STATIC : N_STATIC+N_TEMPORAL]     → temporal min
      feature_w[N_STATIC+N_TEMPORAL : +2*N_TEMPORAL]→ temporal max
      feature_w[...+2*N_TEMPORAL : +3*N_TEMPORAL]   → temporal mean
      feature_w[...+3*N_TEMPORAL : +4*N_TEMPORAL]   → temporal var
    """
    static_w       = feature_w[:N_STATIC]
    temporal_min   = feature_w[N_STATIC                  : N_STATIC +   N_TEMPORAL]
    temporal_max   = feature_w[N_STATIC +   N_TEMPORAL   : N_STATIC + 2*N_TEMPORAL]
    temporal_mean  = feature_w[N_STATIC + 2*N_TEMPORAL   : N_STATIC + 3*N_TEMPORAL]
    temporal_var   = feature_w[N_STATIC + 3*N_TEMPORAL   : N_STATIC + 4*N_TEMPORAL]

    heatmap = np.zeros((N_FEATURES, 4))

    # static rows: broadcast the single learned weight across all 4 columns
    for s_rank, feat_idx in enumerate(static_indices):
        heatmap[feat_idx, :] = static_w[s_rank]

    # temporal rows: one column per aggregate statistic
    for t_rank, feat_idx in enumerate(temporal_indices):
        heatmap[feat_idx, 0] = temporal_min[t_rank]
        heatmap[feat_idx, 1] = temporal_max[t_rank]
        heatmap[feat_idx, 2] = temporal_mean[t_rank]
        heatmap[feat_idx, 3] = temporal_var[t_rank]

    return heatmap


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_heatmap(heatmap: np.ndarray, model_type: str,
                 fold_idx: int, tree_idx: int, node_idx: int) -> None:
    if model_type == 'distill':
        col_labels = [str(h) for h in range(N_HOURS)]
        fig_w = 16
    else:
        col_labels = ['min', 'max', 'mean', 'var']
        fig_w = 5

    fig, ax = plt.subplots(figsize=(fig_w, 7))

    vmax = np.abs(heatmap).max() or 1.0
    im = ax.imshow(heatmap, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)

    ax.set_yticks(range(N_FEATURES))
    ax.set_yticklabels(FEATURE_NAMES, fontsize=9)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels,
                       fontsize=8,
                       rotation=90 if model_type == 'distill' else 0)

    if model_type == 'distill':
        ax.set_xlabel('Hour')
    else:
        ax.set_xlabel('Aggregate statistic')
    ax.set_ylabel('Feature')

    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    ax.set_title(
        f'Internal-node weights  |  model={model_type}  fold={fold_idx}'
        f'  tree={tree_idx}  node={node_idx}'
    )
    plt.tight_layout()
    plt.show()


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Heatmap of dGBDT internal node weights')
    parser.add_argument('--model', choices=['distill', 'distil_agg'], required=True,
                        help='Which distilled model to load')
    parser.add_argument('--fold',  type=int, default=0,
                        help='Fold index (0-4)')
    parser.add_argument('--tree',  type=int, default=0,
                        help='Tree index within the ensemble (0-4)')
    parser.add_argument('--node',  type=int, default=0,
                        help='Internal node index (0 = root; max = 2^depth - 2 = 30 for depth=5)')
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    n_internal = 2 ** TREE_DEPTH - 1  # 31 for depth=5
    if not (0 <= args.tree < 5):
        parser.error(f'--tree must be in [0, 4]')
    if not (0 <= args.node < n_internal):
        parser.error(f'--node must be in [0, {n_internal - 1}] for depth={TREE_DEPTH}')
    if not (0 <= args.fold < 5):
        parser.error(f'--fold must be in [0, 4]')

    input_dim = (N_FEATURES * N_HOURS if args.model == 'distill'
                 else N_STATIC + N_TEMPORAL * 4)

    model = load_model(args.model, args.fold)
    feature_w = extract_node_weights(model, args.tree, args.node, input_dim)

    heatmap = (build_heatmap_temporal(feature_w) if args.model == 'distill'
               else build_heatmap_agg(feature_w))

    plot_heatmap(heatmap, args.model, args.fold, args.tree, args.node)


if __name__ == '__main__':
    main()
