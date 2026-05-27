"""
NHANES 2007-2018 — Insulin Resistance classification
======================================================
Replicates the experimental setup from:
  Xing et al. (2024) "Discriminating insulin resistance in middle-aged
  nondiabetic women using machine learning approaches",
  AIMS Public Health, 11(1):74-100.

Target metric: AUROC ≈ 0.86 on test fold (5-fold CV).
"""

import os
import sys
import argparse

# Ensure NHANES/ is on the path when called from the workspace root
sys.path.insert(0, os.path.dirname(__file__))

import torch

from constants import SEED, LABEL_NAME
from utils import set_seed, load_nhanes_data
from TN_net import TN_DNN, train_kfold_TNT
from dGBT_generated import train_kfold_tnt_dgbdt


def parse_args():
    parser = argparse.ArgumentParser(description="Train TNT on NHANES IR dataset")
    parser.add_argument(
        "--mode",
        choices=["teacher", "tnt", "dgbdt"],
        default="dgbdt",
        help=(
            "teacher: TN-DNN only (k-fold); "
            "tnt: TN-DNN + SDT distillation; "
            "dgbdt: TN-DNN + dGBDT distillation (default)"
        ),
    )
    parser.add_argument("--n_splits",       type=int,   default=5)
    parser.add_argument("--num_epochs",     type=int,   default=50)
    parser.add_argument("--num_epochs_dgbdt", type=int, default=50)
    parser.add_argument("--lr",             type=float, default=1e-3)
    parser.add_argument("--lr_dgbdt",       type=float, default=1e-3)
    parser.add_argument("--batch_size",     type=int,   default=32)
    parser.add_argument("--dgbdt_trees",    type=int,   default=5)
    parser.add_argument("--dgbdt_depth",    type=int,   default=5)
    parser.add_argument("--alpha",          type=float, default=0.5)
    parser.add_argument("--temperature",    type=float, default=1.0)
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "data"),
        help="Directory for NHANES XPT cache and processed CSV",
    )
    parser.add_argument("--device",   default="cpu")
    parser.add_argument("--verbose",  action="store_true", default=True)
    parser.add_argument("--quiet",    dest="verbose", action="store_false")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(SEED)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = "cpu"

    # ------------------------------------------------------------------ #
    # 1. Load data                                                         #
    # ------------------------------------------------------------------ #
    print("=" * 60)
    print("Loading NHANES data ...")
    df = load_nhanes_data(data_dir=args.data_dir)

    n_total = len(df)
    n_ir    = int(df[LABEL_NAME].sum())
    n_no_ir = n_total - n_ir
    print(f"Dataset: {n_total} samples  |  IR=1: {n_ir}  |  IR=0: {n_no_ir}")
    print(f"Prevalence: {n_ir / n_total:.1%}")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    # 2. Run the selected experiment                                       #
    # ------------------------------------------------------------------ #
    # if args.mode == "teacher":
    #     print("\nRunning: TN-DNN teacher only (k-fold CV)")
    #     results = train_kfold_TNT(
    #         df=df,
    #         model_class=TN_DNN,
    #         n_splits=args.n_splits,
    #         num_epochs=args.num_epochs,
    #         lr=args.lr,
    #         batch_size=args.batch_size,
    #         device=device,
    #         verbose=args.verbose,
    #         distilation=False,
    #     )
    #     print("\nFinal Results:")
    #     print(f"  Teacher AUROC: {results['mean_teacher_auroc']:.4f} ± {results['std_teacher_auroc']:.4f}")
    #     print(f"  Teacher AUPRC: {results['mean_teacher_auprc']:.4f} ± {results['std_teacher_auprc']:.4f}")

    # elif args.mode == "tnt":
    #     print("\nRunning: TN-DNN teacher + SDT distillation (k-fold CV)")
    #     results = train_kfold_TNT(
    #         df=df,
    #         model_class=TN_DNN,
    #         n_splits=args.n_splits,
    #         num_epochs=args.num_epochs,
    #         lr=args.lr,
    #         batch_size=args.batch_size,
    #         device=device,
    #         verbose=args.verbose,
    #         distilation=True,
    #     )
    #     print("\nFinal Results:")
    #     print(f"  Teacher AUROC: {results['mean_teacher_auroc']:.4f} ± {results['std_teacher_auroc']:.4f}")
    #     print(f"  Teacher AUPRC: {results['mean_teacher_auprc']:.4f} ± {results['std_teacher_auprc']:.4f}")
    #     if "mean_soft_tree_auroc" in results:
    #         print(f"  SDT     AUROC: {results['mean_soft_tree_auroc']:.4f} ± {results['std_soft_tree_auroc']:.4f}")
    #         print(f"  SDT     AUPRC: {results['mean_soft_tree_auprc']:.4f} ± {results['std_soft_tree_auprc']:.4f}")

    # else:  # dgbdt (default)
    #     print("\nRunning: TN-DNN teacher + dGBDT distillation (k-fold CV)")
    #     results = train_kfold_tnt_dgbdt(
    #         df=df,
    #         model_class=TN_DNN,
    #         n_splits=args.n_splits,
    #         num_epochs=args.num_epochs,
    #         num_epochs_dgbdt=args.num_epochs_dgbdt,
    #         lr=args.lr,
    #         lr_dgbdt=args.lr_dgbdt,
    #         batch_size=args.batch_size,
    #         device=device,
    #         alpha=args.alpha,
    #         temperature=args.temperature,
    #         dgbdt_num_trees=args.dgbdt_trees,
    #         dgbdt_depth=args.dgbdt_depth,
    #         verbose=args.verbose,
    #     )
    #     print("\nFinal Results:")
    #     print(f"  Teacher AUROC: {results['mean_teacher_auroc']:.4f} ± {results['std_teacher_auroc']:.4f}")
    #     print(f"  Teacher AUPRC: {results['mean_teacher_auprc']:.4f} ± {results['std_teacher_auprc']:.4f}")
    #     print(f"  dGBDT   AUROC: {results['mean_dgbdt_auroc']:.4f} ± {results['std_dgbdt_auroc']:.4f}")
    #     print(f"  dGBDT   AUPRC: {results['mean_dgbdt_auprc']:.4f} ± {results['std_dgbdt_auprc']:.4f}")

    # print("\nTarget from Xing et al. 2024: XGBoost test AUC ≈ 0.86")


if __name__ == "__main__":
    main()
