"""
Standalone dGBDT training on MIMIC-III.
Loads data identically to main.py, flattens the (N, 1, 22, 48) tensor to
(N, 1056), and runs k-fold cross-validation.
"""
import os
import numpy as np

from preprocess import build_dataset
from dGBT import train_kfold_dgbdt


def main():
    input_df, labels_df = build_dataset()

    n_patients   = input_df.index.get_level_values('subject_id').nunique()
    n_predictors = input_df.index.get_level_values('measurement_name').nunique()
    print(f"Patients: {n_patients}, Predictors: {n_predictors}, Hours: {input_df.shape[1]}")

    X = input_df.values.reshape((n_patients, n_predictors, 48)).astype(np.float32)
    X_flat = X.reshape(n_patients, -1)   # (N, 22*48) = (N, 1056)
    y = labels_df['mort_icu'].values.astype(np.float32)
    print(f"X_flat shape: {X_flat.shape}, y shape: {y.shape}, mortality rate: {y.mean():.3f}")

    results = train_kfold_dgbdt(
        X_flat=X_flat,
        y=y,
        n_splits=5,
        num_epochs=100,
        lr=1e-3,
        batch_size=512,
        device="cuda" if __import__("torch").cuda.is_available() else "cpu",
        num_trees=5,
        depth=5,
        lamda=1e-3,
        seed=42,
        verbose=True,
    )

    print(
        f"\nFinal  AUROC: {results['mean_auroc']:.4f} ± {results['std_auroc']:.4f}"
        f"  AUPRC: {results['mean_auprc']:.4f} ± {results['std_auprc']:.4f}"
    )


if __name__ == '__main__':
    main()
