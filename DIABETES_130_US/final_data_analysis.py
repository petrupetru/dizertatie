"""
Exploratory data analysis for the FINAL preprocessed Diabetes 130-US dataset.
Assumes preprocess_diabetes() and FEATURE_NAMES are up to date.

Run from the DIABETES_130_US/ directory:
    python final_data_analysis.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from constants import DATA_FILE, FEATURE_NAMES, LABEL_NAME
from diabetes_dataset import preprocess_diabetes

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "final_analysis_plots")
os.makedirs(OUT_DIR, exist_ok=True)

def main():
    print("Loading and preprocessing data …")
    df = preprocess_diabetes(DATA_FILE)
    print(f"Shape: {df.shape}")
    print(f"Nulls: {df.isnull().sum().sum()}")
    print(f"Label distribution: {df[LABEL_NAME].value_counts().to_dict()}")

    # ── 1. Numeric features ────────────────────────────────────────────────
    num_cols = [
        "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
        "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses",
        "age_midpoint"
    ]
    print("\nNumeric features summary:")
    print(df[num_cols].describe().round(2).to_string())

    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    for ax, col in zip(axes.flat, num_cols):
        for cls, lbl, col_ in [(0,"No readm","steelblue"),(1,"<30 readm","salmon")]:
            vals = df.loc[df[LABEL_NAME]==cls, col].clip(upper=df[col].quantile(0.99))
            ax.hist(vals, bins=20, density=False, alpha=0.6, color=col_, label=lbl)  # density=False for counts
        ax.set_title(col, fontsize=9); ax.legend(fontsize=7)
    plt.suptitle("Numeric feature distributions by readmission class (counts)", y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "1_numeric_distributions.png"), dpi=120)
    plt.close(fig)

    # ── 2. Binary features ─────────────────────────────────────────────────
    bin_cols = ["gender_male", "change_bin", "diabetesMed_bin"]
    print("\nBinary features (mean):")
    print(df[bin_cols].mean().round(3))
    fig, ax = plt.subplots(figsize=(6, 2))
    (df[bin_cols].sum().sort_values() if len(bin_cols) > 1 else df[bin_cols].sum()).plot.bar(ax=ax, color="teal")
    ax.set_title("Binary feature counts (sum of 1s)")
    ax.set_ylabel("Count")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "2_binary_counts.png"), dpi=120)
    plt.close(fig)

    # ── 3. Diagnosis multi-label ─────────────────────────────────────────--
    diag_cols = [c for c in df.columns if c.startswith("diag_")]
    print("\nDiagnosis multi-label prevalence:")
    print(df[diag_cols].mean().round(3))
    fig, ax = plt.subplots(figsize=(8, 3))
    df[diag_cols].sum().sort_values().plot.bar(ax=ax, color="purple")
    ax.set_title("Diagnosis category counts (multi-label)")
    ax.set_ylabel("Count")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "3_diag_multilabel.png"), dpi=120)
    plt.close(fig)

    # ── 4. Medication OHE ─────────────────────────────────────────────────-
    med_cols = [c for c in df.columns if c.startswith("med_")]
    print("\nMedication OHE prevalence:")
    print(df[med_cols].mean().round(3))
    fig, ax = plt.subplots(figsize=(12, 4))
    df[med_cols].sum().sort_values().plot.bar(ax=ax, color="orange")
    ax.set_title("Medication OHE counts")
    ax.set_ylabel("Count")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "4_medication_ohe.png"), dpi=120)
    plt.close(fig)

    # ── 5. Categorical OHE features ─────────────────────────────────────---
    ohe_blocks = [
        ("Race", [c for c in df.columns if c.startswith("race_")]),
        ("Discharge", [c for c in df.columns if c.startswith("discharge_id_")]),
        ("AdmissionType", [c for c in df.columns if c.startswith("admission_type_")]),
        ("AdmissionSrc", [c for c in df.columns if c.startswith("admission_src_")]),
        ("Specialty", [c for c in df.columns if c.startswith("specialty_")]),
        ("HbA1c", [c for c in df.columns if c.startswith("hba1c_")]),
        ("Glucose", [c for c in df.columns if c.startswith("glucose_")]),
    ]
    for name, cols in ohe_blocks:
        print(f"\n{name} OHE prevalence:")
        print(df[cols].mean().round(3))
        fig, ax = plt.subplots(figsize=(8, 2))
        df[cols].sum().sort_values().plot.bar(ax=ax)
        ax.set_title(f"{name} OHE counts")
        ax.set_ylabel("Count")
        plt.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f"5_{name.lower()}_ohe.png"), dpi=120)
        plt.close(fig)

    # ── 6. Correlation matrix ─────────────────────────────────────────------
    corr = df[num_cols + bin_cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, linewidths=0.5)
    ax.set_title("Numeric + binary feature correlation matrix")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "6_correlation_matrix.png"), dpi=120)
    plt.close(fig)

    print(f"\nAll plots saved to: {OUT_DIR}")

if __name__ == "__main__":
    main()
