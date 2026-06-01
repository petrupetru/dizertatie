"""
Comprehensive exploratory data analysis for the Diabetes 130-US Hospitals dataset.
Produces figures + a console report to guide feature selection and top_n cutoffs.

Run from the DIABETES_130_US/ directory:
    python diabetes_analysis.py
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # no GUI needed; swap to TkAgg/Qt5Agg if you want pop-ups
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

DATA_FILE   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "diabetic_data.csv")
OUT_DIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis_plots")
os.makedirs(OUT_DIR, exist_ok=True)

# ── ICD-9 → clinical category ──────────────────────────────────────────────────
_DIAG_LABELS = {0:"Circulatory",1:"Respiratory",2:"Digestive",3:"Diabetes",
                4:"Injury",5:"Musculoskeletal",6:"Genitourinary",7:"Neoplasms",8:"Other"}

def _map_diag(d):
    if pd.isna(d): return "Other"
    s = str(d).strip()
    if s.startswith("E") or s.startswith("V"): return "Other"
    try: v = float(s)
    except ValueError: return "Other"
    if 390 <= v <= 459 or v == 785: return "Circulatory"
    if 460 <= v <= 519 or v == 786: return "Respiratory"
    if 520 <= v <= 579 or v == 787: return "Digestive"
    if 250 <= v < 251:              return "Diabetes"
    if 800 <= v <= 999:             return "Injury"
    if 710 <= v <= 739:             return "Musculoskeletal"
    if 580 <= v <= 629 or v == 788: return "Genitourinary"
    if 140 <= v <= 239:             return "Neoplasms"
    return "Other"

# ── Helpers ────────────────────────────────────────────────────────────────────

def cramers_v(col, target):
    """Cramér's V association between a categorical column and the binary target."""
    ct   = pd.crosstab(col, target)
    chi2 = stats.chi2_contingency(ct, correction=False)[0]
    n    = ct.values.sum()
    k    = min(ct.shape) - 1
    return np.sqrt(chi2 / (n * k)) if k > 0 else 0.0

def mann_whitney_auroc(series, target):
    """Rank-biserial / AUROC approximation via Mann-Whitney U (no sklearn needed)."""
    pos = series[target == 1].dropna()
    neg = series[target == 0].dropna()
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    u, _ = stats.mannwhitneyu(pos, neg, alternative="two-sided")
    return u / (len(pos) * len(neg))

def top_n_coverage(series, ns=(3, 5, 8, 10, 15, 20)):
    """Fraction of non-null rows covered by the top-n categories."""
    vc    = series.dropna().value_counts(normalize=True)
    cum   = vc.cumsum()
    result = {}
    for n in ns:
        result[n] = round(cum.iloc[min(n, len(cum)) - 1], 4)
    return result


# ── Main analysis ──────────────────────────────────────────────────────────────

def main():
    print("Loading data …")
    df = pd.read_csv(DATA_FILE)
    df = df.replace("?", np.nan)

    # ── 1. Basic info ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Dataset shape: {df.shape}")
    print(f"{'='*60}")

    # ── 2. Missing values ──────────────────────────────────────────────────────
    missing = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    missing = missing[missing > 0]
    print("\n── Missing values (%) ──")
    print(missing.to_string())

    fig, ax = plt.subplots(figsize=(9, 4))
    missing.plot.barh(ax=ax, color="salmon")
    ax.set_xlabel("Missing (%)")
    ax.set_title("Missing values per column")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "1_missing_values.png"), dpi=120)
    plt.close(fig)

    # ── 3. Target distribution ─────────────────────────────────────────────────
    print("\n── Target: readmitted ──")
    vc = df["readmitted"].value_counts()
    print(vc)
    print("→ Binarised as <30=1, else=0:")
    y = (df["readmitted"] == "<30").astype(int)
    print(y.value_counts())
    print(f"  Positive rate: {y.mean()*100:.1f}%")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    vc.plot.bar(ax=axes[0], color=["steelblue","orange","green"])
    axes[0].set_title("Raw readmission values"); axes[0].set_xlabel("")
    y.value_counts().plot.bar(ax=axes[1], color=["steelblue","salmon"])
    axes[1].set_title("Binary target (1 = <30 days)")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "2_target_distribution.png"), dpi=120)
    plt.close(fig)

    # ── 4. Remove patients who can't be readmitted (same as preprocessing) ────
    DISCHARGE_REMOVE = {11, 13, 14, 19, 20, 21}
    df = df[~df["discharge_disposition_id"].isin(DISCHARGE_REMOVE)].copy()
    df = df.sort_values("encounter_id").drop_duplicates("patient_nbr", keep="first")
    y  = (df["readmitted"] == "<30").astype(int)
    print(f"\nAfter cleaning (deduplicate, remove death/hospice): {len(df)} rows")
    print(f"  Positive rate: {y.mean()*100:.1f}%")

    # ── 5. Numeric feature analysis ────────────────────────────────────────────
    num_cols = [
        "time_in_hospital","num_lab_procedures","num_procedures","num_medications",
        "number_outpatient","number_emergency","number_inpatient","number_diagnoses",
    ]
    print("\n── Numeric features ──")
    print(df[num_cols].describe().round(2).to_string())

    aurocs = {c: mann_whitney_auroc(df[c], y) for c in num_cols}
    aurocs = dict(sorted(aurocs.items(), key=lambda x: abs(x[1]-0.5), reverse=True))
    print("\n  Rank-biserial AUROC (0.5 = no signal, 1.0 = perfect):")
    for k, v in aurocs.items():
        print(f"    {k:<30s} {v:.4f}")

    fig, ax = plt.subplots(figsize=(8, 4))
    pd.Series(aurocs).plot.barh(ax=ax, color="steelblue")
    ax.axvline(0.5, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("AUROC"); ax.set_title("Numeric features – rank-biserial AUROC vs <30 readmission")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "3_numeric_auroc.png"), dpi=120)
    plt.close(fig)

    # Distribution comparison: readmitted=1 vs 0
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    for ax, col in zip(axes.flat, num_cols):
        for cls, lbl, col_ in [(0,"No readm","steelblue"),(1,"<30 readm","salmon")]:
            vals = df.loc[y==cls, col].clip(upper=df[col].quantile(0.99))
            ax.hist(vals, bins=20, density=True, alpha=0.6, color=col_, label=lbl)
        ax.set_title(col, fontsize=8); ax.legend(fontsize=6)
    plt.suptitle("Numeric feature distributions by readmission class", y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "4_numeric_distributions.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Correlation matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    corr = df[num_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, linewidths=0.5)
    ax.set_title("Numeric feature correlation matrix")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "5_numeric_correlation.png"), dpi=120)
    plt.close(fig)

    # ── 6. Categorical feature analysis ───────────────────────────────────────
    # Map categoricals to cleaned strings before computing association
    cat_work = pd.DataFrame({"y": y})

    # age
    df["age_bucket"] = df["age"].map({
        "[0-10)":"le30","[10-20)":"le30","[20-30)":"le30",
        "[30-40)":"30-60","[40-50)":"30-60","[50-60)":"30-60",
        "[60-70)":"gt60","[70-80)":"gt60","[80-90)":"gt60","[90-100)":"gt60",
    })

    # diag → clinical category
    df["diag_1_cat"] = df["diag_1"].apply(_map_diag)
    df["diag_2_cat"] = df["diag_2"].apply(_map_diag)
    df["diag_3_cat"] = df["diag_3"].apply(_map_diag)

    cat_cols_raw = {
        "race":                    df["race"].fillna("Missing"),
        "gender":                  df["gender"].fillna("Missing"),
        "age_bucket":              df["age_bucket"].fillna("Missing"),
        "admission_type_id":       df["admission_type_id"].astype(str),
        "discharge_disposition_id":df["discharge_disposition_id"].astype(str),
        "admission_source_id":     df["admission_source_id"].astype(str),
        "medical_specialty":       df["medical_specialty"].fillna("Missing"),
        "max_glu_serum":           df["max_glu_serum"].fillna("None"),
        "A1Cresult":               df["A1Cresult"].fillna("None"),
        "change":                  df["change"].fillna("No"),
        "diabetesMed":             df["diabetesMed"].fillna("No"),
        "diag_1_cat":              df["diag_1_cat"],
        "diag_2_cat":              df["diag_2_cat"],
        "diag_3_cat":              df["diag_3_cat"],
    }

    print("\n── Categorical features – Cramér's V vs target ──")
    cv_scores = {}
    for name, series in cat_cols_raw.items():
        cv_scores[name] = cramers_v(series, y)
    cv_scores = dict(sorted(cv_scores.items(), key=lambda x: x[1], reverse=True))
    for k, v in cv_scores.items():
        print(f"    {k:<30s} {v:.4f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    pd.Series(cv_scores).plot.barh(ax=ax, color="mediumpurple")
    ax.set_xlabel("Cramér's V"); ax.set_title("Categorical features – Cramér's V vs <30 readmission")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "6_categorical_cramersv.png"), dpi=120)
    plt.close(fig)

    # ── 7. top_n coverage for high-cardinality categoricals ───────────────────
    high_card = {
        "discharge_disposition_id": df["discharge_disposition_id"].astype(str),
        "admission_source_id":      df["admission_source_id"].astype(str),
        "admission_type_id":        df["admission_type_id"].astype(str),
        "medical_specialty":        df["medical_specialty"].fillna("Missing"),
    }
    print("\n── top_n category coverage ──")
    for name, series in high_card.items():
        cov = top_n_coverage(series)
        print(f"  {name}:")
        for n, c in cov.items():
            print(f"    top_{n:2d}: {c*100:.1f}%")
        print(f"  top 10 values: {series.value_counts().head(10).to_dict()}")
        print()

    # ── 8. Medication feature analysis ────────────────────────────────────────
    med_cols_all = [
        "metformin","repaglinide","nateglinide","chlorpropamide","glimepiride",
        "acetohexamide","glipizide","glyburide","tolbutamide","pioglitazone",
        "rosiglitazone","acarbose","miglitol","troglitazone","tolazamide",
        "examide","citoglipton","insulin","glyburide-metformin",
        "glipizide-metformin","glimepiride-pioglitazone",
        "metformin-rosiglitazone","metformin-pioglitazone",
    ]

    # Pct of patients with any prescription (non-"No")
    print("── Medication prescription rates ──")
    med_rates = {}
    for c in med_cols_all:
        rate = (df[c] != "No").mean()
        med_rates[c] = rate
        print(f"  {c:<35s} prescribed: {rate*100:5.1f}%")

    fig, ax = plt.subplots(figsize=(10, 6))
    pd.Series(med_rates).sort_values(ascending=False).plot.bar(ax=ax, color="teal")
    ax.axhline(0.01, color="red", linestyle="--", linewidth=1, label="1% threshold")
    ax.set_ylabel("Fraction prescribed"); ax.set_title("Medication prescription rates")
    ax.legend(); plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "7_medication_rates.png"), dpi=120)
    plt.close(fig)

    # Cramér's V per medication vs target
    print("\n── Medication Cramér's V vs target ──")
    med_cv = {}
    for c in med_cols_all:
        med_cv[c] = cramers_v(df[c].fillna("No"), y)
    med_cv = dict(sorted(med_cv.items(), key=lambda x: x[1], reverse=True))
    for k, v in med_cv.items():
        print(f"  {k:<35s} {v:.4f}")

    fig, ax = plt.subplots(figsize=(10, 6))
    pd.Series(med_cv).sort_values(ascending=False).plot.bar(ax=ax, color="coral")
    ax.set_ylabel("Cramér's V"); ax.set_title("Medication Cramér's V vs <30 readmission")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "8_medication_cramersv.png"), dpi=120)
    plt.close(fig)

    # ── 9. Random Forest feature importance ───────────────────────────────────
    print("\n── Random Forest feature importance ──")

    rf_df = df.copy()
    # Encode numerics directly; encode categoricals with LabelEncoder
    all_feat_cols = num_cols.copy()

    # Binary flags
    rf_df["change_bin"]      = (rf_df["change"] == "Ch").astype(int)
    rf_df["diabetesMed_bin"] = (rf_df["diabetesMed"] == "Yes").astype(int)
    all_feat_cols += ["change_bin", "diabetesMed_bin"]

    # Categorical: label-encode
    le_cols = {
        "age_bucket": rf_df["age_bucket"].fillna("Missing"),
        "race":       rf_df["race"].fillna("Missing"),
        "gender":     rf_df["gender"].fillna("Missing"),
        "admission_type_id":        rf_df["admission_type_id"].astype(str),
        "discharge_disposition_id": rf_df["discharge_disposition_id"].astype(str),
        "admission_source_id":      rf_df["admission_source_id"].astype(str),
        "medical_specialty":        rf_df["medical_specialty"].fillna("Missing"),
        "max_glu_serum":            rf_df["max_glu_serum"].fillna("None"),
        "A1Cresult":                rf_df["A1Cresult"].fillna("None"),
        "diag_1_cat":               rf_df["diag_1_cat"],
    }
    le = LabelEncoder()
    for name, series in le_cols.items():
        rf_df[f"_le_{name}"] = le.fit_transform(series.astype(str))
        all_feat_cols.append(f"_le_{name}")

    # Medications: binary
    for c in med_cols_all:
        rf_df[f"_med_{c}"] = (rf_df[c] != "No").astype(int)
        all_feat_cols.append(f"_med_{c}")

    X = rf_df[all_feat_cols].values
    y_arr = y.values

    rf = RandomForestClassifier(n_estimators=200, max_depth=8, n_jobs=-1,
                                random_state=42, class_weight="balanced")
    rf.fit(X, y_arr)

    feat_names_clean = [
        c.replace("_le_", "").replace("_med_", "med:").replace("_bin", "")
        for c in all_feat_cols
    ]
    imp = pd.Series(rf.feature_importances_, index=feat_names_clean).sort_values(ascending=False)
    print(imp.to_string())

    fig, ax = plt.subplots(figsize=(10, 8))
    imp.head(30).sort_values().plot.barh(ax=ax, color="steelblue")
    ax.set_xlabel("Importance"); ax.set_title("RF feature importances (top 30)")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "9_rf_feature_importance.png"), dpi=120)
    plt.close(fig)

    # ── 10. Readmission rate per categorical level ────────────────────────────
    print("\n── Readmission rate by discharge_disposition_id ──")
    tmp = df[["discharge_disposition_id"]].copy()
    tmp["readmitted"] = y.values
    print(tmp.groupby("discharge_disposition_id")["readmitted"].agg(["mean","count"])
          .sort_values("mean", ascending=False).to_string())

    print("\n── Readmission rate by admission_type_id ──")
    tmp2 = df[["admission_type_id"]].copy()
    tmp2["readmitted"] = y.values
    print(tmp2.groupby("admission_type_id")["readmitted"].agg(["mean","count"])
          .sort_values("mean", ascending=False).to_string())

    print("\n── Readmission rate by medical_specialty (top 15) ──")
    tmp3 = df[["medical_specialty"]].fillna("Missing").copy()
    tmp3["readmitted"] = y.values
    agg = tmp3.groupby("medical_specialty")["readmitted"].agg(["mean","count"])
    print(agg[agg["count"] >= 200].sort_values("mean", ascending=False).head(15).to_string())

    # ── 11. Summary recommendations ───────────────────────────────────────────
    print(f"\n{'='*60}")
    print("RECOMMENDATIONS FOR FEATURE SELECTION")
    print(f"{'='*60}")

    print("""
NUMERIC FEATURES (keep all – low missing, meaningful signal):
  time_in_hospital, num_lab_procedures, num_procedures,
  num_medications, number_outpatient, number_emergency,
  number_inpatient (*strongest predictor), number_diagnoses

HIGH-SIGNAL CATEGORICALS (sorted by Cramér's V above):
  - discharge_disposition_id  → top_n=5 (covers ~95%)
  - diag_1_cat                → 9 ICD categories (keep all)
  - admission_type_id         → top_n=3 (covers ~89%)
  - medical_specialty         → top_n=5 + Missing (49% missing)
  - admission_source_id       → top_n=3 (~92%)
  - A1Cresult                 → 4 values + None (treat 83% missing as "None")
  - age                       → 3 buckets (le30 / 30-60 / gt60)
  - race                      → top_n=2 + Missing + Other
  - max_glu_serum             → 3 values + None (95% missing → "None")
  - change                    → binary
  - diabetesMed               → binary
  - gender                    → binary (Female/Male)

MEDICATIONS (keep those with >5% prescription rate):
  insulin (~54%), metformin (~20%), glipizide (~12%),
  glyburide (~10%), pioglitazone (~7%), rosiglitazone (~6%),
  glimepiride (~5%)
  Encoding: 0=No, 1=Steady, 2=Up/Down

DROP:
  weight (97% missing), payer_code (40% missing, low signal)
  examide, citoglipton, acetohexamide, tolazamide, tolbutamide,
  troglitazone, miglitol, acarbose, glipizide-metformin,
  glimepiride-pioglitazone, metformin-rosiglitazone,
  metformin-pioglitazone  (all near-zero prescription rates)

top_n suggestion per column:
  discharge_disposition_id : 5
  admission_type_id         : 3
  admission_source_id       : 3
  medical_specialty         : 5  (+ "Missing" bucket)
  race                      : 2  (+ "Missing")
""")

    print(f"\nAll plots saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
