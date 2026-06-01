import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

from constants import FEATURE_NAMES, LABEL_NAME

_DISCHARGE_REMOVE = {11, 13, 14, 19, 20, 21}

# Medications with >5% prescription rate; kept for OHE
_MED_COLS = [
    "insulin", "metformin", "glipizide", "glyburide",
    "pioglitazone", "rosiglitazone", "glimepiride",
]
_MED_CATS = ["Down", "No", "Steady", "Up"]   # fixed OHE category set

# Age bracket -> midpoint
_AGE_MIDPOINT = {
    "[0-10)": 5,   "[10-20)": 15,  "[20-30)": 25,  "[30-40)": 35,
    "[40-50)": 45, "[50-60)": 55,  "[60-70)": 65,  "[70-80)": 75,
    "[80-90)": 85, "[90-100)": 95,
}

# ICD-9 clinical category labels (same 9 used for the multi-label diagnosis feature)
_DIAG_CATS = [
    "Circulatory", "Respiratory", "Digestive", "Diabetes",
    "Injury", "Musculoskeletal", "Genitourinary", "Neoplasms", "Other",
]


def _map_diag(d):
    if pd.isna(d):
        return "Other"
    s = str(d).strip()
    if s.startswith("E") or s.startswith("V"):
        return "Other"
    try:
        v = float(s)
    except ValueError:
        return "Other"
    if 390 <= v <= 459 or v == 785: return "Circulatory"
    if 460 <= v <= 519 or v == 786: return "Respiratory"
    if 520 <= v <= 579 or v == 787: return "Digestive"
    if 250 <= v < 251:              return "Diabetes"
    if 800 <= v <= 999:             return "Injury"
    if 710 <= v <= 739:             return "Musculoskeletal"
    if 580 <= v <= 629 or v == 788: return "Genitourinary"
    if 140 <= v <= 239:             return "Neoplasms"
    return "Other"


def _apply_top_n(series, n, missing_fill=None):
    """Keep the top-n most frequent categories; collapse the rest to 'Other'.

    If *missing_fill* is given, NaN is filled with that label before counting.
    """
    s = series.fillna(missing_fill) if missing_fill is not None else series
    top = s.value_counts().nlargest(n).index
    return s.apply(lambda x: x if x in top else "Other")


def preprocess_diabetes(filepath, top_n=None):
    """Preprocess the Diabetes 130-US dataset.

    Parameters
    ----------
    filepath : str
        Path to diabetic_data.csv.
    top_n : dict | None
        Per-column override of how many top categories to retain before
        collapsing the rest to 'Other'.
        Keys: discharge_id, admission_type, admission_src, specialty, race.
        Pass None to use defaults (from diabetes_analysis.py coverage analysis).
    """
    _top_n = {
        "discharge_id":   5,   # ~95% coverage
        "admission_type": 3,   # ~89% coverage
        "admission_src":  3,   # ~92% coverage
        "specialty":      5,   # +Missing bucket
        "race":           3,   # +Missing bucket
    }
    if top_n:
        _top_n.update(top_n)

    df = pd.read_csv(filepath)
    df = df[~df["discharge_disposition_id"].isin(_DISCHARGE_REMOVE)].copy()
    df = df.replace("?", np.nan)
    df = df.sort_values("encounter_id").drop_duplicates("patient_nbr", keep="first")

    df[LABEL_NAME] = (df["readmitted"] == "<30").astype(int)

    # ── Binary flags ──────────────────────────────────────────────────────────
    df["change_bin"]      = (df["change"] == "Ch").astype(float)
    df["diabetesMed_bin"] = (df["diabetesMed"] == "Yes").astype(float)

    # ── Age: midpoint of each 10-year bracket (numeric) ───────────────────────
    df["age_midpoint"] = df["age"].map(_AGE_MIDPOINT).fillna(75).astype(float)

    # ── Gender: binary (1 = Male) ─────────────────────────────────────────────
    df["gender_male"] = (df["gender"] == "Male").astype(float)

    # ── Race: top_n + Missing bucket ──────────────────────────────────────────
    df["race"] = _apply_top_n(df["race"], _top_n["race"], missing_fill="Missing")

    # ── Discharge disposition: top_n ──────────────────────────────────────────
    df["discharge_id"] = _apply_top_n(
        df["discharge_disposition_id"].astype(str), _top_n["discharge_id"]
    )

    # ── Admission type: top_n ─────────────────────────────────────────────────
    df["admission_type"] = _apply_top_n(
        df["admission_type_id"].astype(str), _top_n["admission_type"]
    )

    # ── Admission source: top_n ───────────────────────────────────────────────
    df["admission_src"] = _apply_top_n(
        df["admission_source_id"].astype(str), _top_n["admission_src"]
    )

    # ── Medical specialty: top_n + Missing bucket ─────────────────────────────
    df["specialty"] = _apply_top_n(
        df["medical_specialty"], _top_n["specialty"], missing_fill="Missing"
    )

    # ── Diagnosis: multi-label from diag_1 + diag_2 + diag_3 ─────────────────
    # Each patient gets a 1 in any category present across all 3 diagnosis fields.
    for cat in _DIAG_CATS:
        df[f"diag_{cat}"] = 0.0
    for diag_col in ["diag_1", "diag_2", "diag_3"]:
        mapped = df[diag_col].apply(_map_diag)
        for cat in _DIAG_CATS:
            df.loc[mapped == cat, f"diag_{cat}"] = 1.0

    # ── HbA1c: 5-level ───────────────────────────────────────────────────────
    _high    = df["A1Cresult"] == ">8"
    _changed = df["change"] == "Ch"
    df["hba1c"] = "None"
    df.loc[df["A1Cresult"] == "Norm", "hba1c"] = "Norm"
    df.loc[df["A1Cresult"] == ">7",   "hba1c"] = ">7"
    df.loc[_high & ~_changed,         "hba1c"] = "high_no_change"
    df.loc[_high & _changed,          "hba1c"] = "high_changed"

    # ── Glucose serum: 4 levels ───────────────────────────────────────────────
    df["glucose"] = df["max_glu_serum"].fillna("None")

    # ── Medication OHE (4 categories: Down / No / Steady / Up) ───────────────
    # Fixed category set ensures consistent columns across train/test splits.
    for c in _MED_COLS:
        df[f"med_{c}"] = pd.Categorical(df[c].fillna("No"), categories=_MED_CATS)
    df = pd.get_dummies(df, columns=[f"med_{c}" for c in _MED_COLS], dtype=float)

    # ── OHE for remaining categoricals ───────────────────────────────────────
    def _ohe(col):
        cats = sorted(df[col].dropna().unique())
        df[col] = pd.Categorical(df[col], categories=cats)

    ohe_cols = ["race", "discharge_id", "admission_type",
                "admission_src", "specialty", "hba1c", "glucose"]
    for col in ohe_cols:
        _ohe(col)
    df = pd.get_dummies(df, columns=ohe_cols, dtype=float)

    return df[FEATURE_NAMES + [LABEL_NAME]].reset_index(drop=True)


class DiabetesDataset(Dataset):
    def __init__(self, df):
        self.X = torch.tensor(df[FEATURE_NAMES].values.astype(float), dtype=torch.float32)
        self.y = torch.tensor(df[LABEL_NAME].values.astype(float), dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class TreeEmbedDataset(Dataset):
    def __init__(self, leaf_indices, labels):
        self.leaf_indices = torch.tensor(leaf_indices, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.leaf_indices[idx], self.labels[idx]
