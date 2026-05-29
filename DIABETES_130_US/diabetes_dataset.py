import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

from constants import FEATURE_NAMES, LABEL_NAME


def _map_diag(d):
    """Map an ICD-9 code to one of 9 disease categories (0-8)."""
    if pd.isna(d):
        return 8
    d_str = str(d).strip()
    if d_str.startswith("E") or d_str.startswith("V"):
        return 8
    try:
        d_val = float(d_str)
    except ValueError:
        return 8

    if 390 <= d_val <= 459 or d_val == 785:
        return 0  # circulatory
    if 460 <= d_val <= 519 or d_val == 786:
        return 1  # respiratory
    if 520 <= d_val <= 579 or d_val == 787:
        return 2  # digestive
    if 250 <= d_val < 251:
        return 3  # diabetes
    if 800 <= d_val <= 999:
        return 4  # injury
    if 710 <= d_val <= 739:
        return 5  # musculoskeletal
    if 580 <= d_val <= 629 or d_val == 788:
        return 6  # genitourinary
    if 140 <= d_val <= 239:
        return 7  # neoplasms
    return 8  # other


def preprocess_diabetes(filepath):
    df = pd.read_csv(filepath)

    # Remove patients who died or were discharged to hospice (cannot be readmitted)
    discharge_remove = {11, 13, 14, 19, 20, 21}
    df = df[~df["discharge_disposition_id"].isin(discharge_remove)].copy()

    # Replace sentinel missing value
    df = df.replace("?", np.nan)

    # Keep only the first encounter per patient to avoid temporal leakage
    df = df.sort_values("encounter_id").drop_duplicates(subset="patient_nbr", keep="first")

    # Drop identifier and high-missingness columns
    df = df.drop(columns=["encounter_id", "patient_nbr", "weight", "payer_code"])

    # Binary target: readmitted within 30 days = 1
    df[LABEL_NAME] = (df["readmitted"] == "<30").astype(int)
    df = df.drop(columns=["readmitted"])

    # Age: ordinal
    age_map = {
        "[0-10)": 0, "[10-20)": 1, "[20-30)": 2, "[30-40)": 3, "[40-50)": 4,
        "[50-60)": 5, "[60-70)": 6, "[70-80)": 7, "[80-90)": 8, "[90-100)": 9,
    }
    df["age_enc"] = df["age"].map(age_map).astype(float)
    df = df.drop(columns=["age"])

    # Race: label encode; missing/unknown → 5
    race_map = {"Caucasian": 0, "AfricanAmerican": 1, "Hispanic": 2, "Asian": 3, "Other": 4}
    df["race_enc"] = df["race"].map(race_map).fillna(5).astype(float)
    df = df.drop(columns=["race"])

    # Gender: binary
    df["gender_enc"] = df["gender"].map({"Female": 0, "Male": 1}).fillna(0).astype(float)
    df = df.drop(columns=["gender"])

    # Medical specialty: group rare (<200 occurrences) as "Other", label encode
    df["medical_specialty"] = df["medical_specialty"].fillna("Unknown")
    rare = df["medical_specialty"].value_counts()
    rare = rare[rare < 200].index
    df["medical_specialty"] = df["medical_specialty"].replace(rare, "Other")
    df["medical_specialty_enc"] = df["medical_specialty"].astype("category").cat.codes.astype(float)
    df = df.drop(columns=["medical_specialty"])

    # Glucose serum and HbA1c: ordinal
    df["max_glu_serum_enc"] = df["max_glu_serum"].map({"Norm": 1, ">200": 2, ">300": 3}).fillna(0).astype(float)
    df["A1Cresult_enc"]     = df["A1Cresult"].map({"Norm": 1, ">7": 2, ">8": 3}).fillna(0).astype(float)
    df = df.drop(columns=["max_glu_serum", "A1Cresult"])

    # Diagnoses: ICD-9 → disease category
    for col in ["diag_1", "diag_2", "diag_3"]:
        df[col + "_cat"] = df[col].apply(_map_diag).astype(float)
    df = df.drop(columns=["diag_1", "diag_2", "diag_3"])

    # Medication columns: No/Steady/Up/Down → 0/1/2/3
    med_map = {"No": 0, "Steady": 1, "Up": 2, "Down": 3}
    single_meds = [
        "metformin", "repaglinide", "nateglinide", "chlorpropamide",
        "glimepiride", "acetohexamide", "glipizide", "glyburide",
        "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
        "miglitol", "troglitazone", "tolazamide", "examide",
        "citoglipton", "insulin",
    ]
    for col in single_meds:
        df[col] = df[col].map(med_map).fillna(0).astype(float)

    combo_meds = {
        "glyburide-metformin": "glyburide_metformin",
        "glipizide-metformin": "glipizide_metformin",
        "glimepiride-pioglitazone": "glimepiride_pioglitazone",
        "metformin-rosiglitazone": "metformin_rosiglitazone",
        "metformin-pioglitazone": "metformin_pioglitazone",
    }
    for old, new in combo_meds.items():
        df[new] = df[old].map(med_map).fillna(0).astype(float)
    df = df.drop(columns=list(combo_meds.keys()))

    # Binary flags
    df["change_enc"]      = (df["change"] == "Ch").astype(float)
    df["diabetesMed_enc"] = (df["diabetesMed"] == "Yes").astype(float)
    df = df.drop(columns=["change", "diabetesMed"])

    return df.reset_index(drop=True)


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
