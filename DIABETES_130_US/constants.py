FEATURE_NAMES = [
    # Numeric
    "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses",
    "admission_type_id", "discharge_disposition_id", "admission_source_id",
    # Encoded categorical
    "race_enc", "gender_enc", "age_enc", "medical_specialty_enc",
    "max_glu_serum_enc", "A1Cresult_enc",
    "diag_1_cat", "diag_2_cat", "diag_3_cat",
    # Medications (No/Steady/Up/Down → 0/1/2/3)
    "metformin", "repaglinide", "nateglinide", "chlorpropamide",
    "glimepiride", "acetohexamide", "glipizide", "glyburide",
    "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
    "miglitol", "troglitazone", "tolazamide", "examide",
    "citoglipton", "insulin",
    "glyburide_metformin", "glipizide_metformin", "glimepiride_pioglitazone",
    "metformin_rosiglitazone", "metformin_pioglitazone",
    # Binary
    "change_enc", "diabetesMed_enc",
]

import os as _os

LABEL_NAME = "readmitted_bin"
DATA_FILE  = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "data", "diabetic_data.csv")
SEED       = 42
