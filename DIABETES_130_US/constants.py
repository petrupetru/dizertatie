FEATURE_NAMES = [
    # ── Numeric (pass-through) ────────────────────────────────────────────────
    "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses",
    # ── Scalar engineered ─────────────────────────────────────────────────────
    "age_midpoint",     # midpoint of 10-year bracket (numeric)
    "gender_male",      # binary: 1 = Male
    "change_bin",       # binary: 1 = medication dose changed
    "diabetesMed_bin",  # binary: 1 = any diabetes medication prescribed
    # ── Diagnosis: multi-label from diag_1 + diag_2 + diag_3 (ICD-9 -> 9 cats)
    "diag_Circulatory", "diag_Respiratory", "diag_Digestive", "diag_Diabetes",
    "diag_Injury", "diag_Musculoskeletal", "diag_Genitourinary",
    "diag_Neoplasms", "diag_Other",
    # ── Medication OHE (Down / No / Steady / Up) for 7 drugs ─────────────────
    "med_insulin_Down",       "med_insulin_No",       "med_insulin_Steady",       "med_insulin_Up",
    "med_metformin_Down",     "med_metformin_No",     "med_metformin_Steady",     "med_metformin_Up",
    "med_glipizide_Down",     "med_glipizide_No",     "med_glipizide_Steady",     "med_glipizide_Up",
    "med_glyburide_Down",     "med_glyburide_No",     "med_glyburide_Steady",     "med_glyburide_Up",
    "med_pioglitazone_Down",  "med_pioglitazone_No",  "med_pioglitazone_Steady",  "med_pioglitazone_Up",
    "med_rosiglitazone_Down", "med_rosiglitazone_No", "med_rosiglitazone_Steady", "med_rosiglitazone_Up",
    "med_glimepiride_Down",   "med_glimepiride_No",   "med_glimepiride_Steady",   "med_glimepiride_Up",
    # ── Race OHE (top_n=3 + Other) ────────────────────────────────────────────
    "race_AfricanAmerican", "race_Caucasian", "race_Missing", "race_Other",
    # ── Discharge disposition OHE (top_n=5 + Other) ───────────────────────────
    "discharge_id_1", "discharge_id_18", "discharge_id_2",
    "discharge_id_3", "discharge_id_6", "discharge_id_Other",
    # ── Admission type OHE (top_n=3 + Other) ──────────────────────────────────
    "admission_type_1", "admission_type_2", "admission_type_3", "admission_type_Other",
    # ── Admission source OHE (top_n=3 + Other) ────────────────────────────────
    "admission_src_1", "admission_src_17", "admission_src_7", "admission_src_Other",
    # ── Medical specialty OHE (top_n=5 + Other; Missing is top-5 due to 49% missing)
    "specialty_Cardiology", "specialty_Emergency/Trauma",
    "specialty_Family/GeneralPractice", "specialty_InternalMedicine",
    "specialty_Missing", "specialty_Other",
    # ── HbA1c OHE (5-level: None / Norm / >7 / high_no_change / high_changed) ─
    "hba1c_>7", "hba1c_None", "hba1c_Norm", "hba1c_high_changed", "hba1c_high_no_change",
    # ── Glucose serum OHE (4-level: None / Norm / >200 / >300) ───────────────
    "glucose_>200", "glucose_>300", "glucose_None", "glucose_Norm",
]

import os as _os

LABEL_NAME = "readmitted_bin"
DATA_FILE  = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "data", "diabetic_data.csv")
SEED       = 42
