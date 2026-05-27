SEED = 42
HOMA_IR_THRESHOLD = 2.73
LABEL_NAME = "IR"

# Feature columns as they appear in the preprocessed DataFrame.
# 16 demographic/behavioral (after one-hot expanding race, education,
# marital status, and smoking) + 6 lab + 21 nutrition + 7 reproductive = 50.
# The paper counts 48 original predictor variables before one-hot expansion;
# the slight column-count difference is expected.
FEATURE_NAMES = [
    # --- Demographic & Behavioral (16 columns) ---
    "RIDAGEYR",             # Age at screening
    "RACE_NHB",             # Non-Hispanic Black (vs. Non-Hispanic White ref.)
    "RACE_HISP",            # Hispanic (Mexican-American + Other Hispanic)
    "RACE_OTHER",           # Other / Multi-racial
    "EDU_COLLEGE",          # College or above (vs. HS or below ref.)
    "MARITAL_WID_DIV_SEP",  # Widowed / Divorced / Separated
    "MARITAL_NEVER",        # Never married (vs. Married/Partner ref.)
    "INDFMPIR",             # Family monthly poverty level index
    "SMOKING_CURRENT",      # Current smoker (vs. never ref.)
    "SMOKING_FORMER",       # Former smoker
    "FAMILY_DM_HISTORY",    # Close relative had diabetes
    "BMXBMI",               # Body-mass index (kg/m²)
    "PHYSICAL_ACTIVITY",    # Physical activity (MET-min/week, computed)
    "HYPERTENSION",         # Hypertension (self-report, meds, or BP ≥ threshold)
    "BPXSY1",               # Systolic blood pressure (mmHg)
    "BPXDI1",               # Diastolic blood pressure (mmHg)
    # --- Laboratory (6 columns) ---
    "LBXGLU",               # Fasting plasma glucose (mg/dL)
    "LBDHDD",               # HDL cholesterol (mg/dL)
    "LBXTR",                # Triglycerides (mg/dL)
    "LBXGH",                # Glycohemoglobin / HbA1c (%)
    "LBXTC",                # Total cholesterol (mg/dL)
    "EGFR",                 # Estimated GFR (mL/min/1.73 m², CKD-EPI female)
    # --- Daily Nutritional Intake (21 columns) ---
    "ENERGY_PER_KG",        # Energy intake (kcal / kg body weight)
    "PROTEIN_RATIO",        # Protein intake ratio (protein kcal / total kcal)
    "DR1TSUGR",             # Total sugars (g)
    "DR1TFIBE",             # Dietary fiber (g)
    "DR1TCHOL",             # Cholesterol intake (mg)
    "DR1TSFAT",             # Saturated fatty acid (g)
    "DR1TALCO",             # Alcohol (g)
    "DR1TVC",               # Vitamin C (mg)
    "DR1TVD",               # Vitamin D (mcg)
    "DR1TVB6",              # Vitamin B6 (mg)
    "DR1TVB12",             # Vitamin B12 (mcg)
    "DR1TCAFF",             # Caffeine (mg)
    "DR1TIRON",             # Iron (mg)
    "DR1TCALC",             # Calcium (mg)
    "DR1TZINC",             # Zinc (mg)
    "DR1TSODI",             # Sodium (mg)
    "DR1TPHOS",             # Phosphorus (mg)
    "DR1TCOPP",             # Copper (mg)
    "DR1TSELE",             # Selenium (mcg)
    "DR1TPOTA",             # Potassium (mg)
    "DR1TTHEO",             # Theobromine (mg)
    # --- Reproductive Health (7 columns) ---
    "AGE_MENARCHE",         # Age at first menstrual period (years)
    "NUM_PREGNANCIES",      # Number of times pregnant
    "NUM_ABORTIONS_MISC",   # Abortions / miscarriages / stillbirths
    "BIRTH_CONTROL",        # Ever used birth control pills (0/1)
    "HYSTERECTOMY",         # Had hysterectomy (0/1)
    "OOPHORECTOMY",         # Both ovaries removed (0/1)
    "FEMALE_HORMONES",      # Took female hormones (0/1)
]
