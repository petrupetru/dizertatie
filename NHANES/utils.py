import os
import random

import numpy as np
import pandas as pd
import torch

from constants import SEED, FEATURE_NAMES, LABEL_NAME, HOMA_IR_THRESHOLD

# ---------------------------------------------------------------------------
# NHANES survey cycles covered by the paper (2007-2018)
# ---------------------------------------------------------------------------
CYCLES = [
    ("2007-2008", "E"),
    ("2009-2010", "F"),
    ("2011-2012", "G"),
    ("2013-2014", "H"),
    ("2015-2016", "I"),
    ("2017-2018", "J"),
]

# New CDC path (post-2025 migration):
#   https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{start_year}/DataFiles/{FILE}_{SUFFIX}.xpt
# Old path (still tried as fallback):
#   https://wwwn.cdc.gov/Nchs/Nhanes/{year_range}/{FILE}_{SUFFIX}.XPT
_BASE_URL_NEW = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public"
_BASE_URL_OLD = "https://wwwn.cdc.gov/Nchs/Nhanes"

# Files needed per cycle (filename prefix → description)
_CYCLE_FILES = [
    "DEMO",    # Demographics
    "BMX",     # Body measures (BMI, weight)
    "BPX",     # Blood pressure exam
    "BPQ",     # Blood pressure questionnaire
    "GHB",     # Glycohemoglobin
    "INS",     # Insulin (fasting subsample)
    "TRIGLY",  # Triglycerides / LDL
    "HDL",     # HDL cholesterol
    "TCHOL",   # Total cholesterol
    "GLU",     # Fasting glucose
    "BIOPRO",  # Standard biochemistry (creatinine for eGFR)
    "SMQ",     # Smoking questionnaire
    "MCQ",     # Medical conditions questionnaire
    "DIQ",     # Diabetes questionnaire
    "PAQ",     # Physical activity questionnaire
    "RHQ",     # Reproductive health questionnaire
    "DR1TOT",  # Dietary recall day-1 total nutrients
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _assert_finite_tensor(name, tensor):
    if not torch.isfinite(tensor).all():
        nan_count = torch.isnan(tensor).sum().item()
        inf_count = torch.isinf(tensor).sum().item()
        raise ValueError(f"{name} has {nan_count} NaNs and {inf_count} Infs")


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _download_xpt(year_range, filename, suffix, cache_dir):
    """Download one NHANES XPT file and cache it as a pickle."""
    import io
    import requests

    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{filename}_{suffix}.pkl")
    if os.path.exists(cache_path):
        return pd.read_pickle(cache_path)

    # Sentinel for permanently-missing files — avoids retrying on every run.
    skip_path = os.path.join(cache_dir, f"{filename}_{suffix}.not_found")
    if os.path.exists(skip_path):
        return None

    start_year = year_range.split("-")[0]  # "2007-2008" → "2007"

    # Try new URL format first, then fall back to old format.
    urls_to_try = [
        f"{_BASE_URL_NEW}/{start_year}/DataFiles/{filename}_{suffix}.xpt",
        f"{_BASE_URL_OLD}/{year_range}/{filename}_{suffix}.XPT",
    ]

    headers = {"User-Agent": "Mozilla/5.0 (compatible; research/1.0)"}

    for url in urls_to_try:
        print(f"  Downloading {url} ...")
        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()

            # Reject HTML responses (CDC error/redirect pages).
            ct = response.headers.get("Content-Type", "")
            if "html" in ct.lower():
                continue

            # Also reject if the content starts with '<' (HTML fallback check).
            if response.content[:1] == b"<":
                continue

            buf = io.BytesIO(response.content)
            try:
                df = pd.read_sas(buf, format="xport", encoding="utf-8")
            except Exception:
                buf.seek(0)
                df = pd.read_sas(buf, format="xport", encoding="latin-1")

            df.to_pickle(cache_path)
            return df

        except Exception as e:
            print(f"  Warning: could not download {url}: {e}")
            continue

    print(f"  Warning: all URLs failed for {filename}_{suffix}, skipping.")
    # Cache the failure so we don't retry on the next run.
    open(skip_path, "w").close()
    return None


def _load_cycle(year_range, suffix, cache_dir):
    """Download and merge all required files for one NHANES cycle."""
    frames = {}
    for fname in _CYCLE_FILES:
        df = _download_xpt(year_range, fname, suffix, cache_dir)
        if df is not None and "SEQN" in df.columns:
            frames[fname] = df.set_index("SEQN")

    if "DEMO" not in frames:
        print(f"  Warning: DEMO file missing for cycle {year_range}, skipping.")
        return None

    merged = frames["DEMO"]
    for fname, df in frames.items():
        if fname == "DEMO":
            continue
        merged = merged.join(df, how="left", rsuffix=f"_{fname}")

    merged["CYCLE"] = year_range
    return merged.reset_index()


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _compute_egfr(scr, age):
    """CKD-EPI 2021 creatinine equation (female)."""
    kappa = 0.7
    alpha = -0.241
    scr = np.where(scr <= 0, np.nan, scr)
    ratio = scr / kappa
    egfr = (
        142
        * np.where(ratio < 1, ratio ** alpha, ratio ** (-1.200))
        * (0.9938 ** age)
        * 1.012  # female sex factor
    )
    return egfr


def _compute_physical_activity(df):
    """
    Compute a total MET-min/week score from NHANES PAQ variables.
    Uses moderate and vigorous recreational + work activity where available.
    MET values: vigorous = 8, moderate = 4 (standard WHO approximation).
    """
    met = pd.Series(0.0, index=df.index)

    # Vigorous work activity
    if all(c in df.columns for c in ["PAQ650", "PAQ655", "PAQ660"]):
        vig_work = (
            df["PAQ650"].eq(1).astype(float)
            * df["PAQ655"].fillna(0).clip(0, 7)
            * df["PAQ660"].fillna(0).clip(0, 960)
            * 8.0
        )
        met += vig_work

    # Moderate work activity
    if all(c in df.columns for c in ["PAQ620", "PAQ625", "PAQ630"]):
        mod_work = (
            df["PAQ620"].eq(1).astype(float)
            * df["PAQ625"].fillna(0).clip(0, 7)
            * df["PAQ630"].fillna(0).clip(0, 960)
            * 4.0
        )
        met += mod_work

    # Vigorous recreational activity
    if all(c in df.columns for c in ["PAQ635", "PAQ640", "PAQ645"]):
        vig_rec = (
            df["PAQ635"].eq(1).astype(float)
            * df["PAQ640"].fillna(0).clip(0, 7)
            * df["PAQ645"].fillna(0).clip(0, 960)
            * 8.0
        )
        met += vig_rec

    # Moderate recreational activity
    if all(c in df.columns for c in ["PAQ650", "PAQ655", "PAQ660"]):
        # In some cycles PAQ650+ covers moderate recreational
        pass  # already handled above

    if all(c in df.columns for c in ["PAQ665", "PAQ670", "PAQ675"]):
        mod_rec = (
            df["PAQ665"].eq(1).astype(float)
            * df["PAQ670"].fillna(0).clip(0, 7)
            * df["PAQ675"].fillna(0).clip(0, 960)
            * 4.0
        )
        met += mod_rec

    return met


def _binary_yes(series):
    """Recode NHANES yes/no (1=yes, 2=no) to 1/0."""
    return (series == 1).astype(float)


def preprocess_nhanes(df):
    """
    Filter cohort, engineer features, and produce the final DataFrame
    with columns FEATURE_NAMES + LABEL_NAME.

    Cohort criteria (matching the paper):
      - Female (RIAGENDR == 2)
      - Age 45-64 (RIDAGEYR)
      - Non-diabetic (DIQ010 != 1)
      - No cancer (MCQ220 != 1)
      - Has fasting glucose, insulin, and triglycerides (required for HOMA-IR
        and cohort eligibility per the paper)
      - Has dietary recall data

    Returns a DataFrame ready for model training.
    """
    df = df.copy()

    # ---- Cohort filtering ----
    df = df[df["RIAGENDR"] == 2]                           # female
    df = df[df["RIDAGEYR"].between(45, 64)]                # age 45-64
    df = df[df["DIQ010"] != 1]                             # exclude diabetics
    df = df[df.get("MCQ220", pd.Series(2, index=df.index)) != 1]  # exclude cancer
    df = df.dropna(subset=["LBXGLU", "LBXIN", "LBXTR"])  # required labs
    df = df.dropna(subset=["DR1TKCAL"])                    # required daily nutrient intake
    print(f"  Cohort: {len(df)} samples after all exclusions")

    # ---- HOMA-IR label ----
    homa_ir = (df["LBXIN"] * df["LBXGLU"]) / 405.0
    df[LABEL_NAME] = (homa_ir > HOMA_IR_THRESHOLD).astype(int)

    # ---- HDL: harmonise variable name across cycles ----
    if "LBXHDL" in df.columns and "LBDHDD" not in df.columns:
        df["LBDHDD"] = df["LBXHDL"]
    elif "LBDHDD" not in df.columns:
        df["LBDHDD"] = np.nan

    # ---- eGFR ----
    if "LBXSCR" in df.columns:
        df["EGFR"] = _compute_egfr(df["LBXSCR"].values, df["RIDAGEYR"].values)
    else:
        df["EGFR"] = np.nan

    # ---- Nutrition features ----
    _dr1tkcal = df.get("DR1TKCAL", pd.Series(np.nan, index=df.index))
    _bmxwt    = df.get("BMXWT",    pd.Series(np.nan, index=df.index))
    _dr1tprot = df.get("DR1TPROT", pd.Series(np.nan, index=df.index))
    df["ENERGY_PER_KG"] = np.where(_bmxwt.gt(0), _dr1tkcal / _bmxwt, np.nan)
    df["PROTEIN_RATIO"] = np.where(_dr1tkcal.gt(0), (_dr1tprot * 4.0) / _dr1tkcal, np.nan)

    # ---- Physical activity ----
    df["PHYSICAL_ACTIVITY"] = _compute_physical_activity(df)

    # ---- Hypertension (self-report OR medication OR BP threshold) ----
    hypert = (
        df.get("BPQ020", pd.Series(2, index=df.index)).eq(1)
        | df.get("BPD132", pd.Series(2, index=df.index)).eq(1)
        | df.get("BPXSY1", pd.Series(0.0, index=df.index)).ge(140)
        | df.get("BPXDI1", pd.Series(0.0, index=df.index)).ge(90)
    )
    df["HYPERTENSION"] = hypert.astype(float)

    # ---- Race one-hot (reference = Non-Hispanic White, code 3) ----
    race = df.get("RIDRETH3", pd.Series(np.nan, index=df.index))
    df["RACE_NHB"]   = (race == 4).astype(float)
    df["RACE_HISP"]  = race.isin([1, 2]).astype(float)
    df["RACE_OTHER"] = race.isin([6, 7]).astype(float)

    # ---- Education (1 = college or above, reference = HS or below) ----
    edu = df.get("DMDEDUC2", pd.Series(np.nan, index=df.index))
    df["EDU_COLLEGE"] = edu.isin([4, 5]).astype(float)

    # ---- Marital status (reference = married / living with partner) ----
    marital = df.get("DMDMARTL", pd.Series(np.nan, index=df.index))
    df["MARITAL_WID_DIV_SEP"] = marital.isin([2, 3, 4]).astype(float)
    df["MARITAL_NEVER"]       = (marital == 5).astype(float)

    # ---- Smoking (reference = never smoked) ----
    smoked_ever = df.get("SMQ020", pd.Series(2, index=df.index))
    smokes_now  = df.get("SMQ040", pd.Series(3, index=df.index))
    df["SMOKING_CURRENT"] = (smoked_ever.eq(1) & smokes_now.isin([1, 2])).astype(float)
    df["SMOKING_FORMER"]  = (smoked_ever.eq(1) & smokes_now.eq(3)).astype(float)

    # ---- Family history of diabetes ----
    df["FAMILY_DM_HISTORY"] = _binary_yes(
        df.get("MCQ300C", pd.Series(2, index=df.index))
    )

    # ---- Reproductive health ----
    df["AGE_MENARCHE"]      = df.get("RHQ010",  pd.Series(np.nan, index=df.index))
    df["NUM_PREGNANCIES"]   = df.get("RHQ160",  pd.Series(np.nan, index=df.index))

    # Abortions/miscarriages/stillbirths: pregnancies minus live births
    preg      = df.get("RHQ160", pd.Series(np.nan, index=df.index))
    live      = df.get("RHQ171", df.get("RHQ170", pd.Series(np.nan, index=df.index)))
    df["NUM_ABORTIONS_MISC"] = (preg - live).clip(lower=0)

    df["BIRTH_CONTROL"]  = _binary_yes(df.get("RHQ420", pd.Series(2, index=df.index)))
    df["HYSTERECTOMY"]   = _binary_yes(df.get("RHQ360", pd.Series(2, index=df.index)))
    df["OOPHORECTOMY"]   = _binary_yes(df.get("RHQ380", pd.Series(2, index=df.index)))

    # Female hormones: RHQ540 in earlier cycles, RHQ554 in later ones
    fem_horm = df.get("RHQ540", df.get("RHQ554", pd.Series(2, index=df.index)))
    df["FEMALE_HORMONES"] = _binary_yes(fem_horm)

    # ---- Rename INDFMPIR and BMX/BP columns (already have the right names) ----
    # BMXBMI, BPXSY1, BPXDI1, INDFMPIR, RIDAGEYR, LBXGLU, LBXTR, LBXGH,
    # LBXTC, DR1TSUGR, DR1TFIBE, DR1TCHOL, DR1TSFAT, DR1TALCO, DR1TVC,
    # DR1TVD, DR1TVB6, DR1TVB12, DR1TCAFF, DR1TIRON, DR1TCALC, DR1TZINC,
    # DR1TSODI, DR1TPHOS, DR1TCOPP, DR1TSELE, DR1TPOTA, DR1TTHEO
    # are already using their NHANES variable names defined in FEATURE_NAMES.

    # ---- Fill missing poverty index and physical activity with mean ----
    for col in ["INDFMPIR", "PHYSICAL_ACTIVITY"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mean())

    # ---- Select only the columns we need ----
    available = [c for c in FEATURE_NAMES if c in df.columns]
    missing   = [c for c in FEATURE_NAMES if c not in df.columns]
    if missing:
        print(f"Warning: the following FEATURE_NAMES are missing and will be NaN: {missing}")
        for c in missing:
            df[c] = np.nan

    out = df[FEATURE_NAMES + [LABEL_NAME]].copy()
    out[FEATURE_NAMES] = out[FEATURE_NAMES].apply(pd.to_numeric, errors="coerce")
    out = out.reset_index(drop=True)

    # ---- kNN imputation (paper: "remaining missing values imputed using kNN") ----
    from sklearn.impute import KNNImputer
    imputer = KNNImputer(n_neighbors=5)
    out[FEATURE_NAMES] = imputer.fit_transform(out[FEATURE_NAMES])

    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def load_nhanes_data(data_dir="NHANES/data"):
    """
    Load (and if needed download) NHANES 2007-2018 data, apply cohort
    filtering and feature engineering, and return the final DataFrame.

    Parameters
    ----------
    data_dir : str
        Directory used as both the XPT download cache and the location of the
        processed CSV (nhanes_processed.csv).

    Returns
    -------
    pd.DataFrame  with columns FEATURE_NAMES + LABEL_NAME, n ≈ 2084.
    """
    processed_csv = os.path.join(data_dir, "nhanes_processed.csv")

    if os.path.exists(processed_csv):
        print(f"Loading cached processed data from {processed_csv}")
        df = pd.read_csv(processed_csv)
        return df

    print("Downloading and processing NHANES 2007-2018 ...")
    cycle_frames = []
    for year_range, suffix in CYCLES:
        print(f"\n--- Cycle {year_range} ---")
        frame = _load_cycle(year_range, suffix, data_dir)
        if frame is not None:
            cycle_frames.append(frame)

    if not cycle_frames:
        raise RuntimeError("No NHANES cycle data could be loaded.")

    raw = pd.concat(cycle_frames, ignore_index=True)
    df  = preprocess_nhanes(raw)

    os.makedirs(data_dir, exist_ok=True)
    df.to_csv(processed_csv, index=False)
    print(f"\nProcessed dataset saved to {processed_csv}")
    print(f"Total samples: {len(df)}  |  IR=1: {df[LABEL_NAME].sum()}  |  IR=0: {(df[LABEL_NAME] == 0).sum()}")
    return df
