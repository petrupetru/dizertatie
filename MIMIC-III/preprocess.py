"""
MIMIC-III preprocessing replicating the ISeeU paper pipeline.
Translates the dataset_sapsii_mv.sql logic to pandas and saves pickled output.
"""
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data', 'mimic-iii-clinical-database')
OUT_DIR   = os.path.dirname(__file__)

INPUTS_FILE  = os.path.join(OUT_DIR, 'MIMICIII_sapsii_dataset_inputs.pickle')
LABELS_FILE  = os.path.join(OUT_DIR, 'MIMICIII_sapsii_dataset_mortality_labels.pickle')


# ── helpers ──────────────────────────────────────────────────────────────────

def _between_str(series, lo, hi):
    """Lexicographic string BETWEEN (replicates PostgreSQL varchar BETWEEN)."""
    return (series >= lo) & (series <= hi)


def _aids_flag(codes):
    return _between_str(codes, '042', '0449').any()


def _metastatic_flag(codes):
    return (
        _between_str(codes, '1960', '1991') |
        _between_str(codes, '20970', '20975') |
        (codes == '20979') |
        (codes == '78951')
    ).any()


def _lymphoma_flag(codes):
    return (
        _between_str(codes, '20000', '20238') |
        _between_str(codes, '20240', '20248') |
        _between_str(codes, '20250', '20302') |
        _between_str(codes, '20310', '20312') |
        _between_str(codes, '20302', '20382') |
        _between_str(codes, '20400', '20522') |
        _between_str(codes, '20580', '20702') |
        _between_str(codes, '20720', '20892') |
        (codes == '2386 ') |
        (codes == '2733 ')
    ).any()


# ── load raw tables ───────────────────────────────────────────────────────────

CHUNK_SIZE = 500_000


def _read_chunked(path, usecols_set, item_col, item_ids, date_col):
    """Read a large CSV in chunks, keeping only rows whose item_col is in item_ids."""
    usecols = lambda c: c.upper() in usecols_set
    chunks  = []
    file_bytes = os.path.getsize(path)
    pbar = tqdm(unit='chunk', unit_scale=True,
                desc=f'  {os.path.basename(path)}')
    bytes_read = 0
    for chunk in pd.read_csv(path, chunksize=CHUNK_SIZE, usecols=usecols, low_memory=False):
        chunk.columns = chunk.columns.str.lower()
        chunk = chunk[chunk[item_col].isin(item_ids)]
        if not chunk.empty:
            chunks.append(chunk)
        bytes_read += CHUNK_SIZE * 80  # rough byte estimate per row for display
        pbar.update(1)
    pbar.close()
    df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    df[date_col] = pd.to_datetime(df[date_col])
    return df


def load_raw():
    def p(name):
        return os.path.join(DATA_DIR, name)

    def lc(df):
        df.columns = df.columns.str.lower()
        return df

    print("  Reading small tables...")
    icustays   = lc(pd.read_csv(p('ICUSTAYS.csv')))
    icustays[['intime','outtime']] = icustays[['intime','outtime']].apply(pd.to_datetime)

    patients   = lc(pd.read_csv(p('PATIENTS.csv')))
    patients[['dob','dod']] = patients[['dob','dod']].apply(pd.to_datetime)

    admissions = lc(pd.read_csv(p('ADMISSIONS.csv')))
    admissions[['admittime','dischtime','deathtime']] = admissions[['admittime','dischtime','deathtime']].apply(pd.to_datetime)

    diagnoses  = lc(pd.read_csv(p('DIAGNOSES_ICD.csv')))
    services   = lc(pd.read_csv(p('SERVICES.csv')))

    print("  Reading CHARTEVENTS (chunked, filtering to relevant item IDs)...")
    chart_items = set(CHART_ITEM_MAP.keys()) | FIO2_CHART_ITEMS
    chartevents = _read_chunked(
        p('CHARTEVENTS.csv'),
        usecols_set={'SUBJECT_ID','HADM_ID','ICUSTAY_ID','ITEMID','CHARTTIME','VALUENUM','ERROR'},
        item_col='itemid', item_ids=chart_items, date_col='charttime')

    print("  Reading LABEVENTS (chunked, filtering to relevant item IDs)...")
    labevents = _read_chunked(
        p('LABEVENTS.csv'),
        usecols_set={'SUBJECT_ID','HADM_ID','ITEMID','CHARTTIME','VALUENUM'},
        item_col='itemid', item_ids=set(LAB_ITEM_MAP.keys()), date_col='charttime')

    print("  Reading OUTPUTEVENTS (chunked, filtering to relevant item IDs)...")
    outputevents = _read_chunked(
        p('OUTPUTEVENTS.csv'),
        usecols_set={'SUBJECT_ID','HADM_ID','ICUSTAY_ID','ITEMID','CHARTTIME','VALUE','ISERROR'},
        item_col='itemid', item_ids=URINE_ITEMS, date_col='charttime')

    return icustays, patients, admissions, diagnoses, services, chartevents, labevents, outputevents


# ── build cohort ──────────────────────────────────────────────────────────────

def build_cohort(icustays, patients, admissions):
    df = icustays.merge(patients[['subject_id','dob']], on='subject_id')
    df = df.merge(admissions[['subject_id','hadm_id','admission_type','deathtime']], on=['subject_id','hadm_id'])

    df['age'] = (df['intime'] - df['dob']).dt.total_seconds() / (365.242 * 24 * 3600)
    df['corrected_age'] = df['age'].clip(upper=80.0)
    df['elective'] = (df['admission_type'] == 'ELECTIVE').astype(int)
    df['mort_icu']  = (
        df['deathtime'].notna() &
        (df['deathtime'] >= df['intime']) &
        (df['deathtime'] <= df['outtime'])
    ).astype(int)

    # rank by intime per subject → keep first stay
    df['rank'] = df.groupby('subject_id')['intime'].rank(method='first').astype(int)
    cohort = df[(df['rank'] == 1) & (df['los'] >= 2) & (df['age'] > 16)].copy()
    return cohort[['subject_id','hadm_id','icustay_id','intime','outtime',
                   'corrected_age','elective','mort_icu']].reset_index(drop=True)


# ── per-admission static flags ────────────────────────────────────────────────

def build_diagnosis_flags(diagnoses, cohort_hadm_ids):
    diag = diagnoses[diagnoses['hadm_id'].isin(cohort_hadm_ids)].copy()
    diag['icd9_code'] = diag['icd9_code'].astype(str).str.strip()
    flags = diag.groupby('hadm_id')['icd9_code'].agg(
        aids=_aids_flag,
        metastatic_cancer=_metastatic_flag,
        lymphoma=_lymphoma_flag
    ).astype(int).reset_index()
    return flags


def build_surgical_flag(services, cohort_hadm_ids):
    svc = services[services['hadm_id'].isin(cohort_hadm_ids)].copy()
    svc['surgical'] = svc['curr_service'].str.lower().str.contains('surg', na=False).astype(int)
    return svc.groupby('hadm_id')['surgical'].max().reset_index()


# ── chartevents ───────────────────────────────────────────────────────────────

CHART_ITEM_MAP = {
    # GCS
    723:    ('GCSVerbal', 1,   5,   None),
    223900: ('GCSVerbal', 1,   5,   None),
    454:    ('GCSMotor',  1,   6,   None),
    223901: ('GCSMotor',  1,   6,   None),
    184:    ('GCSEyes',   1,   6,   None),
    220739: ('GCSEyes',   1,   6,   None),
    # Heart rate
    211:    ('HEART RATE',   0, 300, None),
    220045: ('HEART RATE',   0, 300, None),
    # Systolic BP
    51:     ('SYSTOLIC BP',  0, 400, None),
    442:    ('SYSTOLIC BP',  0, 400, None),
    455:    ('SYSTOLIC BP',  0, 400, None),
    6701:   ('SYSTOLIC BP',  0, 400, None),
    220179: ('SYSTOLIC BP',  0, 400, None),
    220050: ('SYSTOLIC BP',  0, 400, None),
    # Diastolic BP
    8368:   ('DIASTOLIC BP', 0, 300, None),
    8440:   ('DIASTOLIC BP', 0, 300, None),
    8441:   ('DIASTOLIC BP', 0, 300, None),
    8555:   ('DIASTOLIC BP', 0, 300, None),
    220180: ('DIASTOLIC BP', 0, 300, None),
    220051: ('DIASTOLIC BP', 0, 300, None),
    # Temperature (Fahrenheit → Celsius)
    223761: ('TEMPERATURE',  70,  120, lambda v: (v - 32) / 1.8),
    678:    ('TEMPERATURE',  70,  120, lambda v: (v - 32) / 1.8),
    # Temperature (Celsius)
    223762: ('TEMPERATURE',  10,  50,  None),
    676:    ('TEMPERATURE',  10,  50,  None),
}

FIO2_CHART_ITEMS = {223835, 3420, 3422, 190}


def _process_fio2_chart(row):
    v = row['valuenum']
    if row['itemid'] == 223835:
        if 0 < v <= 1:
            return v * 100
        elif 1 < v < 21:
            return np.nan
        elif 21 <= v <= 100:
            return v
        else:
            return np.nan
    elif row['itemid'] in (3420, 3422):
        return v
    elif row['itemid'] == 190 and 0.20 < v < 1:
        return v * 100
    return np.nan


def extract_chartevents(chartevents, cohort):
    all_items = set(CHART_ITEM_MAP.keys()) | FIO2_CHART_ITEMS
    ce = chartevents[
        chartevents['icustay_id'].isin(cohort['icustay_id']) &
        chartevents['itemid'].isin(all_items) &
        (chartevents['error'] != 1)
    ].copy()

    ce = ce.merge(cohort[['subject_id','icustay_id','intime']], on=['subject_id','icustay_id'])
    ce['icu_time_hr'] = (ce['charttime'] - ce['intime']).dt.total_seconds() / 3600
    ce = ce[(ce['icu_time_hr'] >= 0) & (ce['icu_time_hr'] < 48)]

    rows = []
    # standard items
    sub = ce[ce['itemid'].isin(CHART_ITEM_MAP)]
    for item_id, (name, lo, hi, transform) in CHART_ITEM_MAP.items():
        mask = sub['itemid'] == item_id
        chunk = sub[mask].copy()
        chunk = chunk[(chunk['valuenum'] > lo) & (chunk['valuenum'] < hi)]
        if transform:
            chunk['valuenum'] = chunk['valuenum'].apply(transform)
        chunk['measurement_name'] = name
        rows.append(chunk[['subject_id','icu_time_hr','measurement_name','valuenum']])

    # FiO2 items
    fio2_sub = ce[ce['itemid'].isin(FIO2_CHART_ITEMS)].copy()
    fio2_sub['valuenum'] = fio2_sub.apply(_process_fio2_chart, axis=1)
    fio2_sub = fio2_sub.dropna(subset=['valuenum'])
    fio2_sub['measurement_name'] = 'FiO2'
    rows.append(fio2_sub[['subject_id','icu_time_hr','measurement_name','valuenum']])

    return pd.concat(rows, ignore_index=True).rename(columns={'valuenum': 'value'})


# ── labevents ─────────────────────────────────────────────────────────────────

LAB_ITEM_MAP = {
    50882:  ('BICARBONATE',  None,   10000),
    50885:  ('BILIRUBIN',    None,   150),
    50971:  ('POTASSIUM',    None,   30),
    50822:  ('POTASSIUM',    None,   30),
    50983:  ('SODIUM',       None,   200),
    50824:  ('SODIUM',       None,   200),
    950824: ('SODIUM',       None,   200),
    51006:  ('BUN',          None,   300),
    51301:  ('WBC',          None,   1000),
    51300:  ('WBC',          None,   1000),
    50821:  ('PO2',          None,   800),
    50816:  ('FiO2',         None,   100),
}


def extract_labevents(labevents, cohort):
    le = labevents[
        labevents['subject_id'].isin(cohort['subject_id']) &
        labevents['itemid'].isin(LAB_ITEM_MAP) &
        labevents['valuenum'].notna() &
        (labevents['valuenum'] > 0)
    ].copy()

    le = le.merge(cohort[['subject_id','hadm_id','intime']], on=['subject_id','hadm_id'])
    le['icu_time_hr'] = (le['charttime'] - le['intime']).dt.total_seconds() / 3600
    le = le[(le['icu_time_hr'] >= 0) & (le['icu_time_hr'] < 48)]

    rows = []
    for item_id, (name, lo, hi) in LAB_ITEM_MAP.items():
        mask = le['itemid'] == item_id
        chunk = le[mask].copy()
        if hi is not None:
            chunk = chunk[chunk['valuenum'] <= hi]
        chunk['measurement_name'] = name
        rows.append(chunk[['subject_id','icu_time_hr','measurement_name','valuenum']])

    return pd.concat(rows, ignore_index=True).rename(columns={'valuenum': 'value'})


# ── outputevents ──────────────────────────────────────────────────────────────

URINE_ITEMS = {
    40055, 43175, 40069, 40094, 40715, 40473, 40085, 40057, 40056, 40405, 40428,
    40086, 40096, 40651,
    226559, 226560, 226561, 226584, 226563, 226564, 226565,
    226567, 226557, 226558, 227488, 227489
}
GU_IRRIGANT = 227488


def extract_outputevents(outputevents, cohort):
    oe = outputevents[
        outputevents['icustay_id'].isin(cohort['icustay_id']) &
        outputevents['itemid'].isin(URINE_ITEMS) &
        (outputevents['iserror'] != 1) &
        (outputevents['value'] < 5000)
    ].copy()

    oe = oe.merge(cohort[['subject_id','icustay_id','intime']], on=['subject_id','icustay_id'])
    oe['icu_time_hr'] = (oe['charttime'] - oe['intime']).dt.total_seconds() / 3600
    oe = oe[(oe['icu_time_hr'] >= 0) & (oe['icu_time_hr'] < 48)]

    oe['value'] = np.where(oe['itemid'] == GU_IRRIGANT, -oe['value'], oe['value'])
    oe['measurement_name'] = 'URINE OUTPUT'
    return oe[['subject_id','icu_time_hr','measurement_name','value']]


# ── pivot + static series ─────────────────────────────────────────────────────

def build_pivot(measurements, urine, cohort, diag_flags, surg_flags):
    """
    Returns sapsII_inputs: MultiIndex DataFrame (subject_id, measurement_name) × hour 0..47
    and mortality_labels: DataFrame with columns [subject_id, mort_icu]
    """
    # Combine measurements
    combined = pd.concat([measurements, urine], ignore_index=True)
    combined['measurement_hour'] = np.floor(combined['icu_time_hr']).astype(int)
    combined = combined[combined['measurement_hour'] != 48]  # match original filter

    # Pivot: mean for everything, sum for URINE
    non_urine = combined[combined['measurement_name'] != 'URINE OUTPUT']
    urine_df  = combined[combined['measurement_name'] == 'URINE OUTPUT']

    mean_pivot = non_urine.groupby(['subject_id','measurement_name','measurement_hour'])['value'].mean().unstack('measurement_hour')
    urine_pivot = urine_df.groupby(['subject_id','measurement_name','measurement_hour'])['value'].sum().unstack('measurement_hour')

    inputs = pd.concat([mean_pivot, urine_pivot])

    # ── add static time series ──────────────────────────────────────────────
    # Build static per-subject values
    cohort_static = cohort[['subject_id','hadm_id','corrected_age','elective','mort_icu']].copy()
    cohort_static = cohort_static.merge(diag_flags, on='hadm_id', how='left').fillna({'aids':0,'metastatic_cancer':0,'lymphoma':0})
    cohort_static = cohort_static.merge(surg_flags, on='hadm_id', how='left').fillna({'surgical':0})

    hours = list(range(48))

    def static_series(subject_ids, values, name):
        idx = pd.MultiIndex.from_arrays([subject_ids, [name]*len(subject_ids)], names=['subject_id','measurement_name'])
        df = pd.DataFrame(np.tile(values[:,None], (1, 48)), index=idx, columns=hours)
        return df

    s = cohort_static.set_index('subject_id')
    ages    = static_series(s.index, s['corrected_age'].values, 'AGE')
    surg    = static_series(s.index, s['surgical'].values,      'SURGICAL')
    elec    = static_series(s.index, s['elective'].values,      'ELECTIVE')
    aids    = static_series(s.index, s['aids'].values,          'AIDS')
    meta    = static_series(s.index, s['metastatic_cancer'].values, 'METASTATIC_CANCER')
    lymph   = static_series(s.index, s['lymphoma'].values,      'LYMPHOMA')

    sapsII_inputs = pd.concat([inputs, ages, surg, elec, aids, meta, lymph]).sort_index()

    # ── reindex so every (subject, measurement) combination exists ──────────
    subject_ids   = sapsII_inputs.index.get_level_values('subject_id').unique().sort_values()
    meas_names    = sapsII_inputs.index.get_level_values('measurement_name').unique().sort_values()
    new_index     = pd.MultiIndex.from_product([subject_ids, meas_names], names=['subject_id','measurement_name'])
    sapsII_inputs = sapsII_inputs.reindex(new_index)

    # Make sure columns are 0..47 (reindex columns too)
    sapsII_inputs = sapsII_inputs.reindex(columns=list(range(48)))

    mortality_labels = cohort_static[['subject_id','mort_icu']].drop_duplicates('subject_id')
    mortality_labels = mortality_labels.set_index('subject_id').reindex(subject_ids).reset_index()

    return sapsII_inputs, mortality_labels


# ── imputation (as in ISeeU notebook) ────────────────────────────────────────

def impute(df):
    def sel(name):
        return df.index.get_level_values('measurement_name') == name

    for gcs in ['GCSEyes', 'GCSMotor', 'GCSVerbal']:
        df.loc[sel(gcs)] = df.loc[sel(gcs)].ffill(axis=1)

    df.loc[sel('FiO2')] = df.loc[sel('FiO2')].ffill(axis=1).fillna(0.2)
    df.loc[sel('PO2')]  = df.loc[sel('PO2')].bfill(axis=1).ffill(axis=1)

    df = df.ffill(axis=1).bfill(axis=1)
    return df


# ── main ──────────────────────────────────────────────────────────────────────

def build_dataset(force=False):
    if not force and os.path.exists(INPUTS_FILE) and os.path.exists(LABELS_FILE):
        print("Loading cached pickles...")
        return pd.read_pickle(INPUTS_FILE), pd.read_pickle(LABELS_FILE)

    print("Loading raw CSV files...")
    icustays, patients, admissions, diagnoses, services, chartevents, labevents, outputevents = load_raw()

    print("Building cohort...")
    cohort = build_cohort(icustays, patients, admissions)
    print(f"  Cohort size: {len(cohort)} patients")

    print("Building diagnosis flags...")
    diag_flags = build_diagnosis_flags(diagnoses, cohort['hadm_id'])
    surg_flags = build_surgical_flag(services, cohort['hadm_id'])

    print("Extracting chartevents...")
    chart_meas = extract_chartevents(chartevents, cohort)
    print(f"  Chart measurements: {len(chart_meas)}")

    print("Extracting labevents...")
    lab_meas = extract_labevents(labevents, cohort)
    print(f"  Lab measurements: {len(lab_meas)}")

    print("Extracting output events (urine)...")
    urine_meas = extract_outputevents(outputevents, cohort)
    print(f"  Urine measurements: {len(urine_meas)}")

    measurements = pd.concat([chart_meas, lab_meas], ignore_index=True)

    print("Building pivot table...")
    sapsII_inputs, mortality_labels = build_pivot(measurements, urine_meas, cohort, diag_flags, surg_flags)
    print(f"  Input shape before imputation: {sapsII_inputs.shape}")
    print(f"  Measurement names: {sapsII_inputs.index.get_level_values('measurement_name').unique().tolist()}")

    print("Imputing...")
    sapsII_inputs = impute(sapsII_inputs)

    print("Saving pickles...")
    sapsII_inputs.to_pickle(INPUTS_FILE)
    mortality_labels.to_pickle(LABELS_FILE)

    print("Done.")
    return sapsII_inputs, mortality_labels


if __name__ == '__main__':
    inputs, labels = build_dataset(force=True)
    print(f"\nFinal input shape:  {inputs.shape}")
    print(f"Final labels shape: {labels.shape}")
    print(f"Mortality rate: {labels['mort_icu'].mean():.3f}")
    print(f"NaN count: {inputs.isna().sum().sum()}")
