import os

MIMIC_3_DIR = 'data/raw'
DATA_DIR = 'data/processed'
DATA_COMPRESSION_GZ = True
MODEL_DIR = 'models/'


class Tables:
    if DATA_COMPRESSION_GZ:
        format_suffix = ".gz"
    else:
        format_suffix = ''
    note_events = os.path.join(DATA_DIR, "NOTEEVENTS.csv"+format_suffix)
    diagnoses_icd = os.path.join(DATA_DIR, "DIAGNOSES_ICD.csv"+format_suffix)
    procedures = os.path.join(DATA_DIR, "PROCEDURES_ICD.csv"+format_suffix)
    lab_events = os.path.join(DATA_DIR, "LABEVENTS.csv"+format_suffix)
    admissions = os.path.join(DATA_DIR, "ADMISSIONS.csv"+format_suffix)


class Keys:
    admission_id = "HADM_ID"
    icd9 = "ICD9_CODE"
    note_category = "CATEGORY"
    text = "TEXT"


PAD_CHAR = "**PAD**"
EMBEDDING_SIZE = 100
MAX_LENGTH = 2500
