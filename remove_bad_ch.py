#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
remove_bad_channels.py

Load the bad_channel list for each subject from patients_gray_matter.mat,
then for each subject CSV (subject_01.csv...subject_20.csv) remove rows
whose 'channel' is in the bad_channel list.
"""

import os
import numpy as np
import pandas as pd
import scipy.io as sio

# === Configuration ===
# Path to the MAT file containing 'patients' struct with bad_channel field
patients_mat_path = "/home/yuantao/Downloads/files/patients_gray_matter.mat"
# Directory where per-subject CSVs are stored
csv_dir = "n_since_last_click_by_channel"
# Number of subjects
n_subjects = 20

def load_patients(patients_mat_path):
    """
    Load the 'patients' struct array from the given .mat file.

    Returns
    -------
    list of dict
        Each dict has at least 'n_subject' and 'bad_channel' keys.
    """
    mat = sio.loadmat(patients_mat_path, struct_as_record=False, squeeze_me=True)
    patients = mat.get('patients')
    # normalize to Python list
    if isinstance(patients, np.ndarray):
        patients = patients.flatten().tolist()
    else:
        patients = [patients]

    # convert each struct to a dictionary for easier access
    patient_list = []
    for p in patients:
        subj_id = int(getattr(p, 'n_subject', None) or getattr(p, 'subject', None))
        bad = np.atleast_1d(getattr(p, 'bad_channel', [])).astype(int).ravel().tolist()
        patient_list.append({'subject': subj_id, 'bad_channel': bad})
    return patient_list

if __name__ == "__main__":
    # Load patients info
    patients = load_patients(patients_mat_path)
    # Build a map subject -> bad_channel list
    subj_to_bad = {p['subject']: p['bad_channel'] for p in patients}

    # Process each subject CSV
    for subj in range(1, n_subjects + 1):
        csv_path = os.path.join(csv_dir, f"subject_{subj:02d}.csv")
        if not os.path.isfile(csv_path):
            print(f"[Subject {subj:02d}] CSV not found at {csv_path}, skipping.")
            continue

        bad_channels = subj_to_bad.get(subj, [])
        if not bad_channels:
            print(f"[Subject {subj:02d}] No bad channels defined, no filtering needed.")
            continue

        df = pd.read_csv(csv_path)
        orig_count = len(df)

        # Remove rows where 'channel' is in bad_channels
        df_clean = df[~df['channel'].isin(bad_channels)]
        new_count = len(df_clean)

        # Overwrite CSV
        df_clean.to_csv(csv_path, index=False)
        print(f"[Subject {subj:02d}] Removed {orig_count - new_count} rows "
              f"({len(bad_channels)} bad channels), saved {new_count} rows.")
