#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
export_n_since_last_click_all_subjects.py

This script iterates over subject files b_data_subject_1.mat through b_data_subject_20.mat,
loads the 'n_since_last_click' array from each, adds a serial index, filters to odd indices,
and exports a combined CSV with columns: subject, index, n_since_last_click.
"""

import os
import scipy.io as sio
import numpy as np
import pandas as pd

# === Modify these paths to suit your setup ===
mat_dir = r"/home/yuantao/Downloads/files/P00_data_behavior"
output_csv_path = "n_since_last_click_all_subjects.csv"

def load_n_since_last_click(mat_path):
    """
    Load the 'n_since_last_click' array from the given .mat file.

    Parameters
    ----------
    mat_path : str
        Full path to the .mat file.

    Returns
    -------
    numpy.ndarray
        1D array of the n_since_last_click values.
    """
    if not os.path.isfile(mat_path):
        raise FileNotFoundError(f"File not found: {mat_path}")

    mat = sio.loadmat(mat_path, struct_as_record=False, squeeze_me=True)
    b_data = mat.get('b_data')
    if b_data is None:
        raise KeyError("'b_data' not found in the MAT file.")

    # ensure iterable
    entries = b_data if isinstance(b_data, (list, np.ndarray)) else [b_data]

    for entry in entries:
        if getattr(entry, 'name', None) == 'n_since_last_click':
            data = getattr(entry, 'x', None)
            if data is None:
                raise ValueError("Field 'x' missing for 'n_since_last_click'.")
            return np.array(data).ravel()

    raise KeyError("'n_since_last_click' entry not found in b_data.")

if __name__ == "__main__":
    df_list = []

    for subj in range(1, 21):
        mat_file_path = os.path.join(mat_dir, f"b_data_subject_{subj}.mat")
        try:
            values = load_n_since_last_click(mat_file_path)
        except Exception as e:
            print(f"Skipping subject {subj}: {e}")
            continue

        # build DataFrame for this subject
        df = pd.DataFrame({
            'subject': subj,
            'index': np.arange(1, len(values) + 1),
            'n_since_last_click': values
        })

        # keep only odd-index rows
        df = df[df['index'] % 2 == 1].copy()
        # renumber the 'index' column to sequential 1,2,3,...
        df['index'] = np.arange(1, len(df) + 1)

        df_list.append(df)
        print(f"Subject {subj}: loaded {len(values)} entries, kept {len(df)} rows after filtering and renumbering.")

    if not df_list:
        print("No data to export.")
    else:
        df_all = pd.concat(df_list, ignore_index=True)
        df_all.to_csv(output_csv_path, index=False)
        print(f"Exported combined CSV: {output_csv_path}")

