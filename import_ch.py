#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
export_n_since_last_click_by_channel_per_subject.py

For each subject (1–20):
  1) read n_since_last_click_all_subjects.csv,
  2) load its n_since_last_click values,
  3) expand across all channels,
  4) save a separate CSV into a dedicated folder.
"""

import os
import numpy as np
import pandas as pd
import scipy.io as sio

# === Paths: adjust as needed ===
csv_input_path    = "n_since_last_click_all_subjects.csv"
patients_mat_path = "/home/yuantao/Downloads/files/patients_gray_matter.mat"
output_dir        = "n_since_last_click_by_channel"  # 保存 20 个 CSV 的文件夹

# create output directory if not exists
os.makedirs(output_dir, exist_ok=True)

# 1) load the combined CSV with subject,index,value
df_all = pd.read_csv(csv_input_path)

# 2) load patients struct to get total_channel per subject
mat = sio.loadmat(patients_mat_path, struct_as_record=False, squeeze_me=True)
patients = mat.get('patients')
if isinstance(patients, np.ndarray):
    patients = patients.flatten().tolist()
else:
    patients = [patients]

subj_to_nchan = {}
for p in patients:
    # field name for subject id might be 'n_subject' or 'n_sub'
    subj = int(getattr(p, 'n_subject', None) or getattr(p, 'n_sub', None))
    subj_to_nchan[subj] = int(p.total_channel)

# 3) per-subject processing and saving
for subj, subdf in df_all.groupby('subject'):
    if subj not in subj_to_nchan:
        print(f"Warning: no channel info for subject {subj}, skipping")
        continue

    nchan = subj_to_nchan[subj]
    channels = np.arange(1, nchan + 1)

    # cross join subject's rows with all channels
    subdf = subdf.reset_index(drop=True)
    subdf['__key'] = 1
    ch_df = pd.DataFrame({'channel': channels})
    ch_df['__key'] = 1

    joined = subdf.merge(ch_df, on='__key').drop(columns='__key')

    # save per-subject CSV
    out_path = os.path.join(output_dir, f"subject_{subj:02d}.csv")
    joined.to_csv(out_path, index=False)
    print(f"Subject {subj}: {len(subdf)} × {nchan} → saved {len(joined)} rows to {out_path}")
