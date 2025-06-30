#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
remove_all_bad_trials.py

For each subject (1–20):
  1) load failed trials ('out_rt') from b_data_subject_i.mat
  2) load per-channel artifact trials ('P02_art') from data_subject_except_power/i.mat
  3) load per-channel abnormal power trials from abnormal_power_trials_all.mat
  4) read the corresponding CSV (subject_XX.csv),
  5) drop rows whose 'index' is in out_rt or P02_art or abnormal_power_trials,
  6) overwrite the CSV with the cleaned data.
"""

import os
import numpy as np
import pandas as pd
import scipy.io as sio

# === 配置路径 ===
behavior_mat_dir   = "/home/yuantao/Downloads/files/P00_data_behavior"
power_mat_dir      = "/home/yuantao/Downloads/files/data_subject_except_power"
abnormal_mat_path  = "/home/yuantao/Downloads/files/abnormal_power_trials_all.mat"
csv_dir            = "n_since_last_click_by_channel"   # 存放 subject_01.csv … subject_20.csv 的文件夹

# 1) 读取 abnormal_power_trials_all.mat
abn_mat = sio.loadmat(abnormal_mat_path, struct_as_record=False, squeeze_me=True)
abn_struct = abn_mat.get('abnormal_power_trials_all')
# normalize to Python list of structs
if isinstance(abn_struct, np.ndarray):
    abn_list = abn_struct.flatten().tolist()
else:
    abn_list = [abn_struct]

for subj in range(1, 21):
    # --- 2) load failed trials ('out_rt') ---
    beh_path = os.path.join(behavior_mat_dir, f"b_data_subject_{subj}.mat")
    if not os.path.isfile(beh_path):
        print(f"[Subj{subj:02d}] Missing {beh_path}, skip.")
        continue
    beh = sio.loadmat(beh_path, struct_as_record=False, squeeze_me=True)
    out_rt = beh.get('out_rt', [])
    failed = np.atleast_1d(out_rt).astype(int).ravel().tolist()

    # --- 3) load artifact trials ('P02_art') ---
    pwr_path = os.path.join(power_mat_dir, f"{subj}.mat")
    art_dict = {}
    if os.path.isfile(pwr_path):
        pwr = sio.loadmat(pwr_path, struct_as_record=False, squeeze_me=True)
        P02_art = pwr.get('P02_art', None)
        if P02_art is not None:
            cell = np.atleast_1d(P02_art)
            for ch_idx, entry in enumerate(cell, start=1):
                trials = np.atleast_1d(entry).astype(int).ravel().tolist()
                art_dict[ch_idx] = trials
    else:
        print(f"[Subj{subj:02d}] No power mat, skip artifacts.")

    # --- 4) load abnormal-power trials for this subject ---
    abn_trials = {}
    if subj-1 < len(abn_list):
        entry = abn_list[subj-1]
        # field name is 'abnormal_power_trials'
        cell = getattr(entry, 'abnormal_power_trials', None)
        if cell is not None:
            cell = np.atleast_1d(cell)
            for ch_idx, arr in enumerate(cell, start=1):
                trials = np.atleast_1d(arr).astype(int).ravel().tolist()
                abn_trials[ch_idx] = trials
    else:
        print(f"[Subj{subj:02d}] No entry in abnormal_power_trials_all.")

    # --- 5) 读取对应 CSV ---
    csv_path = os.path.join(csv_dir, f"subject_{subj:02d}.csv")
    if not os.path.isfile(csv_path):
        print(f"[Subj{subj:02d}] CSV missing: {csv_path}")
        continue
    df = pd.read_csv(csv_path)
    orig_n = len(df)

    # --- 6) 统一过滤 ---
    # 6.1 drop failed trials globally
    if failed:
        df = df[~df['index'].isin(failed)]

    # 6.2 drop P02_art artifacts per channel
    for ch, bad in art_dict.items():
        if bad:
            mask = (df['channel']==ch) & (df['index'].isin(bad))
            df = df[~mask]

    # 6.3 drop abnormal-power trials per channel
    for ch, bad in abn_trials.items():
        if bad:
            mask = (df['channel']==ch) & (df['index'].isin(bad))
            df = df[~mask]

    # 7) 覆盖保存
    kept_n = len(df)
    df.to_csv(csv_path, index=False)
    print(f"[Subj{subj:02d}] dropped {orig_n-kept_n}/{orig_n} rows → kept {kept_n}")
