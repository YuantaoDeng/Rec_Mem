#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
svm_threshold_sweep_parallel.py

Parallelized version of svm_threshold_sweep_with_progress.py:
  - Preloads each subject/channel .mat once into memory
  - Computes all sample features in parallel using multiprocessing
  - Performs threshold sweep with fixed RBF SVM hyperparameters
"""

import os
import numpy as np
import pandas as pd
import scipy.io as sio
from multiprocessing import Pool, cpu_count
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline


# === User settings ===
BY_CHANNEL_DIR = "n_since_last_click_by_channel_filtered"
POWER_DIR      = "/home/yuantao/Downloads/files/data_power_trials"
THRESHOLDS     = list(range(1, 21))
SVM_PARAMS     = {
    'kernel': 'linear',
    'C': 1.0,
    'gamma': 'scale',
    "class_weight": "balanced",
    'decision_function_shape': 'ovr',
    'random_state': 42
}
CV_FOLDS     = 5
RANDOM_STATE = 42

# bands keys in the .mat files
BAND_KEYS = [
    "theta_data_",
    "alpha_data",
    "beta_data",
    "lowgamma_data",

]

# global cache for preloaded band means
band_cache = {}

def label_clicks(n_clicks, threshold):
    """Map click count to class label 0,1,2 based on threshold."""
    if n_clicks == 0:
        return 0
    elif n_clicks < threshold:
        return 1
    else:
        return 2

def preload_band_means(requests):
    """
    Preload every unique (subject,channel) .mat,
    compute mean over freq_bins for each band,
    store in global band_cache[(subj,chan,key)] = 2D array (time_points, n_trials).
    """
    unique_pairs = set((subj, chan) for subj, chan, _, _ in requests)
    print(f"Preloading {len(unique_pairs)} mat files into cache...")
    for subj, chan in unique_pairs:
        mat_path = os.path.join(POWER_DIR, str(subj), f"{chan}.mat")
        mat = sio.loadmat(mat_path, struct_as_record=False, squeeze_me=True)
        for key in BAND_KEYS:
            arr = mat.get(key)
            if arr is None:
                band_cache[(subj, chan, key)] = None
            else:
                # average over freq_bins (axis=1), keep shape (time_points, n_trials)
                band_cache[(subj, chan, key)] = arr.mean(axis=1)
    print("Preloading complete.\n")


WIN_SPLITS = 4    
STAT_FUNCS = [np.mean, np.std]

def compute_features(args):
    """
    args  -> (subj, chan, idx, nclick)
    return-> (feature_vector, nclick)
    """
    subj, chan, idx, nclick = args
    idx0 = idx - 1
    feats = []

    for key in BAND_KEYS:
        band_means = band_cache.get((subj, chan, key))
        if band_means is None:
            feats.extend([0.0] * (2 * (1 + WIN_SPLITS)))
            continue

        ts = band_means[:, idx0]               
        feats.append(ts.mean())
        feats.append(ts.std())

        segments = np.array_split(ts, WIN_SPLITS)
        for seg in segments:
            feats.append(seg.mean())
            feats.append(seg.std())

    return feats, nclick


SUBJ_ID = 1

def main():
    # 1) Load and concat per-subject CSVs
    print("Loading per-subject CSVs...")
    dfs = []
    for fname in sorted(os.listdir(BY_CHANNEL_DIR)):
        if fname.endswith(".csv"):
            dfs.append(pd.read_csv(os.path.join(BY_CHANNEL_DIR, fname)))
    data_df = pd.concat(dfs, ignore_index=True)
    data_df = data_df[data_df["subject"] == SUBJ_ID]
    n_samples = len(data_df)
    print(f"Total samples: {n_samples}\n")

    # build list of extraction requests: (subj, channel, index, nclick)
    requests = [
    (int(r.subject), int(r.channel), int(r.index), int(r.n_since_last_click))
    for r in data_df.itertuples(index=False)
    ]


    # 2) Preload band means into memory
    preload_band_means(requests)

    # 3) Parallel feature extraction
    print(f"Extracting features in parallel on {cpu_count()} cores...")
    with Pool() as pool:
        results = pool.map(compute_features, requests)
    print("Feature extraction complete.\n")

    # unpack results
    X_list, clicks = zip(*results)
    X = np.vstack(X_list)
    clicks = np.array(clicks, dtype=int)

    # prepare cross-validator
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # 4) Threshold sweep with fixed SVM params
    print("Starting threshold sweep...\n")
    results = []
    total_thr = len(THRESHOLDS)
    for i, thr in enumerate(THRESHOLDS, start=1):
        print(f"[{i}/{total_thr}] Threshold = {thr}")
        y = np.array([label_clicks(n, thr) for n in clicks], dtype=int)

        PCA_DIM = 0.95        
        clf = make_pipeline(
        StandardScaler(),                         # zero-mean / unit-var
        PCA(n_components=PCA_DIM, random_state=RANDOM_STATE),
        SVC(**SVM_PARAMS)                         # linear kernel
        )


        print(f"  Running {CV_FOLDS}-fold CV...", end="", flush=True)
        scores = cross_val_score(
        clf, X, y, cv=cv,
        scoring='balanced_accuracy', 
        n_jobs=-1
        )
        acc = scores.mean()
        print(f" done. Accuracy = {acc:.4f}\n")

        results.append({'threshold': thr, 'accuracy': acc})

    # 5) Print summary
    df_res = pd.DataFrame(results)
    print("Threshold sweep complete. Summary:\n")
    print(df_res.to_string(index=False))

if __name__ == "__main__":
    main()
