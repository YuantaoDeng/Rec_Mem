#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch-filter channel CSV files by brain region.

Steps for each subject_i
------------------------
1. Load  LOC_DIR/hdr_subject_i.mat  (struct array 'hdr',  N×1).
2. Build whitelist of electrode indices (1-based) whose 'locus'
   contains "Hippocampus" OR "middletemporal" (case-insensitive).
3. Read  CSV_DIR/subject_XX.csv  and keep rows whose 'channel'
   is in whitelist.
4. Write to  OUT_DIR/subject_XX.csv.

Author : ChatGPT
Date   : 2025-06-27
"""

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

# ------------------------------------------------------------------
CSV_DIR = Path("n_since_last_click_by_channel")           # input CSVs
LOC_DIR = Path("/home/yuantao/Downloads/files/data_localization")
OUT_DIR = Path("n_since_last_click_by_channel_filtered")  # output CSVs
OUT_DIR.mkdir(exist_ok=True)
# ------------------------------------------------------------------

TARGET_REGIONS = [
    "Hippocampus",
    "middletemporal",
]

# Compile a single regex:  \bRegion1\b|\bRegion2\b|...
REGION_PAT = re.compile("|".join(fr"\b{r}\b" for r in TARGET_REGIONS),
                        re.IGNORECASE)


# ---------- helper ------------------------------------------------
def extract_locus(mat_path: Path):
    """
    Return list of region strings (len == #electrodes) or None.

    Handles three storage variants:
      (a) Variable 'locus' exists at top level.
      (b) Struct array 'hdr' with field .locus (N×1 size>1).
      (c) Single struct 'hdr' (size==1).
    """
    mat = sio.loadmat(mat_path, struct_as_record=False, squeeze_me=True)

    # Variant (a): direct variable 'locus'
    if "locus" in mat:
        return [str(x) for x in np.asarray(mat["locus"]).flatten()]

    hdr_obj = mat.get("hdr")
    if hdr_obj is None:
        return None

    # hdr may be ndarray (struct array) or single mat_struct
    if isinstance(hdr_obj, np.ndarray):
        # struct array: iterate every element, grab .locus
        locus_list = []
        for elem in hdr_obj.flatten():
            locus_val = getattr(elem, "locus", None)
            locus_list.append(str(locus_val) if locus_val is not None else "")
        return locus_list
    else:
        # single struct
        locus_field = getattr(hdr_obj, "locus", None)
        if locus_field is None:
            return None
        return [str(x) for x in np.asarray(locus_field).flatten()]


# ---------- main --------------------------------------------------
def main():
    for csv_file in sorted(CSV_DIR.glob("subject_*.csv")):
        m = re.search(r"subject_(\d+)\.csv", csv_file.name, re.IGNORECASE)
        if not m:
            print(f"Skip file (pattern mismatch): {csv_file.name}")
            continue
        subj_id = int(m.group(1).lstrip("0") or "0")

        mat_path = LOC_DIR / f"hdr_subject_{subj_id}.mat"
        if not mat_path.exists():
            print(f"[WARN] Localization missing: {mat_path}")
            continue

        locus_list = extract_locus(mat_path)
        if locus_list is None:
            print(f"[WARN] Could not extract locus from {mat_path}")
            continue

        keep_channels = {
            idx + 1                                  # 1-based channel index
            for idx, region in enumerate(locus_list)
            if REGION_PAT.search(region)
        }
        if not keep_channels:
            print(f"[INFO] Subject {subj_id}: no matching channels.")
            continue

        # Load CSV & filter rows
        df = pd.read_csv(csv_file)
        if "channel" not in df.columns:
            print(f"[WARN] Column 'channel' not found in {csv_file}")
            continue
        df_filt = df[df["channel"].isin(keep_channels)]

        # Save
        out_path = OUT_DIR / csv_file.name
        df_filt.to_csv(out_path, index=False)
        print(f"[OK] Subject {subj_id}: kept {len(df_filt)} / {len(df)} rows "
              f"→ {out_path}")

    print("Finished filtering all subjects.")


if __name__ == "__main__":
    main()
