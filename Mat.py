#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
inspect_mat.py

A simple script to inspect a .mat file (traditional or HDF5-based)
and list its contained variables or datasets.
"""

import os
import scipy.io
import h5py

# === Specify the path to your .mat file here ===
file_path = r"/home/yuantao/Downloads/files/P00_data_behavior/b_data_subject_2.mat"


def inspect_mat_file(path):
    """
    Inspect and print information about a .mat file:
    - Reads the file header to detect format (<=v7.2 vs v7.3/HDF5)
    - Lists variables (traditional MAT) or datasets (HDF5 MAT)
    """
    if not os.path.isfile(path):
        print(f"Error: File not found: {path}")
        return

    # Read the first 128 bytes of the file header
    with open(path, 'rb') as f:
        header = f.read(128)

    # Attempt to decode header for human-readable info
    try:
        header_text = header.decode('ascii')
    except UnicodeDecodeError:
        header_text = header.decode('latin-1', errors='ignore')

    print("=" * 60)
    print("File Header (first 128 bytes):")
    print(header_text)
    print("=" * 60)

    # Detect HDF5 magic number or 'HDF5' signature
    if header.startswith(b'\x89HDF\r\n\x1a\n') or "HDF5" in header_text:
        print("Detected HDF5-based MAT (v7.3). Datasets:")
        with h5py.File(path, 'r') as h5f:
            for name, ds in h5f.items():
                print(f"  • {name}: shape={ds.shape}, dtype={ds.dtype}")
    else:
        print("Detected traditional MAT-file (<= v7.2). Variables:")
        mat = scipy.io.loadmat(path, struct_as_record=False, squeeze_me=True)
        vars = [k for k in mat.keys() if not k.startswith("__")]
        if not vars:
            print("  (No user variables found)")
        else:
            for name in vars:
                v = mat[name]
                if hasattr(v, "shape"):
                    print(f"  • {name}: type={type(v)}, shape={v.shape}")
                else:
                    print(f"  • {name}: type={type(v)}")


if __name__ == "__main__":
    inspect_mat_file(file_path)
