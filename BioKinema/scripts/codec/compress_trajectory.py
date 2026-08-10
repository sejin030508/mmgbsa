#!/usr/bin/env python3
# Copyright 2024 ByteDance and/or its affiliates.
# Licensed under the Apache License, Version 2.0.
"""
Compress ONE trajectory's per-frame bioassembly pickles into the BioKinema
compressed-trajectory format (template + stacked coords).

Input layout (produced by the preprocessing pipeline):
    <system_dir>/<traj>_<frame_id>.pkl.gz   (one per frame)

Output:
    <out_dir>/<traj>.tpl.pkl.gz     (template, coord zeroed)
    <out_dir>/<traj>.coords.npz     (coords [n,N,3] float32 + frame_ids)

Usage:
    python scripts/codec/compress_trajectory.py \
        --system-dir /data/misato_bio_noref/10GS \
        --out-dir    /data/misato_codec
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from protenix.data import traj_codec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system-dir", required=True, help="dir of <traj>_<fid>.pkl.gz")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--traj-name", default=None, help="defaults to system-dir basename")
    ap.add_argument("--quantize", action="store_true", help="int16 0.01A (lossy, ~2x smaller)")
    ap.add_argument("--no-validate", action="store_true", help="skip lossless self-check")
    args = ap.parse_args()

    tpl = traj_codec.compress_system(
        system_dir=args.system_dir,
        out_dir=args.out_dir,
        traj_name=args.traj_name,
        quantize=args.quantize,
        validate=not args.no_validate,
    )
    if tpl is None:
        print(f"[skip] no frame pickles found in {args.system_dir}")
    else:
        print(f"[ok] wrote {tpl}")


if __name__ == "__main__":
    main()
