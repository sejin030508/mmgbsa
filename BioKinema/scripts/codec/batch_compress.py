#!/usr/bin/env python3
# Copyright 2024 ByteDance and/or its affiliates.
# Licensed under the Apache License, Version 2.0.
"""
Batch-compress a whole processed dataset into the compressed-trajectory format.

A processed dataset's ``bioassembly_dict_dir`` is laid out as one subdirectory per
system/trajectory, each holding per-frame ``<traj>_<fid>.pkl.gz`` pickles
(this is how MISATO / MDposit / unbinding are stored; see configs/configs_data.py).
This walks every subdirectory and writes ``<traj>.tpl.pkl.gz`` + ``<traj>.coords.npz``
into ``--out-dir``, which can then be used directly as the dataset's
``bioassembly_dict_dir`` (the codec is auto-detected by protenix/data/dataset.py).

Usage:
    python scripts/codec/batch_compress.py \
        --bio-dir  /data/misato_bio_noref \
        --out-dir  /data/misato_codec \
        --n-workers 32
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from protenix.data import traj_codec


def _one(system_dir, out_dir, quantize, validate):
    try:
        tpl = traj_codec.compress_system(
            system_dir=system_dir, out_dir=out_dir,
            quantize=quantize, validate=validate,
        )
        return (system_dir, "ok" if tpl else "empty", None)
    except Exception as e:  # noqa: BLE001
        return (system_dir, "error", repr(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bio-dir", required=True, help="dir with one subdir per trajectory")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-workers", type=int, default=8)
    ap.add_argument("--quantize", action="store_true")
    ap.add_argument("--no-validate", action="store_true")
    ap.add_argument("--limit", type=int, default=-1, help="process at most N systems (debug)")
    args = ap.parse_args()

    systems = [
        os.path.join(args.bio_dir, d)
        for d in sorted(os.listdir(args.bio_dir))
        if os.path.isdir(os.path.join(args.bio_dir, d))
    ]
    if args.limit > 0:
        systems = systems[: args.limit]
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"compressing {len(systems)} trajectories -> {args.out_dir}")

    n_ok = n_empty = n_err = 0
    with ProcessPoolExecutor(max_workers=args.n_workers) as ex:
        futs = [
            ex.submit(_one, s, args.out_dir, args.quantize, not args.no_validate)
            for s in systems
        ]
        for i, fut in enumerate(as_completed(futs), 1):
            sysd, status, err = fut.result()
            if status == "ok":
                n_ok += 1
            elif status == "empty":
                n_empty += 1
            else:
                n_err += 1
                print(f"  [error] {os.path.basename(sysd)}: {err}")
            if i % 200 == 0:
                print(f"  {i}/{len(systems)} (ok={n_ok} empty={n_empty} err={n_err})")
    print(f"done: ok={n_ok} empty={n_empty} err={n_err}")


if __name__ == "__main__":
    main()
