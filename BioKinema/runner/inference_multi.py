#!/usr/bin/env python
"""Batched inference: load the model once, run on many input PDBs sequentially.

Each line of --pdb_list_file is `<pdb_path>\t<dump_dir>` (TAB or whitespace separated).
For each line, runs the standard `infer_predict` pipeline reusing the same
`InferenceRunner`. Saves N * model-load time vs. invoking inference.py once per PDB.

All other CLI flags are identical to inference.py.

Example:
    python runner/inference_multi.py \
        --pdb_list_file /tmp/wg50_gpu0.txt \
        --load_checkpoint_path /path/to.ckpt \
        --seeds 101 --W_H 1 --W_G 50 \
        --coarse_frame_num 100 --coarse_interval 10 ...
"""
import logging
import os
import sys
import traceback

import torch

from configs.configs_base import configs as configs_base
from configs.configs_data import data_configs
from configs.configs_inference import inference_configs
from protenix.config import parse_configs, parse_sys_args
from runner.inference import InferenceRunner, infer_predict, download_infercence_cache
from runner.dumper import DataDumper

logger = logging.getLogger(__name__)


def _pop_arg(name):
    """Remove `--name VALUE` from sys.argv and return VALUE."""
    if name in sys.argv:
        i = sys.argv.index(name)
        v = sys.argv[i + 1]
        del sys.argv[i:i + 2]
        return v
    return None


def _load_pdb_list(path):
    """Parse `pdb_path<TAB>dump_dir[<TAB>seed[,seed...]]` per line.

    Optional 3rd column overrides `--seeds` for that job (comma-separated list).
    Comments / blanks are skipped.
    """
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"bad line in {path}: {line!r}")
            seeds = None
            if len(parts) >= 3:
                seeds = [int(s) for s in parts[2].split(",") if s]
            items.append((parts[0], parts[1], seeds))
    return items


def _set_dump_dir(runner, dump_dir, need_atom_confidence):
    runner.configs.dump_dir = dump_dir
    runner.dump_dir = dump_dir
    runner.error_dir = os.path.join(dump_dir, "ERR")
    os.makedirs(runner.dump_dir, exist_ok=True)
    os.makedirs(runner.error_dir, exist_ok=True)
    runner.dumper = DataDumper(
        base_dir=runner.dump_dir, need_atom_confidence=need_atom_confidence
    )


def main():
    LOG_FORMAT = "%(asctime)s,%(msecs)-3d %(levelname)-8s [%(filename)s:%(lineno)s %(funcName)s] %(message)s"
    logging.basicConfig(
        format=LOG_FORMAT, level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S", filemode="w",
    )

    pdb_list_file = _pop_arg("--pdb_list_file")
    if not pdb_list_file:
        raise SystemExit("--pdb_list_file is required")
    items = _load_pdb_list(pdb_list_file)
    if not items:
        print(f"[inference_multi] empty pdb list: {pdb_list_file}")
        return
    first_pdb, first_dump_dir, _ = items[0]
    print(f"[inference_multi] {len(items)} jobs from {pdb_list_file}")
    print(f"[inference_multi] first: {first_pdb} -> {first_dump_dir}")

    # Configs initialization (mirror inference.run)
    configs_base["use_deepspeed_evo_attention"] = (
        os.environ.get("USE_DEEPSPEED_EVO_ATTTENTION", False) == "true"
    )
    configs = {**configs_base, **{"data": data_configs}, **inference_configs}
    # Inject required values from first item so parse_configs is happy.
    extra = ["--input_file", first_pdb, "--dump_dir", first_dump_dir]
    arg_str = parse_sys_args() + " " + " ".join(extra)
    configs = parse_configs(
        configs=configs, arg_str=arg_str, fill_required_with_null=True,
    )
    download_infercence_cache(configs, model_version="v0.2.0")

    need_atom_confidence = configs.need_atom_confidence
    runner = InferenceRunner(configs)  # model loaded ONCE

    default_seeds = list(runner.configs.seeds)
    n_ok = 0; n_err = 0
    for idx, (pdb_path, dump_dir, seeds_override) in enumerate(items):
        # Skip if expected output already there. Output naming: <stem>_pred_coordinates.npy under
        # <dump_dir>/<stem>/  -- handled by DataDumper's dump dir convention.
        stem = os.path.splitext(os.path.basename(pdb_path))[0]
        marker = os.path.join(dump_dir, stem, f"{stem}_pred_coordinates.npy")
        if os.path.exists(marker):
            print(f"[{idx+1}/{len(items)}] SKIP {pdb_path} (exists)")
            n_ok += 1
            continue
        runner.configs.input_file = pdb_path
        runner.configs.seeds = seeds_override if seeds_override is not None else default_seeds
        _set_dump_dir(runner, dump_dir, need_atom_confidence)
        print(f"[{idx+1}/{len(items)}] {pdb_path} -> {dump_dir}  seeds={list(runner.configs.seeds)}")
        try:
            infer_predict(runner, runner.configs)
            n_ok += 1
        except Exception as e:
            n_err += 1
            print(f"[ERR] {pdb_path}: {e}")
            traceback.print_exc()
        if hasattr(torch.cuda, "empty_cache"):
            torch.cuda.empty_cache()
    print(f"[inference_multi] done: ok={n_ok} err={n_err}")


if __name__ == "__main__":
    main()
