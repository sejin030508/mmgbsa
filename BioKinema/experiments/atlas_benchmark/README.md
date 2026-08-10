# Reproduce — BioKinema Atlas conformational-ensemble benchmark

This package reproduces BioKinema's result on the **Atlas** test set (82 targets) with the
**10-metric AlphaFlow evaluation suite**. It contains everything needed to (1) roll out
trajectories from the model and (2) score them against the reference MD.

**Reproduce target:** the **`sqrt` checkpoint** (the complex / short-time-MD model — see
"Checkpoint" below) at **1 ns coarse interval** (a 100 ns trajectory, 1 ns/frame, from each
MD frame-0 initial structure, with MSA). Expected metrics are in
[`expected_metrics.txt`](expected_metrics.txt) (headline: **RMWD 2.25, PC-sim 45.7 %,
Pairwise-RMSD r 0.80**).

**Checkpoint.** Download from <https://huggingface.co/fengb/BioKinema>. BioKinema ships two
checkpoints; this benchmark uses the **`sqrt`** one (trained for complexes and short-time MD;
`beta=0.5`). Pass its path via `--checkpoint_path` (it is not hardcoded). The other
checkpoint (`beta=0.25`, for long-time single-chain MD) is used by `kinetics_thermo`, not here.

---

## 1. Contents

```
atlas_benchmark/
├── run_reproduce.sh        # one entry point: inference + analysis (see args below)
├── init_frames/            # 243 initial structures = MD frame-0 of each target's 3 replicas
│                           #   (82 targets x R1/R2/R3), as CIF — shipped so no MD extraction is needed
├── analysis/
│   ├── analyze_ensembles.py  # builds the predicted ensemble, computes the 10 AlphaFlow metrics -> out.pkl
│   └── print_analysis.py     # aggregates out.pkl -> the printed metrics table
├── atlas_test_systems.txt  # the 82 target ids
├── expected_metrics.txt    # reference output to compare against
└── README.md
```

The model/inference code itself lives in the parent BioKinema repo
(`runner/inference_multi.py`, `protenix/…`); `run_reproduce.sh` calls it from the repo root.

---

## 2. Requirements

- The BioKinema conda environment (`conda env create -f ../../environment.yml`, env name `biokinema`/`protenix`).
- A GPU with the model's custom kernels available. `run_reproduce.sh` sets, and you may need to edit:
  `CUTLASS_PATH`, `CUDA_HOME`, `LAYERNORM_TYPE=fast_layernorm`, `USE_DEEPSPEED_EVO_ATTTENTION=true`.
  The kernels are JIT-compiled on first use, so **`ninja` must be on `PATH`** (it is in the conda env).
- Override the Python interpreter with `BIOKINEMA_PY=/path/to/envs/protenix/bin/python` if it differs.
- `--md_dir` (the Atlas MD reference) is only needed for the **analysis** stage; `--stage inference`
  can run without it.
- **Runtime:** 243 rollouts of 100 ns each; on 4×A100 expect a few hours for inference + minutes for analysis.

---

## 3. Required inputs (arguments)

| arg | required | meaning |
|-----|----------|---------|
| `--checkpoint_path` | **yes** | BioKinema model checkpoint (`.pt`) — the **`sqrt`** checkpoint from <https://huggingface.co/fengb/BioKinema>. |
| `--md_dir`          | **yes** | Root of the Atlas MD dataset (used as the reference ensemble in scoring). |
| `--output_dir`      | no (`./reproduce_output`) | where predictions, `out.pkl`, `metrics.txt`, and logs are written. |
| `--init_frames_dir` | no (`init_frames/`) | initial structures; the shipped ones reproduce the reference. |
| `--gpus`            | no (`"0"`) | space-separated GPU ids to shard the 243 rollouts across, e.g. `"0 1 2 3"`. |
| `--msa_cache_dir`   | no (`../../msa`) | MSA cache directory (hashed by sequence); see §6. |
| `--num_workers`     | no (`64`) | CPU workers for the analysis stage. |
| `--stage`           | no (`all`) | `all`, `inference`, or `analysis` only. |

### `--md_dir` layout (Atlas MD dataset)

One sub-directory per target `<name>` (e.g. `5znj_A`), each containing the topology and the
three production replicas:

```
<md_dir>/<name>/<name>.pdb
<md_dir>/<name>/<name>_prod_R1_fit.xtc
<md_dir>/<name>/<name>_prod_R2_fit.xtc
<md_dir>/<name>/<name>_prod_R3_fit.xtc
```

This is the standard ATLAS-simulations layout used by AlphaFlow/MDGen. Download the Atlas MD
trajectories from the ATLAS database (<https://www.dsimb.inserm.fr/ATLAS/>), or with
`scripts/data_prep/download_atlas.sh` from the main repo. Only the 81 benchmark targets are
needed for scoring (`scripts/splits/atlas_test.csv`).

---

## 4. Quick start

```bash
cd experiments/atlas_benchmark
tar -xzf init_frames.tar.gz          # one-time: unpack the bundled initial structures

bash run_reproduce.sh \
    --checkpoint_path /path/to/BioKinema_sqrt_pretrain/5999_ema_0.999.pt \
    --md_dir          /path/to/atlas_sims \
    --output_dir      ./reproduce_output \
    --gpus            "0 1 2 3"
```

- **Stage 1 (inference)** writes one trajectory per init structure to
  `reproduce_output/<name>_R{1,2,3}_0/predictions/*.cif`. It shards the 243 rollouts across the
  given GPUs and automatically retries any CUDA-OOM target at smaller `W_G` (100→50→25→10);
  already-finished targets are skipped, so the run is resumable.
- **Stage 2 (analysis)** writes `reproduce_output/out.pkl` and prints/saves `metrics.txt`.

Run a single stage with `--stage inference` or `--stage analysis` (e.g. re-score without
re-generating).
