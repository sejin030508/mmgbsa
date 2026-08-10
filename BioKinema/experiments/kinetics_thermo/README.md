# BioKinema — Conformational Kinetics & Thermodynamics Benchmark (reproducible)

Self-contained code + data to reproduce the kinetics and thermodynamics evaluation of
BioKinema on the **CATH2-OOD** test set (20 held-out protein domains, sequence identity
≤40 % to the training set). Given a model checkpoint, it (1) generates BioKinema
trajectories from the bundled MD initial frames and (2) compares them against the bundled
MD reference (pre-extracted TICA + Markov state model), producing three artifacts:

1. **Main-text MFPT figure** — pooled mean-first-passage-time joint plot (MD vs BioKinema)
   over all K=10 MSM state pairs of the 20 systems (`mfpt_jointplot`).
2. **Per-system supplement figure** — for each system: structure, MFPT scatter, MSM
   stationary distribution, and MD/BioKinema TICA density maps (`figure_kinetics_sup_page{1,2,3}`).
3. **Aggregated result table** — per-system kinetics + thermodynamics accuracy with a MEAN
   row (`per_system_table.csv`).

The **only required input is the model checkpoint** (passed as `--checkpoint`). Download from
<https://huggingface.co/fengb/BioKinema> and use the **`beta=0.25`** checkpoint — the long-time
single-chain-protein MD model (`beta=0.25`). (BioKinema's other checkpoint, `sqrt`,
is for complexes / short-time MD and is used by `atlas_benchmark` and `protein_ligand_dynamics`,
not here.)

---

## 1. Directory layout

```
kinetics_thermo/
├── README.md
├── run_inference.sh          # STEP 1: generate trajectories  (--checkpoint required)
├── run_analysis.sh           # STEP 2: build figures + table   (--traj_dir required)
├── scripts/
│   ├── traj_msm_utils.py     # shared loaders: BK coords, TICA projection, K-means assign, count matrix
│   ├── prepare_data.py       # builds data_cache.pkl (per-system MFPT, KL, W2, stationary, TICA pools)
│   ├── plot_mfpt.py          # → mfpt_jointplot.{pdf,png}        (main-text MFPT figure)
│   ├── plot_supplement.py    # → figure_kinetics_sup_page{1,2,3}.pdf  (per-system supplement figure)
│   ├── make_table.py         # → per_system_table.csv            (aggregated per-system table)
│   └── plot_common.py        # shared plotting style + the MFPT-jointplot routine
├── example_results/          # reference outputs from the released model (for comparison)
└── data/
    ├── test_systems.txt              # the 20 CATH2-OOD systems
    ├── init_pdbs/<sys>/start_NN.pdb  # MD initial frames (first frame of each MD trajectory)
    ├── msm_cache/<sys>_msm.pkl       # MD reference: pre-extracted TICA basis + K={10,20,50} MSM
    └── structures/<sys>.png          # rendered protein structures (for the supplement figure)
```

### What the bundled MD reference (`data/msm_cache/<sys>_msm.pkl`) contains
A pickled dict per system with everything the analysis needs (no raw MD trajectories required):
- `tica_mean`, `tica_components`, `tica_eigenvalues`, `pair_indices_i/j`, `n_tica_dims` —
  the TICA basis (features are Cα–Cα pairwise distances in **Ångström**;
  `x = (D − tica_mean) @ tica_components`).
- `tica_coords_by_traj` — every MD trajectory already projected to TICA (the MD reference
  point cloud, used for the TICA density maps and W₂).
- `lagtime_frames` (= 1, i.e. 10 ns) and `n_ca_full`.
- `by_k[K]` for `K ∈ {10,20,50}` — `cluster_centers`, `transition_matrix` (reversible-MLE),
  `stationary_dist`, `state_labels`. The benchmark uses **K = 10**.

---

## 2. Requirements

**Inference (STEP 1)** needs the BioKinema model code and its environment:
- The BioKinema repository (provides `runner/inference_multi.py` and the `protenix` package).
  By default `run_inference.sh` assumes the repo root is two levels above this folder
  (`…/BioKinema`); override with `--biokinema_root` or `$BIOKINEMA_ROOT`.
- The `protenix` conda environment (PyTorch + CUDA). Override the interpreter with
  `--python` or `$PYTHON`.
- GPU env vars used by the model kernels — adjust to your system if needed:
  `CUTLASS_PATH`, `CUDA_HOME`, `LAYERNORM_TYPE=fast_layernorm`,
  `USE_DEEPSPEED_EVO_ATTTENTION=true` (sensible defaults are set inside the script).
- One or more CUDA GPUs (default uses 0–7; set `--gpus`).

**Analysis (STEP 2)** is self-contained — only Python with `numpy`, `scipy`, `matplotlib`,
`seaborn`, `mdtraj`, `pillow`. It reads only the bundled `data/` and the generated
trajectories. (The `protenix` env already satisfies these, so the same `--python` works for
both steps.)

---

## 3. Quickstart

```bash
cd experiments/kinetics_thermo
tar -xzf data/init_pdbs.tar.gz -C data   # one-time: unpack the bundled MD initial frames

# STEP 1 — generate 5 × 1 µs @ 10 ns trajectories per init frame (≈3,800 trajectories).
#          The default beta (beta) = 0.25, the released BioKinema model.
bash run_inference.sh \
    --checkpoint /path/to/your_model_5999_ema.pt \
    --output_dir ./out_mymodel

# STEP 2 — build the figures + table against the bundled MD reference.
bash run_analysis.sh --traj_dir ./out_mymodel/biokinema_trajs
```

Results land in `./out_mymodel/analysis/` (MFPT figure, supplement pages, per-system table).
Reference outputs from the released model are provided under `example_results/` for comparison.

On 8×48 GB GPUs, STEP 1 takes ≈4–5 h; STEP 2 takes a few minutes.

The temporal attention bias uses the exponent `β` (`--beta`, default `0.25`); the
released model uses this default and other values need not be changed for reproduction.

---

## 4. Outputs (in `<output_dir>/analysis/`)

| file | content |
|---|---|
| `mfpt_jointplot.{pdf,png}` | **main-text MFPT figure** — pooled mean-first-passage-time joint plot (MD vs BioKinema, all K=10 MSM state pairs over the 20 systems) with log-space Pearson ρ |
| `figure_kinetics_sup_page{1,2,3}.pdf` | **per-system supplement figure** — for each system: structure, per-protein MFPT scatter, MSM stationary distribution (MD vs BK, with D_KL), and MD/BioKinema TICA density maps (with W₂) |
| `per_system_table.csv` | **aggregated result table** — per system: ρ_P, ρ_S (MFPT), D_KL, π MAE, W₂, n_states; final MEAN row |
| `data_cache.pkl` | intermediate cache built by `prepare_data.py` (all per-system arrays) |

Generated trajectories are under `<output_dir>/biokinema_trajs/<sys>/seed_NN/start_NN/start_NN_pred_coordinates.npy`
(shape `[101 frames, N_sample, n_atom, 3]`, Ångström).

**Expected result (verification).** The `MEAN` row of `per_system_table.csv` should land near
the released-model reference (in `example_results/per_system_table.csv`):

| ρ_P (MFPT) ↑ | ρ_S (MFPT) ↑ | D_KL ↓ | π MAE ↓ | W₂ ↓ |
|---|---|---|---|---|
| ≈ 0.77 | ≈ 0.79 | ≈ 0.23 | ≈ 0.041 | ≈ 0.23 |

Small run-to-run differences are expected (stochastic generation + GPU nondeterminism); a
faithful run reproduces these within a few %.

---

## 5. Metric definitions (brief)

- **MFPT** — `m_ij = 1 + Σ_{k≠j} P_ik m_kj`, `m_jj = 0`, solved per target on the K=10 MSM;
  the solution (in lag steps) is multiplied by the 10 ns lag to give physical time. BK and
  MD MFPTs are each computed from their own transition matrix on the shared state space and
  compared, over pairs reachable in both, by Pearson of `log₁₀(MFPT)` (ρ_P) and Spearman (ρ_S).
- **Stationary distribution** — BK uses the reversible-MLE MSM stationary distribution;
  compared to the MD reference by Laplace-smoothed KL (D_KL) and per-state MAE.
- **W₂** — root-mean 1-D Wasserstein-2 distance between MD and BioKinema across the five
  slowest TICA dimensions.
