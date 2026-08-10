# Protein conformational dynamics & ligand stability (ADK / Pin1 / MISATO-OOD)

Reproduces, end-to-end, the all-atom trajectories and the figures for the three
protein systems in the paper:

| System        | Figure | What it shows                                   |
|---------------|--------|-------------------------------------------------|
| **Pin1**      | Fig 4c | allosteric loop (res 60–70) motion, apo vs holo  |
| **ADK**       | Fig 4b | induced-fit open↔closed transition, apo vs holo  |
| **MISATO-OOD**| Fig 2a | ligand physical (bond) stability over time       |

All results use the **`sqrt` checkpoint** — the complex / short-time-MD model
(`beta=0.5`) from <https://huggingface.co/fengb/BioKinema>. (The other released
checkpoint, `beta=0.25` for long single-chain MD, is for `kinetics_thermo`, not these systems.)
Pipeline: **(1) `run_inference.sh`** → trajectories, then **(2) `run_analysis.sh`** → PDFs.

```
protein_ligand_dynamics/
├── run_inference.sh         # STEP 1 — needs --checkpoint_path and --output_dir
├── run_pin1.sh / run_adk.sh / run_misato_ood.sh   # thin per-system wrappers
├── run_analysis.sh          # STEP 2 — needs --trajs_root (= STEP 1 --output_dir)
├── data/
│   └── init_structures/{pin1,adk,misato_ood}/*.cif   # bundled initial-frame structures (inputs)
├── scripts/
│   ├── plot_pin1.py / plot_adk.py / plot_misato.py    # figure drivers
│   ├── lib_pin1.py / lib_adk.py / lib_misato.py       # analysis funcs (verbatim from notebooks)
│   ├── plot_style.py
│   └── refs/1ake_a.cif, 4ake_a.cif                    # ADK reference conformers (closed/open)
└── example_results/         # reference figures to compare against
```

Generated trajectories are **not** stored in this folder — `run_inference.sh` writes
them to a user-chosen `--output_dir` (they are large). Only the bundled initial-frame
structures (`data/init_structures/`) and reference figures (`example_results/`) ship here.

---

## 0. Requirements

* `conda activate biokinema` (env in `../../environment.yml`).
* A CUDA GPU.
* The **`sqrt` checkpoint** — **not hardcoded**, passed via `--checkpoint_path`. Download from
  <https://huggingface.co/fengb/BioKinema> (use the `sqrt` / complex+short-MD variant).
* `ninja` on `PATH` (kernels are JIT-compiled on first use; it ships in the conda env).
* Host-specific paths (override by `export`-ing before running; defaults target our
  dev box): `CUTLASS_PATH`, `CUDA_HOME` (DeepSpeed EvoformerAttention), `BIOKINEMA_MSA_CACHE_DIR`.
* `BIOKINEMA_LIGAND_BONDS_DIR` (optional) — at **inference**, replaces a system's on-the-fly
  ligand bonds with known-correct ones from a precomputed bioassembly dir (`<dir>/<stem>/<stem>_*.pkl.gz`),
  fixing ring-closure bonds that cif perception drops. Used for the MISATO-OOD systems (point it at
  the MISATO bio dir). For ADK/Pin1 it is not needed: their ligand bonds are taken from the input cif,
  and if the strict valence heuristic flags them the inference path warns and proceeds (it is not fatal).

Run the scripts from anywhere — they resolve the BioKinema repo root automatically.

---

## 1. Inference

```bash
tar -xzf data/init_structures.tar.gz -C data   # one-time: unpack the bundled initial structures

CKPT=/abs/path/to/5999_ema_0.999.pt
OUT=/abs/path/for/trajectories          # NOT inside this folder

bash run_pin1.sh        --checkpoint_path $CKPT --output_dir $OUT --gpu 0   # apo+holo, ~1.5 us, 10 replicas
bash run_adk.sh         --checkpoint_path $CKPT --output_dir $OUT --gpu 1   # apo+holo, ~5 us,  10 replicas (long)
bash run_misato_ood.sh  --checkpoint_path $CKPT --output_dir $OUT --gpu 2   # 40 systems, ~1 us, 1 replica
```
Optional: `--input_file <one.cif>` runs a single structure. **To parallelize apo/holo across
GPUs, pass distinct `--gpu` values** (the script binds `CUDA_VISIBLE_DEVICES` to `--gpu`, so an
outer `CUDA_VISIBLE_DEVICES` is overridden — set the GPU only via `--gpu`):

```bash
bash run_inference.sh --system adk --checkpoint_path $CKPT --output_dir $OUT --gpu 0 \
     --input_file data/init_structures/adk/ADK_apo.cif  &
bash run_inference.sh --system adk --checkpoint_path $CKPT --output_dir $OUT --gpu 1 \
     --input_file data/init_structures/adk/ADK_holo.cif &
wait
```

**Runtime:** Pin1 (151 frames ×10) is ~tens of minutes/structure; **ADK (501 frames ×10) is
the long pole — a few hours/structure** (run apo & holo on separate GPUs).

**Output** (per input structure):
```
$OUT/<system>/<stem>/<stem>_pred_coordinates.npy            # [frame, sample, atom, 3]
$OUT/<system>/<stem>/predictions/<stem>_s{S}_f{F}_wounresol.cif
```
Frame **f0 is the input (initial) structure** (from `data/init_structures/`); f1… is the
generated path. Wait for the `[reproduce] DONE system=<sys>` line — the per-sample CIF
files are written **after** the `.npy`, so the run is only complete once DONE is printed.

### Exact sampling settings (identical to the paper run; baked into `run_inference.sh`)

| param | value | | param | value |
|---|---|---|---|---|
| seed | 101 | | coarse_interval | 10 ns |
| N_cycle | 10 | | fine_frame_num | 1 (no interpolation) |
| N_step | 20 | | W_H | 1 |
| noise_scale_lambda | 1.75 | | history_noise | 0.0 |
| step_scale_eta | 1.5 | | history_t | 1.6e-1 |
| beta | 0.5 (sqrt-t ALiBi) | | causal_mask | false |

| system | coarse_frame_num | total time | N_sample | W_G |
|---|---|---|---|---|
| pin1       | 151 | ~1.5 µs | 10 | 50 |
| adk        | 501 | ~5.0 µs | 10 | 10 |
| misato_ood | 101 | ~1.0 µs | 1  | 100 |

---

## 2. Figures

```bash
# all three (or --system pin1|adk|misato_ood); --trajs_root = the --output_dir above
BIOKINEMA_LIGAND_BONDS_DIR=/path/to/ligand_bonds \
bash run_analysis.sh --trajs_root $OUT --system all --out_dir ./my_figures
```

Produced PDFs (compare with `example_results/`):
```
fig4c_pin1_rmsf.pdf
fig4c_pin1_rmsd_align1-999_calc60-70.pdf       # loop 60-70 RMSD-to-f0, x-axis in us (0.0..1.5)
fig4b_adk_{rmsd,scatter,rmsf}_{apo,holo}.pdf    # RMSF uses the 4-5 us window only
misato_bond_length_error.pdf
misato_bond_angle_error.pdf                     # reported to stdout
```

The plotting code in `scripts/lib_*.py` is extracted **verbatim** from the manuscript
notebooks (`fig4b_induced_fit.ipynb`, `fig4c_allosteric.ipynb`,
`fig2a_physical_stability.ipynb`); only the final figure parameters are baked in.

