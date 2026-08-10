# Reproducing BioKinema's benchmarks

Three self-contained packages reproduce the paper's quantitative results end-to-end (generate
trajectories from a released checkpoint, then score/plot them against bundled references):

| Package | Reproduces | Test set | Checkpoint to use |
|---|---|---|---|
| [`atlas_benchmark/`](atlas_benchmark) | Conformational-ensemble accuracy (10-metric AlphaFlow suite) | Atlas | **`sqrt`** |
| [`protein_ligand_dynamics/`](protein_ligand_dynamics) | Fig 4b ADK induced-fit, Fig 4c Pin1 allostery, Fig 2a MISATO ligand stability | ADK / Pin1 / MISATO-OOD | **`sqrt`** |
| [`kinetics_thermo/`](kinetics_thermo) | Conformational kinetics + thermodynamics (MFPT / MSM / TICA) | CATH2-OOD (20 domains) | **`beta=0.25`** |

## Checkpoints

Download from **<https://huggingface.co/fengb/BioKinema>**. BioKinema releases **two** checkpoints
(same architecture, different temporal-attention setting and training regime):

- **`sqrt`** — complexes (protein–ligand) and **short-time** MD. `beta = 0.5`.
  Used by `atlas_benchmark` and `protein_ligand_dynamics`.
- **`beta=0.25`** — **long-time, single-chain** protein MD. `beta = 0.25`.
  Used by `kinetics_thermo`.

Each package takes the checkpoint via a CLI flag (`--checkpoint_path` / `--checkpoint`); none is
hardcoded. Pick the variant from the table above.

## Shared setup

- `conda env create -f ../environment.yml && conda activate biokinema`.
- **Unpack the bundled inputs first.** The initial structures ship as compressed archives (to
  keep the repo small); extract them once before running any package:
  ```bash
  tar -xzf atlas_benchmark/init_frames.tar.gz                  -C atlas_benchmark
  tar -xzf kinetics_thermo/data/init_pdbs.tar.gz               -C kinetics_thermo/data
  tar -xzf protein_ligand_dynamics/data/init_structures.tar.gz -C protein_ligand_dynamics/data
  ```
- The model's custom CUDA kernels are JIT-compiled on first use → **`ninja` must be on `PATH`**
  (it ships in the conda env). GPU env vars (`CUTLASS_PATH`, `CUDA_HOME`,
  `LAYERNORM_TYPE=fast_layernorm`, `USE_DEEPSPEED_EVO_ATTTENTION=true`) have defaults inside each
  script; edit them for your machine.
- Each package's `README.md` lists its exact inputs, outputs, expected reference numbers, and a
  "how to verify a match" check. Generated trajectories are large and are written to a
  user-chosen output dir, **not** committed here; only bundled inputs + reference results ship.

Start with whichever package matches the result you want to reproduce.
