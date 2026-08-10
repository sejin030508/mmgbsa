# Inference (trajectory generation)

BioKinema generates trajectories with a two-stage hierarchical scheme:

1. **Coarse forecasting** — autoregressively generates coarse keyframes with a sliding
   window. `W_H` history frames condition the generation of `W_G` new frames;
   `coarse_frame_num` keyframes are produced at spacing `coarse_interval` (ns).
2. **Fine interpolation** — when `fine_frame_num > 1`, the model fills intermediate
   frames between coarse keyframes for a smoother trajectory.

(A single-shot `conformation_sampling` mode is also available for ensemble generation.)

## Single input

```bash
bash inference.sh \
    --checkpoint_path ./checkpoints/BioKinema_atlas+misato+mdposit_sqrt.pt \
    --dump_dir ./output \
    --input_file ./experiments/atlas_benchmark/init_frames/7lp1_A_R1_0.cif \
    --beta 0.5 \
    --coarse_frame_num 50 --coarse_interval 2 --fine_frame_num 1 --W_H 1 --W_G 50 --N_sample 1
```

`inference.sh` runs `runner/inference.py`. For multi-GPU / multi-system sharding use
`runner/inference_multi.py` (see `example_runs/run_full_pipeline.sh`).

## Inputs

- A single-chain or complex `.pdb` / `.cif`. MSA is fetched/cached automatically
  (`--data.msa.enable true`); the cache dir is controlled by `BIOKINEMA_MSA_CACHE_DIR`.
- Embeddings are computed on the fly (no precomputed embeddings needed for inference).

### MSA cache

Set one persistent cache directory shared by all inference jobs:

```bash
export BIOKINEMA_MSA_CACHE_DIR=/path/to/biokinema_msa_cache
```

The cache is keyed by the normalized protein sequence, not by the input filename,
PDB ID, output directory, coordinates, or ligand. A complex with two different
proteins therefore uses two independent cache entries. Reusing either protein in a
different complex or with a different ligand reuses its existing MSA. Homomers share
one entry.

Each completed entry is validated against its query sequence and marked complete in
`manifest.json`. Failed or interrupted searches are not accepted as cache hits, and
concurrent jobs requesting the same sequence share a file lock so only one submits an
MSA search. The legacy name-based cache layout is intentionally ignored because its
contents cannot be matched safely to a sequence.

## Outputs

```
dump_dir/<dataset>/<id>/seed_<seed>/predictions/
    *_sample_*.cif            predicted structures (per frame / sample)
    *_pred_coordinates.npy    raw coordinate arrays
    *_GT_coordinates.npy      ground truth (if available)
```

Convert to an `.xtc` trajectory:

```bash
python scripts/inference/merge_structure_predictions_to_xtc.py --help
```

## Batch / benchmarks

- `scripts/inference/batch_structure_trajectories.py` — generate trajectories for many
  systems from a directory of CIF/PDB (Protenix or BioEmu layout).
- `benchmarks/atlas/` — the Atlas kinetics benchmark scripts used in the manuscript.
