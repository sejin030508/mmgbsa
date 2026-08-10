# Compressed-trajectory codec (MISATO / MDposit / unbinding)

The preprocessing for MISATO, MDposit and unbinding is highly source-specific and not
practical to publish. Instead we publish the **processed** data — but in a compact form.

## Why a codec

The training pipeline consumes one `bioassembly_dict` pickle **per frame**
(`<traj>/<traj>_<frame_id>.pkl.gz`). Within a single trajectory these dicts are
byte-identical except for three things:

- `pdb_id`  (e.g. `10GS_47`)
- `frame_id`
- `atom_array.coord`  — the `[N_atom, 3]` float32 coordinates

Topology, bonds, token array, sequences and all other annotations are constant across
frames. So shipping hundreds of near-duplicate pickles per trajectory is wasteful.

## Format

For each trajectory we ship two files:

```
<traj>.tpl.pkl.gz    # ONE template bioassembly_dict (coord zeroed out)
<traj>.coords.npz    # coords [n_frames, N_atom, 3] float32  +  frame_ids [n_frames] int32
```

`protenix/data/traj_codec.py:load_frame()` reconstructs any frame on the fly
(template + spliced coord), producing an object **identical** to the original per-frame
pickle. This is verified lossless (exact float32 + byte-identical non-coord content).

## Decompression is automatic

`protenix/data/dataset.py` auto-detects the codec: if a dataset's
`bioassembly_dict_dir` contains `<traj>.tpl.pkl.gz`, frames are loaded via the codec
instead of per-frame pickles. **No config change is needed** — point
`bioassembly_dict_dir` at the codec directory.

## Compressing your own processed data

```bash
# one trajectory
python scripts/codec/compress_trajectory.py \
    --system-dir /data/misato_bio_noref/10GS \
    --out-dir    /data/misato_codec

# a whole dataset (one subdir per trajectory)
python scripts/codec/batch_compress.py \
    --bio-dir   /data/misato_bio_noref \
    --out-dir   /data/misato_codec \
    --n-workers 32
```

Flags:
- `--quantize` — store coords as int16 fixed-point at 0.01 A (~2x smaller, slightly
  lossy). Default is float32 (lossless, ~3x smaller than the raw per-frame pickles).
- `--no-validate` — skip the per-frame "non-coord content matches template" self-check
  (the check guards the lossless assumption; keep it on unless you trust the input).

Then set the dataset's `bioassembly_dict_dir` (via `BIOKINEMA_UNBINDING_ROOT` and
`configs/configs_data.py`) to the codec output directory.
