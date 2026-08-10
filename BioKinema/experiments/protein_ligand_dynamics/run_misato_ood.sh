#!/bin/bash
# MISATO-OOD ligand stability (Fig 2a): 40 protein-ligand systems, ~1 us, 1 replica each.
# Usage: bash run_misato_ood.sh --checkpoint_path /abs/path/5999_ema_0.999.pt [--gpu 0] [--output_dir DIR]
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${HERE}/run_inference.sh" --system misato_ood "$@"
