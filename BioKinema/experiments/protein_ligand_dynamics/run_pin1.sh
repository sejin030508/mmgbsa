#!/bin/bash
# Pin1 allosteric (Fig 4c): apo + holo, ~1.5 us, 10 replicas.
# Usage: bash run_pin1.sh --checkpoint_path /abs/path/5999_ema_0.999.pt [--gpu 0] [--output_dir DIR]
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${HERE}/run_inference.sh" --system pin1 "$@"
