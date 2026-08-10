#!/bin/bash
# ADK induced-fit (Fig 4b): apo + holo, ~5 us, 10 replicas (long run).
# Usage: bash run_adk.sh --checkpoint_path /abs/path/5999_ema_0.999.pt [--gpu 0] [--output_dir DIR]
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${HERE}/run_inference.sh" --system adk "$@"
