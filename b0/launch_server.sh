#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ $# -lt 2 ]]; then
  echo 'Usage: bash b0/launch_server.sh DATA_ROOT NEW_OUTPUT_DIR [extra b0.run arguments]'
  exit 2
fi
task_data_root=$1
task_output=$2
shift 2
task_python=${B0_PYTHON:-python}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2}
"$task_python" -m b0.run \
  --data-root "$task_data_root" \
  --train-manifest prepare/manifests/train_N28_seed0.csv \
  --validation-manifest prepare/manifests/validation_N28_seed0.csv \
  --output "$task_output" \
  --scope development_baseline \
  --steps 200000 --lr-horizon-steps 200000 \
  --batch-size 2 --seed 0 --device cuda --num-workers 0 \
  --checkpoint-every 10000 --log-every 100 \
  "$@"
