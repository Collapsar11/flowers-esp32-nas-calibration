#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="${FLOWERS_GPU:-1}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
mkdir -p logs
run_one() {
  local model="$1" name="$2"
  shift 2
  if [[ ! -f "runs/$name/summary.json" ]]; then
    .venv/bin/python -u pipeline.py train --model "$model" --run "$name" "$@" > "logs/$name.log" 2>&1
  fi
  if [[ ! -f "runs/$name/export/report.json" ]]; then
    .venv/bin/python -u pipeline.py export --run "$name" >> "logs/$name.log" 2>&1
  fi
}
test -f logs/dataset.json
for model in mbv3-small mcunet-in2 ofa-min ofa-mid ofa-max; do
  run_one "$model" "smoke-$model-v1" --head-epochs 1 --epochs 1 --max-batches 2
done
for model in mbv3-small mcunet-in2 ofa-min ofa-mid ofa-max; do
  run_one "$model" "pilot-$model-v1"
done
echo 'PILOT_TRAINING_AND_FP32_EXPORT_COMPLETE; INT8_AND_QEMU_VALIDATION_SEPARATE'
