#!/usr/bin/env bash
# No physical board operations. Stop immediately if any server gate fails.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
run="${1:?run name required}"
qatdir="${2:?QAT result directory required}"
quantdir="${3:?unique accepted-candidate export directory required}"
firmwaredir="firmware-$quantdir"
.venv/bin/python - "$run" "$qatdir" "$quantdir" <<'PY'
import json, pathlib, shutil, sys
import pipeline as p
run = p.ROOT / 'runs' / sys.argv[1]
source, dest = run / sys.argv[2] / 'best-export', run / sys.argv[3]
report = json.loads((source / 'report.json').read_text())
selected = json.loads((run / sys.argv[2] / 'best.json').read_text())
assert selected['production_sha256'] == report['production_sha256']
quality = p.config()['quality']
reference = json.loads((run / 'summary.json').read_text())['validation']['top1']
assert report['validation_count'] == 1020 and report['test_used'] is False
assert report['int8_top1'] >= quality['validation_top1_target'], 'Accuracy floor failed'
assert reference - report['int8_top1'] <= quality['max_quantization_drop'], 'Quantization drop failed'
assert p.digest(source / 'model.tflite') == report['production_sha256']
dest.mkdir(exist_ok=False)
for name in ['model.tflite', 'model.audit.json', 'report.json']:
    shutil.copy2(source / name, dest / name)
p.save_json(dest / 'training-provenance.json', {'qat_directory': sys.argv[2],
    'selected_epoch': selected['epoch'],
    'checkpoint_sha256': p.digest(run / sys.argv[2] / 'best.pt'),
    'production_sha256': report['production_sha256'], 'test_used': False})
PY
CUDA_VISIBLE_DEVICES=-1 .tflite-venv/bin/python src/tflite_golden.py --run "$run" --quant-dir "$quantdir"
bash scripts/build/build_tflite_firmware.sh "$run" "$quantdir" "$firmwaredir"
.venv/bin/python src/verification.py --run "$run" --firmware-dir "$firmwaredir" --select
