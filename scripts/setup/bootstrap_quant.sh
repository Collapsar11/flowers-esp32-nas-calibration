#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
test -f upstream/esp-ppq/requirements.txt
if [[ ! -d .quant-venv ]]; then
  "$HOME/miniconda3/envs/edgenas/bin/python" -m venv --system-site-packages .quant-venv
fi
.quant-venv/bin/python - <<'PY'
from pathlib import Path
req = Path('upstream/esp-ppq/requirements.txt').read_text().splitlines()
Path('logs/quant-requirements.txt').write_text('\n'.join(x for x in req if not x.startswith(('torch', 'numpy'))) + '\nnumpy>=1.24,<2\n')
PY
.quant-venv/bin/python -m pip install -r logs/quant-requirements.txt
.quant-venv/bin/python -m pip freeze > logs/quant-environment.txt
PYTHONPATH="$PWD/upstream/esp-ppq" .quant-venv/bin/python -c 'from esp_ppq.api import espdl_quantize_onnx; print("ESP_PPQ_IMPORT_OK")'
