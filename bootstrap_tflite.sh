#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
if [[ ! -d .tflite-venv ]]; then
  "$HOME/miniconda3/envs/edgenas/bin/python" -m venv .tflite-venv
fi
.tflite-venv/bin/python -m pip install tensorflow==2.16.2 tf-keras==2.16.0 \
  onnx2tf==1.22.3 onnx==1.16.1 onnxruntime==1.18.1 onnxsim==0.4.36 \
  onnx-graphsurgeon==0.5.2 sng4onnx==1.0.4 psutil
.tflite-venv/bin/python -m pip freeze > logs/tflite-environment.txt
if [[ ! -d upstream/esp-tflite-micro ]]; then
  git clone --depth 1 https://github.com/espressif/esp-tflite-micro.git upstream/esp-tflite-micro
fi
git -C upstream/esp-tflite-micro rev-parse HEAD > logs/esp-tflite-micro.commit
