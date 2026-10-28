#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
root="$PWD"
export IDF_TOOLS_PATH="$root/toolchains"
export LD_LIBRARY_PATH="$root/toolchains/system-libs/root/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export PATH="$HOME/miniconda3/envs/edgenas/bin:$PATH"
source upstream/esp-idf/export.sh
cd firmware-measure
idf.py reconfigure
cmake --build build --parallel 4
idf.py size > build/size-report.txt
xtensa-esp-elf-g++ --version > build/compiler-version.txt
cd build
esptool.py --chip esp32s3 merge_bin --fill-flash-size 16MB -o flash_image.bin @flash_args
# Deliberately contains no serial device, flash, upload or monitor operation.
