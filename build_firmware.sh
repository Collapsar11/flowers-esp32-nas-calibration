#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
root="$PWD"
run="${1:?provide the run directory name}"
quantdir="${2:-int8-esp32s3}"
firmware_dir="${3:-firmware}"
test -f "runs/$run/$quantdir/model-test.espdl"
export IDF_TOOLS_PATH="$root/toolchains"
export LD_LIBRARY_PATH="$root/toolchains/system-libs/root/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export PATH="$HOME/miniconda3/envs/edgenas/bin:$PATH"
source upstream/esp-idf/export.sh
cp "runs/$run/$quantdir/model-test.espdl" firmware/main/model.espdl
cd firmware
idf.py reconfigure
cmake --build build --parallel 4
idf.py size > "$root/runs/$run/$quantdir/firmware-size.txt"
cd build
esptool.py --chip esp32s3 merge_bin --fill-flash-size 16MB -o flash_image.bin @flash_args
mkdir -p "$root/runs/$run/$firmware_dir"
cp flash_image.bin flowers_validation.bin flowers_validation.elf "$root/runs/$run/$firmware_dir/"
cp ../sdkconfig "$root/runs/$run/$firmware_dir/"
cd "$root"
.venv/bin/python - "$run" "$quantdir" "$firmware_dir" <<'PY'
import sys, pathlib, json, hashlib
root = pathlib.Path.cwd()
run, quantdir, firmware_dir = sys.argv[1:]
quant = root / 'runs' / run / quantdir
out = root / 'runs' / run / firmware_dir
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
manifest = {'quantization_directory': quantdir, 'test_model_sha256': sha(quant / 'model-test.espdl'),
            'production_model_sha256': sha(quant / 'model.espdl'), 'flash_image_sha256': sha(out / 'flash_image.bin'),
            'sdkconfig_sha256': sha(out / 'sdkconfig'), 'internal_activation_pool_bytes': 65536,
            'weights_residency': 'flash', 'real_hardware_validated': False}
(out / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
PY
.venv/bin/python qemu_check.py --run "$run" --firmware-dir "$firmware_dir"
