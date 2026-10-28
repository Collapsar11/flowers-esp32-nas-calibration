#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
root="$PWD"
run="${1:?run name required}"
quantdir="${2:-tflite-int8-v2}"
outdir="${3:-firmware-$quantdir}"
offset="${4:-0}"
count="${5:-0}"
.venv/bin/python - "$run" "$quantdir" "$offset" "$count" <<'PY'
import sys, pathlib, shutil, numpy as np
root = pathlib.Path.cwd()
source = root / 'runs' / sys.argv[1] / sys.argv[2]
dest = root / 'firmware-tflite/main'
# Refresh mtime as well as bytes: switching back to an older checkpoint must
# invalidate the shared CMake embedded-model object.
shutil.copyfile(source / 'model.tflite', dest / 'model.tflite')
data = np.load(source / 'golden-int8.npz')
offset, count = map(int, sys.argv[3:5])
end = offset + count if count else len(data['inputs'])
assert 0 <= offset < end <= len(data['inputs'])
data['inputs'][offset:end].tofile(dest / 'golden-inputs.bin')
data['outputs'][offset:end].tofile(dest / 'golden-outputs.bin')
PY
export IDF_TOOLS_PATH="$root/toolchains"
export LD_LIBRARY_PATH="$root/toolchains/system-libs/root/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export PATH="$HOME/miniconda3/envs/edgenas/bin:$PATH"
source upstream/esp-idf/export.sh
cd firmware-tflite
if [[ ! -f sdkconfig ]]; then idf.py set-target esp32s3; fi
idf.py reconfigure
cmake --build build --parallel 4
cd build
esptool.py --chip esp32s3 merge_bin --fill-flash-size 16MB -o flash_image.bin @flash_args
mkdir -p "$root/runs/$run/$outdir"
cp flash_image.bin flowers_tflite_validation.elf flowers_tflite_validation.bin "$root/runs/$run/$outdir/"
cp ../sdkconfig "$root/runs/$run/$outdir/"
cd "$root"
.venv/bin/python - "$run" "$quantdir" "$outdir" "$offset" "$count" <<'PY'
import sys, pathlib, json, hashlib
root = pathlib.Path.cwd()
run, quantdir, outdir = sys.argv[1:4]
source, out = root / 'runs' / run / quantdir, root / 'runs' / run / outdir
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
manifest = {'runtime': 'tflite_micro_esp_nn', 'quantization_directory': quantdir,
    'production_model_sha256': sha(source / 'model.tflite'),
    'golden_inputs_sha256': sha(root / 'firmware-tflite/main/golden-inputs.bin'),
    'golden_outputs_sha256': sha(root / 'firmware-tflite/main/golden-outputs.bin'),
    'flash_image_sha256': sha(out / 'flash_image.bin'), 'sdkconfig_sha256': sha(out / 'sdkconfig'),
    'weights_residency': 'flash', 'tensor_arena_residency': 'psram',
    'arena_policy': 'runtime_required_size_aligned_plus_4096', 'real_hardware_validated': False}
manifest['golden_offset'] = int(sys.argv[4])
manifest['golden_count_requested'] = int(sys.argv[5])
(out / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
PY
if [[ "${TFLM_BUILD_ONLY:-0}" != "1" ]]; then
    .venv/bin/python src/qemu_check.py --run "$run" --firmware-dir "$outdir" --timeout "${TFLM_QEMU_TIMEOUT:-600}"
fi
