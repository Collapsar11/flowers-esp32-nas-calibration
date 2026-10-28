#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p upstream logs
if [ ! -d upstream/esp-idf/.git ]; then
  git clone --depth 1 --branch v5.5.1 --recursive --shallow-submodules https://github.com/espressif/esp-idf.git upstream/esp-idf
fi
if [ ! -d upstream/esp-dl/.git ]; then
  git clone --depth 1 https://github.com/espressif/esp-dl.git upstream/esp-dl
fi
if [ ! -d upstream/esp-ppq/.git ]; then
  git clone --depth 1 https://github.com/espressif/esp-ppq.git upstream/esp-ppq
fi
for component in esp-idf esp-dl esp-ppq; do
  git -C "upstream/$component" rev-parse HEAD > "logs/$component.commit"
done
export IDF_TOOLS_PATH="$PWD/toolchains"
if ! ldconfig -p | rg -q 'libslirp.so.0' && [[ ! -e toolchains/system-libs/root/usr/lib/x86_64-linux-gnu/libslirp.so.0 ]]; then
  mkdir -p toolchains/system-libs
  (cd toolchains/system-libs && apt download libslirp0 && dpkg-deb -x libslirp0*.deb root)
fi
export LD_LIBRARY_PATH="$PWD/toolchains/system-libs/root/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
# The server's system Python lacks ensurepip; the isolated Conda Python supports venv.
export PATH="$HOME/miniconda3/envs/edgenas/bin:$PATH"
bash upstream/esp-idf/install.sh esp32s3
source upstream/esp-idf/export.sh
python upstream/esp-idf/tools/idf_tools.py install qemu-xtensa
if [[ ! -x toolchains/qemu-20260417/qemu/bin/qemu-system-xtensa ]]; then
  mkdir -p toolchains/qemu-20260417
  curl -L --fail --silent --show-error https://github.com/espressif/qemu/releases/download/esp-develop-9.2.2-20260417/qemu-xtensa-softmmu-esp_develop_9.2.2_20260417-x86_64-linux-gnu.tar.xz -o toolchains/dist/qemu-20260417.tar.xz
  tar -xf toolchains/dist/qemu-20260417.tar.xz -C toolchains/qemu-20260417
fi
toolchains/qemu-20260417/qemu/bin/qemu-system-xtensa --version > logs/qemu-version.txt
sha256sum toolchains/dist/qemu-20260417.tar.xz > logs/qemu-archive.sha256
echo 'ESP-IDF / S3 / QEMU tool installation complete; no board was flashed.'
