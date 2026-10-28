#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p upstream logs data runs cache
if [ ! -x .venv/bin/python ]; then
  "$HOME/miniconda3/envs/edgenas/bin/python" -m venv --system-site-packages .venv
fi
if [ ! -d upstream/once-for-all/.git ]; then
  git clone --depth 1 https://github.com/mit-han-lab/once-for-all.git upstream/once-for-all
fi
if [ ! -d upstream/mcunet/.git ]; then
  git clone --depth 1 https://github.com/mit-han-lab/mcunet.git upstream/mcunet
fi
.venv/bin/python -m pip install --no-deps deap==1.4.3 gdown==5.2.0
# Import upstream source directly: MCUNet's dated setup.py emits a version
# rejected by modern packaging. Keep upstream source unmodified.
.venv/bin/python -m pip freeze > logs/environment.txt
git -C upstream/once-for-all rev-parse HEAD > logs/ofa.commit
git -C upstream/mcunet rev-parse HEAD > logs/mcunet.commit
.venv/bin/python pipeline.py prepare
NOTICE='No physical board access. Emulator timing must never become ESP32 latency training labels.'
echo "$NOTICE"
