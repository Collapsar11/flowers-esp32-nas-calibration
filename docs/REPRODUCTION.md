# Reproduction scope

## 1. Published aggregate reproduction

Run `reproduce_results.py` with the dependencies in `requirements-analysis.txt`. It checks the six-network support split, fits only those six labels, reproduces the original predictions and decisions, then evaluates the nine query observations. `test_strict6.py` provides synthetic protocol rejection and numerical correctness tests; its fixtures are not hardware evidence.

`results/published/frozen-predictions.json` is an export of selected fields from the original lock, **not the original immutable lock file**. The original lock hash appears in `provenance.json`. The repository alone cannot authenticate raw acquisition, individual serial samples, or board identity.

## 2. Full raw-record audit

Use the separate local `flowers-esp32-code-and-results-20261028.zip` delivery. Its `verify_delivery.py` verifies every delivered file hash and replays held-out evaluation from the original receipt/serial records. Original complete-audit scripts additionally refer to a private original-firmware backup that is intentionally not included in either delivery or Git.

Historical `physical_board_flashed: false` fields in the test-stage summary describe that earlier software-only test stage. The latest hardware results are under `results/published/`; these historic fields are not current hardware status.

## 3. Training and firmware reconstruction

The training scripts originate from a Linux GPU server environment, with separate PyTorch and TensorFlow environments. They retain that directory layout and do not constitute a newly validated one-command installation on arbitrary hosts.

Relevant entry points:

- `bootstrap_server.sh`: OFA/MCUNet source setup and dataset preparation; originally uses an existing Conda environment.
- `bootstrap_tflite.sh`: TensorFlow 2.16.2, tf-keras 2.16.0 and conversion dependencies.
- `bootstrap_esp.sh`: ESP-IDF v5.5.1 and Xtensa QEMU tooling.
- `pipeline.py`, `search_parallel.py`, `campaign.py`: pretrained search, full candidate training and verification.
- `qat_tflite.py`, `build_tflite_firmware.sh`, `build_measure_firmware.sh`: accepted TFLite Micro path.

Prepare Flowers 102 and the upstream pretrained weights under their own terms. Install suitable PyTorch/torchvision and the conversion/toolchain dependencies. Check the bootstrap scripts and adapt their environment paths. Some upstream checkout commands track a moving branch; exact reconstruction should use the original experiment's recorded revisions/build manifests from the complete evidence delivery rather than assume a fresh clone is identical.

This upload did not rerun NAS, training, QAT, firmware builds or hardware measurement. Checkpoints, models, golden-input binaries, third-party sources and build intermediates are excluded from Git. The full local delivery supplies final TFLite models and measurement flash images, but not every server training checkpoint/toolchain.

## 4. Hardware scripts

The accepted board configuration is ESP32-S3-N16R8, CPU 240 MHz, 8 MB Octal PSRAM at 80 MHz and 16 MB Flash. The measurement firmware is a standalone harness, not a complete integration with the original board application.

For the four original field acquisition/audit scripts, board-specific literals were replaced by required `FLOWERS_BOARD_MAC` and, for acquisition, `FLOWERS_SERIAL_PORT` environment variables. Missing configuration fails at import rather than selecting another device automatically. These scripts also require the original compatible firmware/manifests, prior backup evidence and frozen campaign directory structure. Merely setting the environment variables is not sufficient to run a new experiment.

The source-change manifest records original source hashes. The published numerical core (`strict6.py`, `measurement_io.py`) and frozen protocol/public features were not changed. Original local code and evidence remain intact. New campaigns must use separate output directories and preserve the strict six-support/nine-query boundary.

## 5. Interpretation

The main method was the structural ridge model. Do not relabel the better proportional baseline as the prespecified primary method after observing results. Raw MACs is a ranking baseline, not a millisecond estimate. QEMU timing is never physical-device latency evidence. Golden-output agreement establishes functional equivalence, not 100% classification accuracy.

Selection used validation accuracy; test accuracy is final reporting only. The selected model's 89.4942% test accuracy is below 90%, despite successful hardware/resource checks. No retrospective replacement was performed.
