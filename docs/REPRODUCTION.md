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

统一对外表述见 [实验流程与结果（队友同步版）](EXPERIMENT_RESULTS.md)。

实验已经跑完。在当前这块板和这批候选上，用 6 个网络的实测数据，能够较准确地预测另外 9 个网络的延迟，并选中实际最快的网络，说明少量实测校准这条路线是可行的。

但目前复杂校准方法没有超过简单基线：主方法平均误差为 **6.06%**，简单 MACs 比例校准为 **3.74%**；直接按 MACs 排序也选中了同一个网络。内存仿真本来就准确，因此没有体现额外的内存校准收益。

另外，选中网络的测试精度只有 **89.49%**，如果最终要求测试精度 ≥90%，就还没有完全达标。**目前证明了流程可行，但没有证明复杂方法更优，也还不能说全部性能要求都满足了。**

决策阶段只用了 **6 个网络的硬件标签**；为了验证结果，评估阶段又测了其余 9 个。不能说“总共只实测了 6 个”。相对穷举 15 个，决策所需标签数量减少 60%，不等于整个实验耗时减少 60%。

## 指标口径

- 平均延迟误差指 9 个查询网络上的 MAPE（平均绝对百分比误差）；主方法为 `structure_ridge_primary`，简单基线为 `scaled_macs`。
- 选中网络的延迟预测误差为 **2.34%**；所有方法都选中同一个网络，未体现主方法相对 MACs 的额外选型收益。
- 精度指 Flowers 102 的 Top-1 分类准确率。选型使用验证集；最终测试集只作报告，不参与重新选型。
- 模型存储限制为 12 MiB，模型相关 PSRAM 限制为 6 MiB，增量内部堆限制为 192 KiB；这些内存指标不等于整机总 SRAM 占用。
- 延迟是固定运行时下的 `Invoke` 耗时。没有预先定义实际应用的延迟上限，所以“选中最快”不代表达到某个实际应用的速度要求。
- 原始结果、预测锁和审计记录保持不变。本说明只统一实验流程和结果表述。

The main method remains structural ridge; the better proportional method remains a baseline. QEMU timing is not physical-device latency evidence. Golden-output agreement is functional equivalence, not 100% classification accuracy.
