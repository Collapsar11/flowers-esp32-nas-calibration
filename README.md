# Flowers 102 NAS and six-sample ESP32 latency calibration

IERG5250 course experiment: generate and train edge-compatible image classifiers, validate INT8 deployment, then use **six measured architectures** to calibrate hardware estimates and select among **nine held-out architectures**.

本仓库保存实验代码、15 个候选的架构和可复算的汇总结果。原始实板日志、模型权重、测量固件和完整交付包保存在本地，不提交到 Git 历史。

## Experiment

1. Search a pretrained OFA MobileNetV3 space, then fully train and quantize candidates on Flowers 102.
2. Freeze 15 candidates after validation accuracy, model storage, memory and TFLite Micro/QEMU checks.
3. Measure six fixed support networks on ESP32-S3 and freeze calibration, predictions and decisions.
4. Predict the remaining nine and choose a network using validation accuracy and hardware constraints.
5. Measure those nine only for independent evaluation. No query-label fitting or retrospective reselection.

Hardware: ESP32-S3-N16R8, 16 MB Flash, 8 MB Octal PSRAM; CPU 240 MHz, PSRAM 80 MHz. Runtime: TFLite Micro with ESP-NN. Input: 160 × 160 RGB images. This is image **classification**, using a public dataset; no camera or cloud upload is required.

Each architecture has three independent memory boots and three independent timing boots. Each timing boot uses 10 warmups and 204 timed invocations. The latency label is the median of three session medians. Memory labels are the maximum across three sessions. Timing covers `Invoke`, excluding input transfer, serial logging and output comparisons.

## Results

Independent evaluation on the nine held-out networks:

| Method | Calibration labels | Latency MAPE | MAE (ms) | Spearman |
|---|---:|---:|---:|---:|
| Raw MACs ranking | 0 | N/A | N/A | 1.0000 |
| Proportional MACs calibration | 6 | **3.7371%** | **81.081** | 1.0000 |
| Log-MACs ridge | 6 | 6.3134% | 128.289 | 1.0000 |
| Structure ridge, prespecified primary | 6 | 6.0550% | 136.554 | 0.9667 |

All methods selected `0ec51e472c33beae`, the actual fastest held-out candidate. Primary prediction: **1364.798 ms**; measured: **1333.577 ms**; error: **2.3412%**. Selected model storage is 3,593,216 bytes and model-related PSRAM peak is 757,128 bytes.

The measured results support low-measurement-budget **absolute latency calibration** on this board and candidate set. They do **not** demonstrate that the primary structural model improves over the simple proportional baseline, or that calibration improves selection over raw MACs. The QEMU memory baseline was already exact for the measured metrics; the six-label memory correction was zero.

**Accuracy limitation:** selection used validation accuracy ≥90%. The selected model has validation Top-1 **93.2353%**, but final test Top-1 **89.4942%**. It does not satisfy a final-test ≥90% requirement. Twelve of the 15 candidates reach 90% on the full test set. Test labels were not used for training or selection.

Six of 15 hardware labels were used to make the decision: a **60% label-count reduction** compared with exhaustive acquisition. The evaluation campaign nevertheless measured all 15. This is not a claim of 60% end-to-end time saving. Results cover one board, fixed clocks/runtime and one candidate set; no application latency SLA or cross-device generalization has been established. Internal heap increments do not measure total SRAM usage.

## Reproduce the published numbers

Python 3.9–3.11 and NumPy 1.24.4 suffice; no GPU, dataset or board is needed for this step:

```bash
python3 -m venv .venv-review
.venv-review/bin/python -m pip install -r requirements-analysis.txt
.venv-review/bin/python reproduce_results.py
.venv-review/bin/python -m unittest test_strict6 -v
```

`reproduce_results.py` rebuilds calibration using only the six support aggregates, checks it against the exported frozen parameters/predictions, and then computes the held-out metrics and decisions. It does not modify the experiment or use query labels for fitting. This is **numerical reproduction from aggregates**, not a fresh hardware experiment or a raw-log provenance audit.

## Repository map

| Path | Purpose |
|---|---|
| `config.json`, `architectures/` | Search/training configuration and 15 final architectures |
| `pipeline.py`, `search_parallel.py`, `campaign.py` | Pretrained training, NAS and candidate campaign |
| `qat_tflite.py`, `convert_tflite.py`, `validate_candidate.py` | QAT, INT8 export and validation |
| `final_test.py` | Full Flowers 102 test-set evaluation |
| `firmware-measure/`, `build_measure_firmware.sh` | Measurement firmware source/build |
| `firmware-tflite/` | TFLite deployment validation firmware |
| `firmware/`, `quantize.py`, `qat.py` | Earlier experimental ESP-DL path; not the accepted final runtime |
| `strict6.py`, `measurement_io.py` | Frozen protocol, calibration, selection and record validation |
| `results/strict6-esp32-v1/` | Original design and public features, unchanged |
| `results/published/` | Derived aggregate export, frozen predictions, metrics, accuracy and provenance |
| `reproduce_results.py` | Portable numeric reproduction |
| `docs/REPRODUCTION.md` | Training, firmware and full-evidence boundaries |

The original acquisition/audit scripts are included for traceability. Their serial port and device MAC are now required environment variables; they are **not** portable one-click flash commands. See [reproduction notes](docs/REPRODUCTION.md) before using hardware scripts.

## Evidence and upstream work

The published aggregate exports are explicitly marked as derivatives. `results/published/provenance.json` records hashes of the original immutable prediction lock, evaluation, summary and audit, plus the source changes made for this repository. The original local evidence and full delivery ZIP remain unchanged.

The full local delivery includes 15 INT8 models, 15 measurement firmware images, 36 support receipts, 54 query receipts, raw logs, test predictions and an offline raw-record replay script. It is separate from this lightweight source repository. Original device firmware backups are excluded from that delivery and from this repository.

This project builds on [Once-for-All](https://github.com/mit-han-lab/once-for-all), [MCUNet](https://github.com/mit-han-lab/mcunet), [ESP-IDF](https://github.com/espressif/esp-idf), [ESP-TFLite-Micro](https://github.com/espressif/esp-tflite-micro) and the [Oxford Flowers 102 dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/). Upstream code, datasets, weights and their licenses are not bundled. No additional license grant is declared for this repository.
