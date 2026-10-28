# Flowers 102 NAS and six-sample ESP32 latency calibration

IERG5250 course experiment: generate and train edge-compatible image classifiers, validate INT8 deployment, then use **six measured architectures** to calibrate hardware estimates and select among **nine held-out architectures**.

本仓库保存实验代码、15 个候选的架构和可复算的汇总结果。原始实板日志、模型权重、测量固件和完整交付包保存在本地，不提交到 Git 历史。

## 实验流程与结果

**实验流程：**
15 个合格候选 → 实测其中 6 个 → 校准硬件指标估计 → 预测剩余 9 个 → 根据精度、内存和延迟要求筛选 → 选出 1 个网络 → 评估指标。

这里的“合格”指通过验证集精度、模型存储、内存和仿真检查，不代表所有网络的最终测试精度都达到 90%。

| 步骤 | 做了什么 | 得到的结果 |
|---|---|---|
| 1. 得到 15 个候选 | 用 NAS 搜索网络，完成训练、量化和仿真验证 | 15 个网络通过验证集精度和资源约束检查 |
| 2. 实测其中 6 个 | 在 ESP32-S3 上测量，每个网络做 3 次独立内存采样和 3 次独立计时采样 | 得到 6 个网络的真实延迟和内存数据 |
| 3. 校准硬件估计 | 用这 6 个网络的数据，建立 MACs／结构指标到真实延迟的映射，并检查内存偏差 | 得到校准模型；内存修正量为 0 |
| 4. 预测剩余 9 个 | 预测未参与校准的 9 个网络，冻结预测结果 | 得到 9 个网络的预计延迟和内存占用 |
| 5. 按要求筛选 | 先检查验证集精度 ≥90%、模型存储和内存限制，再比较预计延迟 | 在满足这些约束的候选中选择预计延迟最低的网络 |
| 6. 选出 1 个网络 | 锁定 `0ec51e472c33beae`，再进行实板验证并报告最终测试精度 | 预测延迟 **1364.80 ms**，实测 **1333.58 ms**；内存通过；测试精度 **89.49%** |
| 7. 评估指标 | 实测全部 9 个查询网络，对比冻结预测；不用于重新校准或选型 | 主方法平均延迟误差 **6.06%**；简单 MACs 比例校准误差 **3.74%**；选中的网络确实是这 9 个中最快的 |

## 结论

实验已经跑完。在当前这块板和这批候选上，用 6 个网络的实测数据，能够较准确地预测另外 9 个网络的延迟，并选中实际最快的网络，说明少量实测校准这条路线是可行的。

但目前复杂校准方法没有超过简单基线：主方法平均误差为 **6.06%**，简单 MACs 比例校准为 **3.74%**；直接按 MACs 排序也选中了同一个网络。内存仿真本来就准确，因此没有体现额外的内存校准收益。

另外，选中网络的测试精度只有 **89.49%**，如果最终要求测试精度 ≥90%，就还没有完全达标。**目前证明了流程可行，但没有证明复杂方法更优，也还不能说全部性能要求都满足了。**

决策阶段只用了 **6 个网络的硬件标签**；为了验证结果，评估阶段又测了其余 9 个。不能说“总共只实测了 6 个”。相对穷举 15 个，决策所需标签数量减少 60%，不等于整个实验耗时减少 60%。

完整口径见 [队友同步版](docs/EXPERIMENT_RESULTS.md)。

## Hardware and detailed metrics

Hardware: ESP32-S3-N16R8, 16 MB Flash, 8 MB Octal PSRAM; CPU 240 MHz, PSRAM 80 MHz. Runtime: TFLite Micro with ESP-NN. Input: 160 × 160 RGB images. This is image **classification**, using a public dataset; no camera or cloud upload is required.

Each architecture has three independent memory boots and three independent timing boots. Each timing boot uses 10 warmups and 204 timed invocations. The latency label is the median of three session medians. Memory labels are the maximum across three sessions. Timing covers `Invoke`, excluding input transfer, serial logging and output comparisons.

Independent evaluation on the nine held-out networks:

| Method | Calibration labels | Latency MAPE | MAE (ms) | Spearman |
|---|---:|---:|---:|---:|
| Raw MACs ranking | 0 | N/A | N/A | 1.0000 |
| Proportional MACs calibration | 6 | **3.7371%** | **81.081** | 1.0000 |
| Log-MACs ridge | 6 | 6.3134% | 128.289 | 1.0000 |
| Structure ridge, prespecified primary | 6 | 6.0550% | 136.554 | 0.9667 |

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
