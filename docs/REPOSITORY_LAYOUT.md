# 目录与运行入口

根目录只保留 README 和项目目录，Python、Shell、依赖清单和配置分别放入子目录。

| 目录 | 内容 | 常用入口 |
|---|---|---|
| `src/` | Python 实现与命令入口 | `reproduce_results.py`、`strict6.py`、`pipeline.py`、`campaign.py` |
| `scripts/setup/` | 环境准备 | `bootstrap_server.sh`、`bootstrap_tflite.sh`、`bootstrap_esp.sh` |
| `scripts/build/` | 固件构建 | `build_measure_firmware.sh`、`build_tflite_firmware.sh` |
| `scripts/run/` | 运行与验收 | `run_pilots.sh`、`accept_tflite_qat.sh` |
| `tests/` | 自动测试 | `test_strict6.py`；`test_campaign_evidence.py` 还需要训练环境依赖 |
| `requirements/` | 依赖清单 | `analysis.txt` 用于结果复算；`measurement.txt` 加入串口采集依赖 |
| `configs/` | 搜索、训练和硬件约束 | `config.json` |
| `architectures/` | 最终 15 个网络的架构 | 每个网络一个 JSON |
| `results/` | 原冻结协议及可复算汇总数据 | `published/summary.json`、`published/evaluation.json` |
| `docs/` | 实验说明 | `EXPERIMENT_RESULTS.md`、`REPRODUCTION.md` |
| `firmware-measure/` | 实板测量固件源码 | `main/app_main.cpp` |
| `firmware-tflite/` | TFLite 验证固件源码 | `main/app_main.cpp` |
| `firmware/` | 历史 ESP-DL 固件源码 | 非最终采用的运行时 |

## 复算已有结果

以下命令在仓库根目录执行，不需要连接 ESP32：

```bash
python3 -m venv .venv-review
.venv-review/bin/python -m pip install -r requirements/analysis.txt
.venv-review/bin/python src/reproduce_results.py
.venv-review/bin/python -m unittest discover -s tests -p 'test_strict6.py' -v
```

Python 文件在 `src/` 中仍以原模块名互相导入。直接执行 `python src/文件名.py` 时，Python 会将 `src/` 加入模块搜索路径；测试文件会显式加入该目录。`scripts/` 中的 Shell 入口会先切换到仓库根目录，并设置 `PYTHONPATH`，供内嵌 Python 使用。

## 与原始实验的关系

本次只整理 GitHub 对应的本地仓库，没有迁移原始实验目录、服务器运行目录或已交付 ZIP。原始实验的 `runs/`、模型权重和固件二进制不在 Git 中；重新训练、构建或采集仍需按复现文档准备完整环境和输入。

文件移动清单见 [file-moves.json](file-moves.json)。目录变化同步修正了 Python 的项目根路径、子进程入口、Shell 工作目录、文档和 GitHub Actions。实验算法和 `results/` 中的数据不变，但源码文件哈希会因路径改动而变化；不要修改旧预测锁来适配新目录。旧证据的原始审计继续使用原交付包。
