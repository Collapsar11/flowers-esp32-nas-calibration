"""Immutable QAT export -> complete class coverage in bounded QEMU payloads."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from execution import preserve_cpu_affinity
preserve_cpu_affinity()
import numpy as np
import pipeline as p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--qat", required=True)
    ap.add_argument("--quant", default="tflite-final")
    ap.add_argument("--max-batch-samples", type=int, default=102)
    a = ap.parse_args()
    run = p.ROOT / "runs" / a.run
    source = run / a.qat / "best-export"
    selected = json.loads((run / a.qat / "best.json").read_text())
    report = json.loads((source / "report.json").read_text())
    cfg = p.config()
    reference = json.loads((run / "summary.json").read_text())["validation"]["top1"]
    assert report["int8_top1"] >= cfg["quality"]["validation_top1_target"]
    assert reference - report["int8_top1"] <= cfg["quality"]["max_quantization_drop"]
    assert report["production_model_bytes"] <= cfg["hardware"]["flash_budget_bytes"]
    assert p.digest(source / "model.tflite") == selected["production_sha256"] == report["production_sha256"]
    quant = run / a.quant
    if not quant.exists():
        quant.mkdir()
        for filename in ["model.tflite", "report.json", "model.audit.json"]:
            shutil.copy2(source / filename, quant / filename)
        p.save_json(quant / "training-provenance.json", {"qat_directory": a.qat,
            "selected_epoch": selected["epoch"], "checkpoint_sha256": p.digest(run / a.qat / "best.pt"),
            "production_sha256": report["production_sha256"], "test_used": False})
    assert p.digest(quant / "model.tflite") == report["production_sha256"]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1")
    if not (quant / "golden-protocol.json").exists():
        subprocess.run([str(p.ROOT / ".tflite-venv/bin/python"), "tflite_golden.py",
            "--run", a.run, "--quant-dir", a.quant], check=True, cwd=p.ROOT, env=env)
    gold = np.load(quant / "golden-int8.npz")
    protocol = json.loads((quant / "golden-protocol.json").read_text())
    assert len(gold["inputs"]) == 102
    # 14 MiB app partition; conservatively leave 1 MiB for runtime/metadata.
    sample_bytes = gold["inputs"][0].nbytes + gold["outputs"][0].nbytes
    batch_size = min(102, a.max_batch_samples, (14 * 1024**2 - report["production_model_bytes"] - 1024**2) // sample_bytes)
    assert batch_size >= 1
    fwname = "firmware-" + a.quant
    firmware = run / fwname
    firmware.mkdir(exist_ok=True)
    batches = []
    for number, offset in enumerate(range(0, 102, batch_size)):
        count = min(batch_size, 102 - offset)
        name = "batch-%03d" % number
        directory = firmware / name
        relative = fwname + "/" + name
        if not (directory / "manifest.json").exists():
            with (p.ROOT / "logs/tflite-firmware.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                subprocess.run(["bash", "build_tflite_firmware.sh", a.run, a.quant,
                    relative, str(offset), str(count)], check=True, cwd=p.ROOT,
                    env=dict(env, TFLM_BUILD_ONLY="1"))
        manifest = json.loads((directory / "manifest.json").read_text())
        assert manifest["production_model_sha256"] == report["production_sha256"]
        assert manifest["golden_inputs_sha256"] == hashlib.sha256(gold["inputs"][offset:offset+count].tobytes()).hexdigest()
        qfile = directory / "qemu-report.json"
        if not qfile.exists() or not json.loads(qfile.read_text()).get("functional_pass"):
            libpath = str(p.ROOT / "toolchains/system-libs/root/usr/lib/x86_64-linux-gnu")
            subprocess.run([sys.executable, "qemu_check.py", "--run", a.run, "--firmware-dir", relative,
                "--timeout", "1800"], check=True, cwd=p.ROOT,
                env=dict(env, LD_LIBRARY_PATH=libpath + ":" + env.get("LD_LIBRARY_PATH", "")))
        result = json.loads(qfile.read_text())
        assert result["functional_pass"] and result["golden_count"] == count
        assert result["integer_mismatches"] == result["max_integer_error"] == 0
        batches.append({"directory": name, "offset": offset, "count": count,
            "flash_image_sha256": manifest["flash_image_sha256"], "result": result})
    shutil.copy2(firmware / batches[0]["directory"] / "sdkconfig", firmware / "sdkconfig")
    manifest = {"runtime": "tflite_micro_esp_nn", "quantization_directory": a.quant,
        "production_model_sha256": report["production_sha256"],
        "golden_inputs_sha256": protocol["golden_inputs_sha256"],
        "golden_outputs_sha256": protocol["golden_outputs_sha256"],
        "sdkconfig_sha256": p.digest(firmware / "sdkconfig"),
        "batches": [{k: v for k, v in b.items() if k != "result"} for b in batches],
        "weights_residency": "flash", "tensor_arena_residency": "psram",
        "real_hardware_validated": False}
    aggregate = {"functional_pass": True, "reason": "all_class_batches_completed",
        "scope": "standalone_inference_harness_not_full_board_application", "golden_count": 102,
        "class_agreement": 102, "integer_mismatches": 0, "max_integer_error": 0,
        "batch_count": len(batches), "real_hardware_validated": False, "latency_label_ms": None,
        "host_wall_seconds_not_device_latency": sum(b["result"]["host_wall_seconds_not_device_latency"] for b in batches)}
    for key in ["validation_internal_peak_delta", "validation_psram_peak_delta",
                "tensor_arena_reserved_bytes", "tensor_arena_used_bytes"]:
        aggregate[key] = max(b["result"][key] for b in batches)
    p.save_json(firmware / "manifest.json", manifest)
    p.save_json(firmware / "qemu-report.json", aggregate)
    subprocess.run([sys.executable, "verification.py", "--run", a.run,
        "--firmware-dir", fwname, "--select"], check=True, cwd=p.ROOT, env=env)


if __name__ == "__main__":
    main()
