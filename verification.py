"""Fail-closed server deployment acceptance. No physical hardware claims."""
import json
from pathlib import Path
import pipeline as p


def deployment_gate(run, firmware_dir=None):
    run = Path(run)
    cfg = p.config()
    checks = {}
    if firmware_dir is None:
        selected = run / "deployment-selection.json"
        firmware_dir = json.loads(selected.read_text())["firmware_directory"] if selected.exists() else "firmware"
    firmware = run / firmware_dir
    manifest_file = firmware / "manifest.json"
    qemu_file = firmware / "qemu-report.json"
    if not manifest_file.exists() or not qemu_file.exists():
        return {"passed": False, "reason": "missing firmware manifest or QEMU result"}
    manifest = json.loads(manifest_file.read_text())
    qemu = json.loads(qemu_file.read_text())
    expected_runtime = cfg["hardware"].get("runtime")
    if expected_runtime and manifest.get("runtime") != expected_runtime:
        return {"passed": False, "reason": "firmware runtime differs from the selected experiment runtime"}
    quant = run / manifest["quantization_directory"]
    if manifest.get("runtime") == "tflite_micro_esp_nn":
        return tflite_gate(run, quant, firmware, manifest, qemu, cfg)
    report_file, audit_file = quant / "report.json", quant / "integer_constraint_audit.json"
    if not report_file.exists() or not audit_file.exists():
        return {"passed": False, "reason": "missing quantization report or integer audit"}
    report = json.loads(report_file.read_text())
    checks["precision"] = report.get("quantization_bits", 8) == 8
    checks["quality"] = report.get("int8_top1") is not None and report["int8_top1"] >= cfg["quality"]["validation_top1_target"]
    checks["quantization_drop"] = report["quantization_drop"] <= cfg["quality"]["max_quantization_drop"]
    checks["full_validation"] = report["validation_count"] == 1020 and report["test_used"] is False
    checks["integer_constraints"] = json.loads(audit_file.read_text())["passed"]
    checks["storage"] = report["production_model_bytes"] <= cfg["hardware"]["flash_budget_bytes"]
    checks["qemu_functional"] = qemu["functional_pass"]
    for field, budget in [("validation_internal_peak_delta", "internal_ram_budget_bytes"),
                          ("validation_psram_peak_delta", "psram_budget_bytes")]:
        value = qemu.get(field)
        checks[field] = value is not None and 0 <= value <= cfg["hardware"][budget]
    checks["model_identity"] = (report["test_model_sha256"] == manifest["test_model_sha256"] and
        report["production_sha256"] == manifest["production_model_sha256"])
    checks["flash_identity"] = (manifest["flash_image_sha256"] == p.digest(firmware / "flash_image.bin") and
        qemu.get("flash_image_sha256") == manifest["flash_image_sha256"])
    return {"passed": all(checks.values()), "checks": checks, "real_hardware_validated": False}


def tflite_gate(run, quant, firmware, manifest, qemu, cfg):
    required = [quant / "report.json", quant / "model.tflite", quant / "model.audit.json",
        quant / "golden-protocol.json", firmware / "sdkconfig"]
    if not manifest.get("batches"):
        required.append(firmware / "flash_image.bin")
    if any(not f.exists() for f in required):
        return {"passed": False, "reason": "missing TFLite precision, export audit, golden protocol, or firmware artifact"}
    report = json.loads((quant / "report.json").read_text())
    audit = json.loads((quant / "model.audit.json").read_text())
    golden = json.loads((quant / "golden-protocol.json").read_text())
    quality, hardware = cfg["quality"], cfg["hardware"]
    reference = json.loads((run / "summary.json").read_text())["validation"]["top1"]
    accuracy = report.get("int8_top1", -1)
    model_sha = p.digest(quant / "model.tflite")
    checks = {
        "precision": report.get("quantization_bits") == 8 and report.get("all_tensors_integer") is True,
        "quality": accuracy >= quality["validation_top1_target"],
        "quantization_drop": reference - accuracy <= quality["max_quantization_drop"],
        "reference_identity": report.get("original_fp32_top1") == reference and abs(report.get("quantization_drop", 999) - (reference - accuracy)) < 1e-9,
        "full_validation": report.get("validation_count") == 1020 and report.get("test_used") is False,
        "fixed_graph_export": audit.get("passed") is True and audit.get("only_weight_bias_buffers_changed") is True,
        "storage": (quant / "model.tflite").stat().st_size == report.get("production_model_bytes") and report["production_model_bytes"] <= hardware["flash_budget_bytes"],
        "qemu_functional": qemu.get("functional_pass") is True,
        "strict_integer_parity": qemu.get("golden_count") == 102 and qemu.get("integer_mismatches") == 0 and qemu.get("max_integer_error") == 0,
        "golden_class_coverage": golden.get("count") == 102 and golden.get("per_class") == 1 and golden.get("split") == "val",
        "golden_identity": golden.get("production_sha256") == model_sha and
            golden.get("golden_inputs_sha256") == manifest.get("golden_inputs_sha256") and
            golden.get("golden_outputs_sha256") == manifest.get("golden_outputs_sha256"),
        "model_identity": model_sha == report.get("production_sha256") == audit.get("production_sha256") == manifest.get("production_model_sha256"),
        "flash_identity": True if manifest.get("batches") else p.digest(firmware / "flash_image.bin") == manifest.get("flash_image_sha256") == qemu.get("flash_image_sha256"),
        "embedded_model_identity": True if manifest.get("batches") else (firmware / "flash_image.bin").read_bytes().count((quant / "model.tflite").read_bytes()) == 1,
        "sdkconfig_identity": p.digest(firmware / "sdkconfig") == manifest.get("sdkconfig_sha256"),
    }
    for field, budget in [("validation_internal_peak_delta", "internal_ram_budget_bytes"),
                          ("validation_psram_peak_delta", "psram_budget_bytes"),
                          ("tensor_arena_reserved_bytes", "psram_budget_bytes")]:
        value = qemu.get(field)
        checks[field] = value is not None and 0 <= value <= hardware[budget]
    used, reserved = qemu.get("tensor_arena_used_bytes"), qemu.get("tensor_arena_reserved_bytes")
    checks["arena_measured"] = used is not None and reserved is not None and 0 < used <= reserved
    if manifest.get("batches"):
        checks.update(batch_checks(quant, firmware, manifest, qemu, model_sha))
    return {"passed": all(checks.values()), "checks": checks, "production_sha256": model_sha,
        "scope": "standalone_server_emulation_not_full_board_application", "real_hardware_validated": False,
        "latency_label_ms": None}


def batch_checks(quant, firmware, manifest, aggregate, model_sha):
    """Recheck original images and every binary/result; do not trust aggregation alone."""
    import hashlib
    import numpy as np
    checks = {"batch_artifacts": True, "batch_model_identity": True,
        "batch_coverage": True, "batch_golden_identity": True, "batch_strict_parity": True,
        "batch_memory_aggregation": True, "flash_identity": True, "embedded_model_identity": True}
    npz = quant / "golden-int8.npz"
    if not npz.exists():
        return {k: False for k in checks}
    gold = np.load(npz)
    model_bytes = (quant / "model.tflite").read_bytes()
    offset = 0
    measurements = []
    for batch in manifest["batches"]:
        directory = firmware / batch["directory"]
        files = [directory / n for n in ["manifest.json", "qemu-report.json", "flash_image.bin", "sdkconfig"]]
        if any(not f.exists() for f in files):
            checks["batch_artifacts"] = False
            continue
        m, q = [json.loads(f.read_text()) for f in files[:2]]
        count = batch["count"]
        checks["batch_coverage"] &= batch["offset"] == offset and 0 < count <= 102 - offset
        checks["batch_model_identity"] &= m.get("production_model_sha256") == model_sha
        checks["batch_model_identity"] &= p.digest(directory / "sdkconfig") == m.get("sdkconfig_sha256") == manifest.get("sdkconfig_sha256")
        for key in ["inputs", "outputs"]:
            expected = hashlib.sha256(gold[key][offset:offset+count].tobytes()).hexdigest()
            checks["batch_golden_identity"] &= m.get("golden_" + key + "_sha256") == expected
        checks["batch_strict_parity"] &= q.get("functional_pass") is True and q.get("golden_count") == count and q.get("integer_mismatches") == 0 and q.get("max_integer_error") == 0
        checks["flash_identity"] &= p.digest(directory / "flash_image.bin") == m.get("flash_image_sha256") == q.get("flash_image_sha256") == batch["flash_image_sha256"]
        checks["embedded_model_identity"] &= (directory / "flash_image.bin").read_bytes().count(model_bytes) == 1
        measurements.append(q)
        offset += count
    checks["batch_coverage"] &= offset == 102 and len(gold["inputs"]) == len(gold["outputs"]) == 102
    for key in ["validation_internal_peak_delta", "validation_psram_peak_delta", "tensor_arena_reserved_bytes", "tensor_arena_used_bytes"]:
        checks["batch_memory_aggregation"] &= bool(measurements) and all(q.get(key) is not None for q in measurements) and aggregate.get(key) == max(q.get(key, -1) for q in measurements)
    return checks


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--firmware-dir")
    ap.add_argument("--select", action="store_true", help="Select this deployment only if every gate passes")
    args = ap.parse_args()
    run = p.ROOT / "runs" / args.run
    result = deployment_gate(run, args.firmware_dir)
    print(json.dumps(result, indent=2))
    if args.select:
        if not result["passed"] or not args.firmware_dir:
            raise SystemExit("Deployment selection refused: all gates and explicit firmware directory required")
        p.save_json(run / "deployment-selection.json", {"firmware_directory": args.firmware_dir, **result})
    if not result["passed"]:
        raise SystemExit(1)
