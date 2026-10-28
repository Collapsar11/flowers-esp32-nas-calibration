"""Run only a local QEMU image. No serial devices or physical flash commands."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import selectors
import shutil
import subprocess
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--firmware-dir", default="firmware")
    args = p.parse_args()
    root = Path(__file__).resolve().parent
    out = root / "runs" / args.run / args.firmware_dir
    pinned = root / "toolchains/qemu-20260417/qemu/bin/qemu-system-xtensa"
    binary = str(pinned) if pinned.exists() else shutil.which("qemu-system-xtensa")
    if not binary:
        binary = str(next((root / "toolchains/tools/qemu-xtensa").glob("*/qemu/bin/qemu-system-xtensa")))
    cmd = [binary, "-nographic", "-machine", "esp32s3", "-m", "8M",
           "-global", "driver=ssi_psram,property=is_octal,value=true",
           "-global", "driver=timer.esp32c3.timg,property=wdt_disable,value=true",
           "-drive", f"file={out / 'flash_image.bin'},if=mtd,format=raw"]
    start = time.monotonic()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    selector = selectors.DefaultSelector()
    selector.register(proc.stdout, selectors.EVENT_READ)
    captured = b""
    reason = "timeout"
    try:
        with (out / "qemu.log").open("wb") as log:
            while time.monotonic() - start < args.timeout:
                if selector.select(timeout=1):
                    chunk = proc.stdout.read1(65536)
                    if not chunk:
                        reason = "process_exited"
                        break
                    log.write(chunk)
                    log.flush()
                    captured += chunk
                    if b"MODEL_DESTROYED" in captured:
                        reason = "completed"
                        break
                    if any(x in captured for x in [b"Guru Meditation", b"abort() was called", b"Assertion failed"]):
                        reason = "firmware_failure"
                        break
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    text = captured.decode(errors="replace")
    passed = reason == "completed" and "GOLDEN_TEST_PASSED" in text and "psram_bytes=8388608" in text
    peak = re.search(r"validation_internal_peak_delta=(\d+) validation_psram_peak_delta=(\d+)", text)
    golden = re.search(r"golden_count=(\d+) class_agreement=(\d+) integer_mismatches=(\d+) max_integer_error=(\d+)", text)
    arena = re.search(r"tensor_arena_reserved_bytes=(\d+) tensor_arena_used_bytes=(\d+)", text)
    report = {"functional_pass": passed, "reason": reason, "command": cmd,
        "flash_image_sha256": hashlib.sha256((out / "flash_image.bin").read_bytes()).hexdigest(),
        "scope": "standalone_inference_harness_not_full_board_application",
        "host_wall_seconds_not_device_latency": time.monotonic()-start,
        "real_hardware_validated": False, "latency_label_ms": None,
        "validation_internal_peak_delta": int(peak[1]) if peak else None,
        "validation_psram_peak_delta": int(peak[2]) if peak else None}
    if golden:
        report.update(dict(zip(["golden_count", "class_agreement", "integer_mismatches", "max_integer_error"], map(int, golden.groups()))))
    if arena:
        report.update(tensor_arena_reserved_bytes=int(arena[1]), tensor_arena_used_bytes=int(arena[2]))
    (out / "qemu-report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
