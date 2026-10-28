"""Finish the frozen NAS, fully train/quantize/verify candidates, lock 15 networks.

All stage receipts and failed attempts are retained. No serial/board operations,
test-set tuning, simulated latency labels, or threshold relaxation exist here.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import shutil
import signal
import subprocess
import sys
import time
from execution import environment, preserve_cpu_affinity
preserve_cpu_affinity()

ROOT = Path(__file__).resolve().parent
PY = ROOT / ".venv/bin/python"
TF = ROOT / ".tflite-venv/bin/python"


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024**2), b""):
            h.update(block)
    return h.hexdigest()


def archive_incomplete(path):
    if path.exists():
        path.rename(path.with_name(path.name + ".incomplete-" + str(time.time_ns())))


def command(cmd, logfile, env, cpus, timeout):
    logfile.parent.mkdir(parents=True, exist_ok=True)
    with logfile.open("a") as log:
        log.write(json.dumps({"command": list(map(str, cmd)), "time_unix": time.time()}) + "\n")
        log.flush()
        proc = subprocess.Popen([str(PY), "supervise.py", "--cpus", ",".join(map(str, cpus)),
            "--descendants", "--", *map(str, cmd)],
            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            raise RuntimeError(f"Stage timeout; preserved log {logfile}")
        if code:
            raise RuntimeError(f"Stage exited {code}; preserved log {logfile}")


def candidate(args, record, gpu, cpus):
    cfg = read(ROOT / "config.json")
    campaign = ROOT / "runs" / args.run
    ident = record["id"]
    statefile = campaign / "states" / (ident + ".json")
    runname = args.run + "/models/" + ident
    run = ROOT / "runs" / runname
    locks = campaign / "locks"
    locks.mkdir(exist_ok=True)
    logs = campaign / "logs" / ident
    env = dict(environment(gpu), FLOWERS_PERSISTENT_WORKERS="1")
    cpuenv = dict(env, CUDA_VISIBLE_DEVICES="-1", TF_USE_LEGACY_KERAS="1",
        PATH=str(ROOT / ".tflite-venv/bin") + ":" + env.get("PATH", ""))
    state = {"id": ident, "run": runname, "arch": record["arch"], "gpu": gpu,
        "started_unix": time.time(), "test_used": False, "real_hardware_validated": False}

    def stage(name):
        state.update(stage=name, updated_unix=time.time())
        write(statefile, state)
        print(json.dumps({"candidate": ident, "stage": name}), flush=True)

    with (locks / (ident + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if statefile.exists() and read(statefile).get("stage") in {"accepted", "rejected_quality"}:
            return read(statefile)
        try:
            if not (run / "summary.json").exists():
                archive_incomplete(run)
                stage("full_training")
                archfile = ROOT / "runs" / args.search / ident / "arch.json"
                command([PY, "pipeline.py", "train", "--model", "ofa", "--arch", archfile,
                    "--run", runname], logs / "full-training.log", env, cpus, 3600)
            summary = read(run / "summary.json")
            protocol = read(run / "protocol.json")
            assert protocol["head_epochs"] == cfg["train"]["head_epochs"]
            assert protocol["finetune_epochs"] == cfg["train"]["finetune_epochs"]
            assert protocol["max_batches"] is None and summary["test_used"] is False
            state["fp32_top1"] = summary["validation"]["top1"]
            if state["fp32_top1"] < cfg["quality"]["validation_top1_target"]:
                state["reason"] = "full FP32 training did not meet accuracy floor"
                stage("rejected_quality")
                return state
            if not (run / "export-native-pool/report.json").exists():
                archive_incomplete(run / "export-native-pool")
                stage("onnx_export")
                command([PY, "pipeline.py", "export", "--run", runname,
                    "--native-activations", "--native-pooling"], logs / "export.log", cpuenv, cpus, 900)
            shared = ROOT / "runs/pilot-ofa-min-v1/tflite-data"
            if not (run / "tflite-data").exists():
                (run / "tflite-data").symlink_to(shared, target_is_directory=True)
            assert (run / "tflite-data").resolve() == shared.resolve()
            write(run / "shared-data.json", {"path": str(shared), "protocol_sha256": digest(shared / "protocol.json"),
                "reason": "same fixed image size, ImageNet preprocessing, training calibration IDs and validation split", "test_used": False})
            source = "tflite-ptq"
            if not (run / source / "report.json").exists():
                archive_incomplete(run / source)
                stage("tflite_conversion")
                command([TF, "convert_tflite.py", "--run", runname, "--tag", source,
                    "--mcu-activations"], logs / "conversion.log", cpuenv, cpus, 1800)
            accepted_qat = None
            for round_index, lr in [(1, 5e-5), (2, 2e-5)]:
                qat = "qat-tflite-round%d" % round_index
                if not (run / qat / "summary.json").exists():
                    archive_incomplete(run / qat)
                    stage("qat_round_%d" % round_index)
                    cmd = [PY, "qat_tflite.py", "--run", runname, "--source", source,
                        "--name", qat, "--epochs", "50", "--lr", str(lr), "--final-topk", "3"]
                    if round_index == 2:
                        cmd += ["--resume", str(run / "qat-tflite-round1/best.pt")]
                    command(cmd, logs / (qat + ".log"), env, cpus, 5400)
                qreport = read(run / qat / "summary.json")
                if qreport["int8_top1"] >= cfg["quality"]["validation_top1_target"] and state["fp32_top1"] - qreport["int8_top1"] <= cfg["quality"]["max_quantization_drop"]:
                    accepted_qat = qat
                    break
            if accepted_qat is None:
                state.update(reason="neither full QAT attempt met unchanged integer accuracy/drop gates", qat=qreport)
                stage("rejected_quality")
                return state
            state.update(qat_directory=accepted_qat, int8_top1=qreport["int8_top1"],
                quantization_drop=state["fp32_top1"] - qreport["int8_top1"])
            stage("firmware_and_qemu")
            command([PY, "validate_candidate.py", "--run", runname, "--qat", accepted_qat],
                logs / "firmware-qemu.log", cpuenv, cpus, 7200)
            selection = read(run / "deployment-selection.json")
            assert selection["passed"] and all(selection["checks"].values())
            fw = run / selection["firmware_directory"]
            qemu = read(fw / "qemu-report.json")
            manifest = read(fw / "manifest.json")
            quant = run / manifest["quantization_directory"]
            report = read(quant / "report.json")
            assert digest(quant / "model.tflite") == selection["production_sha256"] == report["production_sha256"]
            state.update(production_sha256=selection["production_sha256"],
                model_file=str(quant / "model.tflite"), quant_directory=manifest["quantization_directory"],
                firmware_directory=selection["firmware_directory"], production_model_bytes=report["production_model_bytes"],
                psram_peak_bytes=qemu["validation_psram_peak_delta"], internal_heap_peak_bytes=qemu["validation_internal_peak_delta"],
                tensor_arena_bytes=qemu["tensor_arena_used_bytes"], golden_count=qemu["golden_count"],
                integer_mismatches=qemu["integer_mismatches"], macs=record["proxy"]["macs"],
                depthwise_mac_fraction=record["proxy"]["depthwise_mac_fraction"],
                latency_label_ms=None, finished_unix=time.time())
            stage("accepted")
        except Exception as exc:
            state.update(error=str(exc), finished_unix=time.time())
            stage("failed_execution")
        return state


def select_diverse(states, count=15):
    import numpy as np
    states = sorted(states, key=lambda s: (-s["int8_top1"], s["id"]))
    assert len(states) >= count
    features = np.array([[math.log(s["macs"]), math.log(s["production_model_bytes"]),
        math.log(s["psram_peak_bytes"]), s["depthwise_mac_fraction"], sum(s["arch"]["d"])] for s in states])
    span = np.maximum(features.max(0) - features.min(0), 1e-12)
    features = (features - features.min(0)) / span
    chosen = list(range(min(5, count)))
    while len(chosen) < count:
        remaining = [i for i in range(len(states)) if i not in chosen]
        chosen.append(max(remaining, key=lambda i: (float(((features[chosen] - features[i])**2).sum(1).min()),
            states[i]["int8_top1"], -i)))
    return [states[i] for i in chosen]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--search", default="nas-tflite-v1")
    ap.add_argument("--run", default="nas-final-v1")
    ap.add_argument("--gpus", type=int, nargs="+", default=[1, 2, 7])
    ap.add_argument("--candidate", help="Fully evaluate one already searched architecture; does not lock a shortlist")
    ap.add_argument("--cpus", default="84,85,86,87")
    a = ap.parse_args()
    out = ROOT / "runs" / a.run
    out.mkdir(exist_ok=True)
    if a.candidate:
        records = read(ROOT / "runs" / a.search / "evaluated.json")
        record = next(r for r in records if r["id"] == a.candidate)
        assert record["proxy_feasible"]
        result = candidate(a, record, a.gpus[0], list(map(int, a.cpus.split(','))))
        print(json.dumps(result), flush=True)
        return
    with (out / "campaign.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cfg = read(ROOT / "config.json")
        plan = {"search": a.search, "config": cfg, "initial_full_training_candidates": 30,
            "full_training": "5 head + 50 whole-network epochs from pretrained OFA weights",
            "qat": {"epochs": 50, "lr": 5e-5, "selection": "top 3 fake-quant validation checkpoints plus last, each evaluated as exported INT8",
                "retry": "50 extra epochs at 2e-5 only for candidates failing the unchanged accuracy/drop requirements"},
            "selection": "5 highest INT8 validation accuracy, then 10 max-min resource-diverse accepted networks",
            "fallback": "if fewer than 15 pass, fully evaluate further proxy-feasible NAS results in batches of 10",
            "quality_unchanged": True, "physical_board_flashed": False, "test_used": False}
        planfile = out / "plan.json"
        if planfile.exists():
            assert read(planfile) == plan
        else:
            write(planfile, plan)
        write(out / "execution.json", {"code_sha256": digest(__file__), "gpu_slots": a.gpus,
            "started_unix": time.time(), "actual_data_workers": 0,
            "execution_reason": "shared server blocks in pipe creation after CUDA; use in-process image loading"})
        search = ROOT / "runs" / a.search
        while not (search / "stage-status.json").exists():
            evaluations = read(search / "evaluated.json") if (search / "evaluated.json").exists() else []
            write(out / "status.json", {"stage": "waiting_for_formal_search", "evaluated": len(evaluations),
                "search_budget": cfg["search"]["unique_budget"], "updated_unix": time.time(), "qualified_count": 0})
            time.sleep(30)
        assert read(search / "stage-status.json")["evaluations"] == cfg["search"]["unique_budget"]
        records = read(search / "full-training-shortlist.json")
        assert len(records) == cfg["search"]["full_training_shortlist"]
        reserve = sorted([r for r in read(search / "evaluated.json") if r["proxy_feasible"] and r["id"] not in {x["id"] for x in records}],
            key=lambda r: (-r["validation_top1"], r["id"]))
        slots = queue.Queue()
        for index, gpu in enumerate(a.gpus):
            slots.put((gpu, list(range(40 + 4 * index, 44 + 4 * index))))
        results = []
        def work(record):
            gpu, cpus = slots.get()
            try:
                return candidate(a, record, gpu, cpus)
            finally:
                slots.put((gpu, cpus))
        with ThreadPoolExecutor(max_workers=len(a.gpus)) as pool:
            while records:
                write(out / "current-batch.json", records)
                futures = [pool.submit(work, record) for record in records]
                for future in as_completed(futures):
                    results.append(future.result())
                    write(out / "results.json", sorted(results, key=lambda s: s["id"]))
                    write(out / "status.json", {"stage": "candidate_training_and_validation",
                        "completed_candidates": len(results), "qualified_count": sum(s["stage"] == "accepted" for s in results),
                        "updated_unix": time.time(), "physical_board_flashed": False})
                if sum(s["stage"] == "accepted" for s in results) >= 15:
                    break
                records, reserve = reserve[:10], reserve[10:]
        passed = [s for s in results if s["stage"] == "accepted"]
        if len(passed) < 15:
            write(out / "status.json", {"stage": "insufficient_qualified_candidates", "qualified_count": len(passed),
                "quality_unchanged": True, "requires_followup": True, "updated_unix": time.time()})
            raise SystemExit("Fewer than 15 pass; thresholds were not relaxed")
        selected = select_diverse(passed)
        # Re-evaluate every selected model's stored hashes and all batch evidence.
        for state in selected:
            command([PY, "verification.py", "--run", state["run"]],
                out / "logs" / state["id"] / "final-audit.log", environment(-1), [60, 61, 62, 63], 120)
        target = out / "final15"
        target.mkdir(exist_ok=True)
        for index, state in enumerate(selected, 1):
            modeldir = target / ("%02d-%s" % (index, state["id"]))
            modeldir.mkdir(exist_ok=True)
            run = ROOT / "runs" / state["run"]
            quant = run / state["quant_directory"]
            shutil.copy2(quant / "model.tflite", modeldir / "model.tflite")
            for filename in ["report.json", "model.audit.json", "golden-protocol.json", "training-provenance.json"]:
                shutil.copy2(quant / filename, modeldir / filename)
            shutil.copy2(run / "deployment-selection.json", modeldir / "deployment-selection.json")
            shutil.copy2(run / state["firmware_directory"] / "qemu-report.json", modeldir / "qemu-report.json")
            shutil.copy2(run / "summary.json", modeldir / "fp32-summary.json")
            shutil.copy2(run / "protocol.json", modeldir / "full-training-protocol.json")
            shutil.copy2(run / state["qat_directory"] / "protocol.json", modeldir / "qat-protocol.json")
            fw = run / state["firmware_directory"]
            shutil.copy2(fw / "manifest.json", modeldir / "firmware-manifest.json")
            for batch in read(fw / "manifest.json").get("batches", []):
                receipts = modeldir / "firmware-receipts" / batch["directory"]
                receipts.mkdir(parents=True, exist_ok=True)
                for filename in ["manifest.json", "qemu-report.json"]:
                    shutil.copy2(fw / batch["directory"] / filename, receipts / filename)
            write(modeldir / "architecture.json", state["arch"])
            write(modeldir / "provenance.json", state)
        manifest = {"status": "locked", "count": 15, "networks": selected,
            "selection_policy": plan["selection"], "config": cfg, "locked_unix": time.time(),
            "test_used_for_selection": False, "real_hardware_validated": False, "physical_board_flashed": False,
            "latency_labels_collected": 0}
        write(out / "selected-networks.json", manifest)
        fields = ["id", "fp32_top1", "int8_top1", "quantization_drop", "macs", "production_model_bytes",
            "psram_peak_bytes", "internal_heap_peak_bytes", "tensor_arena_bytes", "production_sha256"]
        with (out / "selected-networks.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows({k: s[k] for k in fields} for s in selected)
        write(out / "status.json", {"stage": "complete_15_qualified_networks", "qualified_count": len(passed),
            "selected_count": 15, "manifest_sha256": digest(out / "selected-networks.json"),
            "updated_unix": time.time(), "physical_board_flashed": False})
        print(json.dumps({"completed": True, "selected": 15, "manifest": str(out / "selected-networks.json")}), flush=True)


if __name__ == "__main__":
    main()
