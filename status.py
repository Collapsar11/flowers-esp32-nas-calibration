"""Read-only experiment status (never evaluates the Flowers test set)."""
import json
from pathlib import Path
from datetime import datetime, timezone

root = Path(__file__).resolve().parent
records = []
for run in sorted((root / "runs").glob("pilot-*")):
    item = {"run": run.name}
    summary = run / "summary.json"
    if summary.exists():
        data = json.loads(summary.read_text())
        item.update(status="training_complete", validation_top1=data["validation"]["top1"],
                    test_used=data["test_used"], checkpoint_sha256=data["checkpoint_sha256"])
    elif (run / "history.jsonl").exists():
        lines = (run / "history.jsonl").read_text().splitlines()
        data = json.loads(lines[-1])
        item.update(status="training_in_progress", phase=data["phase"], epoch=data["epoch"],
                    latest_validation_top1=data["validation"]["top1"])
    else:
        item["status"] = "initializing"
    item["quantization"] = []
    for quant in sorted(list(run.glob("int8-*")) + list(run.glob("tflite-int8-*"))):
        report = quant / "report.json"
        audit = quant / "integer_constraint_audit.json"
        entry = {"directory": quant.name}
        if report.exists():
            data = json.loads(report.read_text())
            entry.update({k: data.get(k) for k in ["quantization_bits", "int8_top1", "quantized_top1",
                "quantization_drop", "production_model_bytes", "quality_target_met", "quantization_drop_gate_met"]})
            entry["status"] = "host_validation_complete"
            # Old runs without a strict integer audit are diagnostic, never accepted.
            entry["integer_constraints_verified"] = audit.exists() and json.loads(audit.read_text())["passed"]
        elif audit.exists():
            entry["integer_constraints_verified"] = json.loads(audit.read_text())["passed"]
            entry["status"] = "incomplete"
        else:
            entry["status"] = "incomplete"
        item["quantization"].append(entry)
    report = run / "firmware/qemu-report.json"
    item["qemu"] = json.loads(report.read_text()) if report.exists() else None
    item["firmware_variants"] = {f.parent.name: json.loads(f.read_text())
        for f in sorted(run.glob("firmware*/qemu-report.json"))}
    item["qat"] = {f.parent.name: json.loads(f.read_text())
        for f in sorted(run.glob("qat*/summary.json"))}
    item["qat_progress"] = {}
    for directory in sorted(run.glob("qat-tflite-*")):
        progress = {"completed": (directory / "summary.json").exists()}
        for filename, key in [("protocol.json", "protocol"), ("best.json", "best_integer_export")]:
            if (directory / filename).exists():
                progress[key] = json.loads((directory / filename).read_text())
        history = directory / "history.jsonl"
        if history.exists() and history.read_text().strip():
            progress["latest_epoch"] = json.loads(history.read_text().strip().splitlines()[-1])
        item["qat_progress"][directory.name] = progress
    selection = run / "deployment-selection.json"
    item["selected_deployment_snapshot"] = json.loads(selection.read_text()) if selection.exists() else None
    records.append(item)
searches = []
for plan_file in sorted((root / "runs").glob("*/plan.json")):
    plan = json.loads(plan_file.read_text())
    if plan.get("scope") != "formal_search":
        continue
    directory = plan_file.parent
    evaluated_file = directory / "evaluated.json"
    evaluated = json.loads(evaluated_file.read_text()) if evaluated_file.exists() else []
    stage_file = directory / "stage-status.json"
    launched = (directory / "deployment-precondition.json").exists() or bool(evaluated)
    searches.append({"run": directory.name, "method": plan["method"], "evaluated_count": len(evaluated),
        "launch_gate_accepted": launched,
        "unique_budget": plan["config"]["search"]["unique_budget"],
        "latest_evaluation": evaluated[-1] if evaluated else None,
        "stage": json.loads(stage_file.read_text()) if stage_file.exists() else ("not_yet_complete" if launched else "not_launched_gate_not_passed"),
        "formal_candidates_locked": False, "test_used": False})
campaigns = []
for directory in sorted((root / "runs").glob("nas-final-*")):
    status_file = directory / "status.json"
    item = {"run": directory.name,
        "status": json.loads(status_file.read_text()) if status_file.exists() else {"stage": "initializing"}}
    item["candidates"] = [json.loads(f.read_text()) for f in sorted((directory / "states").glob("*.json"))]
    manifest = directory / "selected-networks.json"
    item["selected_manifest"] = str(manifest) if manifest.exists() else None
    campaigns.append(item)
complete = any(c["status"].get("stage") == "complete_15_qualified_networks" for c in campaigns)
print(json.dumps({"timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "root": str(root), "pilots": records, "searches": searches, "physical_board_flashed": False,
    "campaigns": campaigns, "server_stage_complete": complete}, indent=2))
