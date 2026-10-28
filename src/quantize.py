"""ESP-PPQ INT8 conversion; host simulation is never a hardware timing label."""
import argparse
import json
from pathlib import Path
import shutil
import sys

import pipeline as p
sys.path.insert(0, str(p.ROOT / "upstream" / "esp-ppq"))
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


def load_quant_graph(path):
    graph = torch.load(path, map_location="cpu", weights_only=False)
    for op in graph.operations.values():
        op.inputs[:] = [graph.variables[name] for name in op.inputs]
        op.outputs[:] = [graph.variables[name] for name in op.outputs]
    for var in graph.variables.values():
        var.dest_ops[:] = [graph.operations[name] for name in var.dest_ops]
        if var.source_op is not None:
            var.source_op = graph.operations[var.source_op]
    graph._graph_inputs = {name: graph.variables[name] for name in graph._graph_inputs}
    graph._graph_outputs = {name: graph.variables[name] for name in graph._graph_outputs}
    return graph


def legalize_integer_scales(graph):
    """Coarsen only incompatible activation scales; preserve floating weights.

    ESP-DL requires output exponent >= input exponent + weight exponent.
    Recompute passive bias scales with the official PPQ pass after changing activations.
    All changes are audited and the resulting accuracy must be re-evaluated.
    """
    from esp_ppq.core import QuantizationStates
    from esp_ppq.quantization.optim import PassiveParameterQuantizePass
    changes = []
    for iteration in range(20):
        changed = False
        for op in graph.topological_sort():
            if op.type not in {"Conv", "Gemm", "MatMul", "Mul"} or not hasattr(op, "input_quant_config"):
                continue
            inp, out = op.input_quant_config, op.output_quant_config
            if len(inp) < 2 or not out or any(c.scale is None for c in [inp[0], inp[1], out[0]]):
                continue
            minimum = inp[0].scale * inp[1].scale
            root = out[0].dominated_by
            if root.scale.numel() != 1 or minimum.numel() != 1:
                raise ValueError("Scale legalization is defined only for per-tensor S3 quantization")
            if root.scale.item() + 1e-20 < minimum.item():
                old = root.scale.item()
                new = torch.maximum(root.scale, torch.pow(2.0, torch.ceil(torch.log2(minimum))))
                state = root.state
                root.state = QuantizationStates.INITIAL
                root.scale = new
                root.state = state
                changes.append({"op": op.name, "old_scale": old, "new_scale": new.item(), "iteration": iteration})
                changed = True
        if not changed:
            PassiveParameterQuantizePass().optimize(graph)
            return changes
    raise RuntimeError("Integer scale constraints failed to converge")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--simulation-only", action="store_true")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--activation-calibration", default="kl", choices=["kl", "minmax", "percentile"])
    parser.add_argument("--tag", default="")
    parser.add_argument("--legalize-scales", action="store_true")
    parser.add_argument("--equalize", action="store_true")
    parser.add_argument("--bits", type=int, default=8, choices=[8, 16])
    parser.add_argument("--source-graph")
    parser.add_argument("--onnx-dir", default="export")
    args = parser.parse_args()
    cfg = p.config()
    if args.target not in {"esp32", "esp32s2", "esp32s3", "esp32c3", "esp32c6", "esp32p4", "c"}:
        raise ValueError("Unsupported or unverified target")
    if not args.simulation_only and cfg["hardware"]["target"] != args.target:
        raise ValueError("Confirm the actual board before target-specific validation")
    from esp_ppq.api import espdl_quantize_onnx
    from esp_ppq.api import QuantizationSettingFactory
    from esp_ppq.api.espdl_interface import generate_test_value, get_target_platform
    from esp_ppq.executor import TorchExecutor
    import esp_ppq.lib as pfl

    p.seed_all(cfg["seed"])
    run = p.ROOT / "runs" / args.run
    out = run / ("int8-" + args.target + ("-" + args.tag if args.tag else ""))
    out.mkdir(exist_ok=False)
    ds = p.dataset("train")
    rng = np.random.default_rng(cfg["seed"])
    labels = np.array(ds._labels)
    indices = sorted(np.concatenate([rng.choice(np.flatnonzero(labels == k), 5, replace=False)
                                     for k in range(102)]).tolist())
    assert len(set(indices)) == 510
    p.save_json(out / "calibration.json", {"split": "train", "per_class": 5, "count": 510,
        "indices": indices, "image_ids": [ds._image_files[i].name for i in indices],
        "seed": cfg["seed"], "hardware_measurement_labels_used": 0})
    data = DataLoader(Subset(ds, indices), batch_size=1, shuffle=False, num_workers=2)
    def collate(batch):
        return batch[0].to(args.device)
    x = ds[indices[0]][0].unsqueeze(0).to(args.device)
    # ESP-PPQ simplifies its input ONNX in-place. Keep the verified source immutable.
    onnx_path = out / "quant-input.onnx"
    shutil.copy2(run / args.onnx_dir / "model.onnx", onnx_path)
    setting = QuantizationSettingFactory.espdl_setting(num_of_bits=args.bits)
    setting.quantize_activation_setting.calib_algorithm = args.activation_calibration
    if args.equalize:
        setting.equalization = True
        setting.equalization_setting.iterations = 4
        setting.equalization_setting.value_threshold = 0.4
        setting.equalization_setting.opt_level = 2
    if args.source_graph:
        graph = load_quant_graph(args.source_graph)
        for op in graph.operations.values():
            if op.type == "Reshape" and op.inputs[1].is_parameter:
                shape = op.inputs[1].value
                if isinstance(shape, torch.Tensor) and shape.ndim == 1 and shape.tolist() == [0, -1]:
                    op.inputs[1].value = torch.tensor([1, -1], dtype=shape.dtype)
        p.save_json(out / "source-graph.json", {"path": str(Path(args.source_graph).resolve()),
                    "sha256": p.digest(args.source_graph), "calibration_repeated": False})
    else:
        graph = espdl_quantize_onnx(onnx_import_file=str(onnx_path),
            espdl_export_file=str(out / "model.espdl"), calib_dataloader=data,
            calib_steps=len(data), input_shape=[1, 3, cfg["image_size"], cfg["image_size"]],
            inputs=[x], target=args.target, num_of_bits=args.bits, collate_fn=collate,
            device=args.device, error_report=False, skip_export=True, setting=setting, debug=args.bits == 16)
    changes = legalize_integer_scales(graph) if args.legalize_scales else []
    p.save_json(out / "scale_adjustments.json", changes)
    violations = []
    for op in graph.operations.values():
        if op.type not in {"Conv", "Gemm", "MatMul", "Mul"} or not hasattr(op, "input_quant_config"):
            continue
        inp, outp = op.input_quant_config, op.output_quant_config
        if len(inp) < 2 or not outp or any(c.scale is None for c in [inp[0], inp[1], outp[0]]):
            continue
        shift = torch.log2(outp[0].scale / (inp[0].scale * inp[1].scale)).min().item()
        if shift < -1e-5:
            violations.append({"op": op.name, "type": op.type, "requantization_shift": shift})
    p.save_json(out / "integer_constraint_audit.json", {"passed": not violations, "violations": violations})
    if violations:
        raise RuntimeError(f"ESP-DL integer constraints fail for {len(violations)} operations; not deployable")
    torch.save(graph, out / "quantized_graph.pt")
    # Evaluate before exporters, and give each exporter its own value-owning graph copy.
    executor = TorchExecutor(graph=graph, device=args.device)
    state = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    net = p.build_model(state["model"], state["arch"], pretrained=False).to(args.device).eval()
    net.load_state_dict(state["state_dict"], strict=True)
    fp_correct = quant_correct = agree = count = 0
    per_n = np.zeros(102, dtype=int)
    per_ok = np.zeros(102, dtype=int)
    with torch.no_grad():
        for image, target in p.loader("val", batch=1):
            image = image.to(args.device)
            fp = int(net(image).argmax(1).item())
            quant = int(executor.forward(image)[0].argmax(1).item())
            y = int(target.item())
            fp_correct += fp == y
            quant_correct += quant == y
            agree += fp == quant
            count += 1
            per_n[y] += 1
            per_ok[y] += quant == y
            if count % 100 == 0:
                print(json.dumps({"validated": count, "fp32_top1": fp_correct/count,
                                  "quantized_top1": quant_correct/count, "bits": args.bits}), flush=True)
    fp_acc, quant_acc = fp_correct/count, quant_correct/count
    p.save_json(out / "host-accuracy.json", {"fp32_top1": fp_acc, "quantized_top1": quant_acc,
        "bits": args.bits, "count": count})
    test_values = generate_test_value(graph, args.device, [x])
    exporter = pfl.Exporter(platform=get_target_platform(args.target, args.bits))
    exporter.export(file_path=str(out / "model.espdl"), graph=graph.copy(copy_value=True),
                    values_for_test=None, export_config=True)
    exporter.export(file_path=str(out / "model-test.espdl"), graph=graph.copy(copy_value=True),
                    values_for_test=test_values, export_config=True)
    p.save_json(out / "report.json", {"status": "host_quantized_validation_complete",
        "target": args.target, "simulation_only": args.simulation_only,
        "quantization_bits": args.bits,
        "activation_calibration": args.activation_calibration,
        "cross_layer_equalization": args.equalize,
        "scale_adjustment_count": len(changes),
        "validation_count": count, "fp32_top1": fp_acc, "int8_top1": quant_acc if args.bits == 8 else None,
        "quantized_top1": quant_acc,
        "mean_class_accuracy": float(np.mean(per_ok/per_n)),
        "quantization_drop": fp_acc-quant_acc, "fp32_int8_agreement": agree/count,
        "quality_target_met": quant_acc >= cfg["quality"]["validation_top1_target"],
        "quantization_drop_gate_met": fp_acc-quant_acc <= cfg["quality"]["max_quantization_drop"],
        "production_model_bytes": (out / "model.espdl").stat().st_size,
        "embedded_test_model_bytes": (out / "model-test.espdl").stat().st_size,
        "production_sha256": p.digest(out / "model.espdl"),
        "test_model_sha256": p.digest(out / "model-test.espdl"),
        "test_used": False, "firmware_verified": False, "real_hardware_validated": False,
        "peak_runtime_memory_bytes": None, "real_latency_ms": None})
    print((out / "report.json").read_text(), flush=True)


if __name__ == "__main__":
    main()
