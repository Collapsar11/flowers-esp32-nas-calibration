"""Fixed-scale QAT of the exact verified TFLite operator graph on the GPU.

Training uses STE for TFLite's per-channel INT8 weights, INT8 activations and
INT32 biases. Integer kernels may round differently: acceptance/checkpoint
selection therefore uses the exported TFLite interpreter, never fake-QAT alone.
"""
import argparse
import faulthandler
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
from execution import preserve_cpu_affinity
preserve_cpu_affinity()
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import pipeline as p

faulthandler.register(signal.SIGUSR1)
VALIDATION_LOADER = None


class TFLiteGraph(nn.Module):
    def __init__(self, graph_dir):
        super().__init__()
        self.meta = json.loads((graph_dir / "graph.json").read_text())
        self.ops = self.meta["ops"]
        arrays = np.load(graph_dir / "float-constants.npz")
        self.values = nn.ParameterDict()
        self.constants = {}
        self.parameter_ops = {}
        for op in self.ops:
            if op["name"] in {"CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED"}:
                for pos, tid in enumerate(op["inputs"][1:], 1):
                    if tid < 0:
                        continue
                    key = str(tid)
                    val = arrays[key].copy()
                    if pos == 1:
                        if op["name"] == "CONV_2D":
                            val = val.transpose(0, 3, 1, 2)
                        elif op["name"] == "DEPTHWISE_CONV_2D":
                            val = val.transpose(3, 0, 1, 2)
                    self.values[key] = nn.Parameter(torch.from_numpy(val.copy()))
                    self.parameter_ops[key] = (op["name"], pos)
                    scales = np.array(self.meta["tensors"][key]["scales"], dtype=np.float32)
                    self.register_buffer("scale_" + key, torch.from_numpy(scales))
                    self.register_buffer("zero_" + key, torch.zeros(len(scales), dtype=torch.int32))
        for key in arrays.files:
            if key not in self.values:
                self.constants[int(key)] = arrays[key].copy()

    def activation(self, x, tid, quant):
        if not quant:
            return x
        d = self.meta["tensors"][str(tid)]
        assert len(d["scales"]) == len(d["zeros"]) == 1
        return torch.fake_quantize_per_tensor_affine(x, d["scales"][0], d["zeros"][0], -128, 127)

    def parameter(self, tid, quant):
        key = str(tid)
        x = self.values[key]
        if not quant:
            return x
        scale = getattr(self, "scale_" + key)
        if self.parameter_ops[key][1] == 2:
            # Bias uses the fixed input_scale * per-channel weight_scale ABI.
            q = (x / scale).round().clamp(-2147483648, 2147483647) * scale
            return x + (q - x).detach()
        if len(scale) > 1:
            return torch.fake_quantize_per_channel_affine(x, scale,
                getattr(self, "zero_" + key), 0, -127, 127)
        return torch.fake_quantize_per_tensor_affine(x, float(scale[0]), 0, -127, 127)

    def forward(self, x, quant=True):
        tensors = {self.meta["input"]: self.activation(x, self.meta["input"], quant)}
        for op in self.ops:
            ins, name, opt = op["inputs"], op["name"], op["options"]
            x = tensors[ins[0]]
            if name in {"CONV_2D", "DEPTHWISE_CONV_2D"}:
                w = self.parameter(ins[1], quant)
                b = self.parameter(ins[2], quant) if len(ins) > 2 and ins[2] >= 0 else None
                sh, sw = opt["StrideH"], opt["StrideW"]
                dh, dw = opt["DilationHFactor"], opt["DilationWFactor"]
                if opt["Padding"] == 0:
                    h, width = x.shape[2:]
                    ph = max((math.ceil(h / sh) - 1) * sh + (w.shape[2] - 1) * dh + 1 - h, 0)
                    pw = max((math.ceil(width / sw) - 1) * sw + (w.shape[3] - 1) * dw + 1 - width, 0)
                    x = F.pad(x, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2))
                else:
                    assert opt["Padding"] == 1
                groups = x.shape[1] if name == "DEPTHWISE_CONV_2D" else 1
                y = F.conv2d(x, w, b, (sh, sw), dilation=(dh, dw), groups=groups)
            elif name == "FULLY_CONNECTED":
                assert x.ndim == 2 or tuple(x.shape[2:]) == (1, 1)
                y = F.linear(x.flatten(1), self.parameter(ins[1], quant), self.parameter(ins[2], quant))
            elif name == "HARD_SWISH":
                y = F.hardswish(x)
            elif name in {"ADD", "MUL"}:
                if ins[1] in tensors:
                    other = tensors[ins[1]]
                else:
                    other = torch.as_tensor(self.constants[ins[1]], device=x.device)
                    assert other.numel() == 1, "only scalar arithmetic constants are supported"
                    if quant:
                        d = self.meta["tensors"][str(ins[1])]
                        other = torch.fake_quantize_per_tensor_affine(other, d["scales"][0], d["zeros"][0], -128, 127)
                y = x + other if name == "ADD" else x * other
            elif name == "MEAN":
                assert self.constants[ins[1]].tolist() == [1, 2]
                y = x.mean((2, 3), keepdim=bool(opt["KeepDims"]))
            elif name == "PAD":
                pads = self.constants[ins[1]]
                assert not pads[[0, 3]].any()
                y = F.pad(x, (int(pads[2, 0]), int(pads[2, 1]), int(pads[1, 0]), int(pads[1, 1])))
            else:
                raise ValueError(name)
            activation = opt.get("FusedActivationFunction", 0)
            if activation == 1:
                y = F.relu(y)
            elif activation == 3:
                y = F.relu6(y)
            else:
                assert activation == 0, activation
            tensors[op["outputs"][0]] = self.activation(y, op["outputs"][0], quant)
        return tensors[self.meta["output"]]

    def export_weights(self, path):
        arrays = {}
        for key, param in self.values.items():
            value = param.detach().cpu().numpy()
            op, position = self.parameter_ops[key]
            if position == 1 and op == "CONV_2D":
                value = value.transpose(0, 2, 3, 1)
            elif position == 1 and op == "DEPTHWISE_CONV_2D":
                value = value.transpose(1, 2, 3, 0)
            arrays[key] = value
        np.savez(path, **arrays)


@torch.no_grad()
def validate(model, quant=True):
    correct = count = 0
    for x, y in (VALIDATION_LOADER if VALIDATION_LOADER is not None else p.loader("val")):
        logits = model(x.cuda(), quant=quant)
        correct += logits.argmax(1).cpu().eq(y).sum().item()
        count += len(y)
    assert count == 1020
    return correct / count


@torch.no_grad()
def validate_fp32_pair(model, teacher):
    """Compare in the same FP32 arithmetic; keep historical score as quality reference."""
    correct = teacher_correct = disagreements = count = 0
    maximum = 0.0
    for x, y in (VALIDATION_LOADER if VALIDATION_LOADER is not None else p.loader("val")):
        x = x.cuda()
        actual, expected = model(x, quant=False), teacher(x)
        maximum = max(maximum, (actual - expected).abs().max().item())
        a, b = actual.argmax(1).cpu(), expected.argmax(1).cpu()
        correct += a.eq(y).sum().item()
        teacher_correct += b.eq(y).sum().item()
        disagreements += a.ne(b).sum().item()
        count += len(y)
    assert count == 1020 and maximum < 0.002 and disagreements <= 1
    return {"reconstructed_top1": correct / count, "teacher_same_precision_top1": teacher_correct / count,
        "max_abs_error": maximum, "prediction_disagreements": disagreements, "count": count}


def bridge(*args):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", TF_CPP_MIN_LOG_LEVEL="2")
    subprocess.run([str(p.ROOT / ".tflite-venv/bin/python"), str(p.ROOT / 'src/tflite_qat_bridge.py'),
        *map(str, args)], check=True, env=env)


def main():
    global VALIDATION_LOADER
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--source", default="tflite-int8-v3")
    ap.add_argument("--name", default="qat-tflite-v1")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--export-every", type=int, default=5)
    ap.add_argument("--final-topk", type=int, default=0,
        help="After all epochs, integer-evaluate the top K fake-QAT checkpoints plus the last; 0 keeps periodic exports")
    ap.add_argument("--resume", type=Path)
    a = ap.parse_args()
    p.seed_all(p.config()["seed"])
    VALIDATION_LOADER = p.loader("val")
    training_loader = p.loader("train", True)
    # Folded deployment convolutions must be compared in FP32, not Ampere TF32.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    print("QAT stage: initialized PyTorch and GPU", flush=True)
    run = p.ROOT / "runs" / a.run
    out = run / a.name
    out.mkdir(exist_ok=False)
    bridge("extract", "--source", run / a.source, "--out", out / "graph")
    model = TFLiteGraph(out / "graph").cuda()
    print("QAT stage: reconstructed deployment graph", flush=True)
    state = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    teacher = p.build_model(state["model"], state["arch"], pretrained=False).cuda().eval()
    teacher.load_state_dict(state["state_dict"])
    print("QAT stage: loaded original FP32 teacher", flush=True)
    for param in teacher.parameters():
        param.requires_grad_(False)
    x, _ = next(iter(p.loader("val", batch=16)))
    with torch.no_grad():
        parity = (model(x.cuda(), quant=False) - teacher(x.cuda())).abs().max().item()
    assert parity < 0.002, ("FP32 graph reconstruction failed", parity)
    print(json.dumps({"fp32_parity_max_error": parity}), flush=True)
    pair = validate_fp32_pair(model, teacher)
    fp32_accuracy = pair["reconstructed_top1"]
    reference = json.loads((run / "summary.json").read_text())["validation"]["top1"]
    if a.resume:
        model.load_state_dict(torch.load(a.resume, map_location="cuda", weights_only=True))
    protocol = {"method": "fixed-scale TFLite graph QAT with STE", "source": a.source,
        "code_sha256": p.digest(__file__), "bridge_sha256": p.digest(p.ROOT / 'src/tflite_qat_bridge.py'),
        "pipeline_sha256": p.digest(p.ROOT / 'src/pipeline.py'), "torch": torch.__version__,
        "source_sha256": model.meta["source_sha256"], "fp32_parity_max_error": parity,
        "reconstructed_fp32_top1": fp32_accuracy, "original_fp32_top1": reference,
        "full_fp32_pair_check": pair,
        "train_count": 1020, "validation_count": 1020, "epochs": a.epochs, "lr": a.lr,
        "loss": "0.5 CE(label_smoothing=0.05) + 0.5 KL(T=2) to original frozen FP32 teacher",
        "augmentation": "pipeline.py full original training augmentation", "scales": "fixed from train-only calibration",
        "model_selection": "exported integer TFLite full validation accuracy",
        "final_topk": a.final_topk, "export_every": a.export_every,
        "actual_data_workers": int(os.environ.get("FLOWERS_DATA_WORKERS", p.config()["train"]["workers"])),
        "persistent_data_workers": os.environ.get("FLOWERS_PERSISTENT_WORKERS") == "1" and os.environ.get("FLOWERS_DATA_WORKERS") != "0",
        "resume": str(a.resume) if a.resume else None, "test_used": False,
        "quality": p.config()["quality"], "search_launch_performed_by_this_stage": False, "real_hardware_validated": False}
    p.save_json(out / "protocol.json", protocol)
    optimizer = torch.optim.Adam(model.parameters(), lr=a.lr)
    best = -1.0
    start = time.time()

    def export(epoch):
        nonlocal best
        folder = out / ("epoch-%03d" % epoch)
        folder.mkdir()
        model.export_weights(folder / "weights.npz")
        bridge("patch", "--graph", out / "graph", "--weights", folder / "weights.npz", "--dest", folder / "model.tflite")
        bridge("evaluate", "--model", folder / "model.tflite", "--data", run / "tflite-data",
            "--dest", folder / "report.json", "--run", run)
        report = json.loads((folder / "report.json").read_text())
        if report["int8_top1"] > best:
            best = report["int8_top1"]
            torch.save(model.state_dict(), out / "best.pt")
            target = out / "best-export"
            target.mkdir(exist_ok=True)
            for filename in ["model.tflite", "model.audit.json", "report.json"]:
                shutil.copy2(folder / filename, target / filename)
            p.save_json(out / "best.json", {"epoch": epoch, **report})
        return report

    initial_fake = validate(model)
    initial = export(0) if not a.final_topk else None
    print(json.dumps({"epoch": 0, "fake_quant_top1": initial_fake, "integer_top1": initial["int8_top1"] if initial else None,
        "fp32_parity_max_error": parity}), flush=True)
    retained = []
    for epoch in range(1, a.epochs + 1):
        lr = a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (epoch - 1) / a.epochs)))
        for group in optimizer.param_groups:
            group["lr"] = lr
        total, count = 0.0, 0
        for x, y in training_loader:
            x, y = x.cuda(), y.cuda()
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            with torch.no_grad():
                teacher_logits = teacher(x)
            ce = F.cross_entropy(logits, y, label_smoothing=0.05)
            kd = F.kl_div(F.log_softmax(logits / 2, 1), F.softmax(teacher_logits / 2, 1), reduction="batchmean") * 4
            loss = 0.5 * (ce + kd)
            assert torch.isfinite(loss)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(y)
            count += len(y)
        fake = validate(model)
        record = {"epoch": epoch, "loss": total / count, "lr": lr,
            "fake_quant_top1": fake, "elapsed_s": time.time() - start}
        if a.final_topk:
            candidate = (fake, -epoch)
            ranked = sorted(retained + [candidate], reverse=True)[:a.final_topk]
            if candidate in ranked:
                torch.save(model.state_dict(), out / ("retained-%03d.pt" % epoch))
            for score, negative_epoch in retained:
                if (score, negative_epoch) not in ranked:
                    (out / ("retained-%03d.pt" % -negative_epoch)).unlink()
            retained = ranked
        elif epoch % a.export_every == 0 or epoch == a.epochs:
            report = export(epoch)
            record["integer_top1"] = report["int8_top1"]
        torch.save(model.state_dict(), out / "last.pt")
        with (out / "history.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
    if a.final_topk:
        epochs = sorted({-negative_epoch for _, negative_epoch in retained} | {a.epochs})
        p.save_json(out / "integer-evaluation-plan.json", {"epochs": epochs,
            "selection": "top K fake-QAT validation plus final epoch; choose by full integer validation", "test_used": False})
        for epoch in epochs:
            checkpoint = out / ("retained-%03d.pt" % epoch)
            if not checkpoint.exists():
                assert epoch == a.epochs
                checkpoint = out / "last.pt"
            model.load_state_dict(torch.load(checkpoint, map_location="cuda", weights_only=True))
            report = export(epoch)
            print(json.dumps({"integer_evaluated_epoch": epoch, "integer_top1": report["int8_top1"]}), flush=True)
    result = json.loads((out / "best.json").read_text())
    quality = p.config()["quality"]
    p.save_json(out / "summary.json", {**result, "best_validation_top1": best,
        "accuracy_gate_passed": best >= quality["validation_top1_target"] and reference - best <= quality["max_quantization_drop"],
        "epochs": a.epochs, "firmware_verified": False, "search_launch_performed_by_this_stage": False})


if __name__ == "__main__":
    main()
