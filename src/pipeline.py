"""Reproducible pretrained Flowers training and host deployment verification."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import faulthandler
import signal
from execution import preserve_cpu_affinity

preserve_cpu_affinity()

faulthandler.register(signal.SIGUSR1)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "upstream" / "once-for-all"))
sys.path.insert(0, str(ROOT / "upstream" / "mcunet"))
os.environ.setdefault("TORCH_HOME", str(ROOT / "cache" / "torch"))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


def config():
    return json.loads((ROOT / 'configs/config.json').read_text())


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(2**20), b""):
            h.update(b)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def transform(train=False):
    size = config()["image_size"]
    ops = [transforms.RandomResizedCrop(size, scale=tuple(config()["train"]["crop_scale"])),
           transforms.RandomHorizontalFlip(), transforms.ColorJitter(0.15, 0.15, 0.15, 0.03)] if train else [
               transforms.Resize(round(size / 0.875)), transforms.CenterCrop(size)]
    return transforms.Compose(ops + [transforms.ToTensor(), transforms.Normalize(
        [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def dataset(split, augment=False, download=False):
    return datasets.Flowers102(str(ROOT / "data"), split=split,
                               transform=transform(augment), download=download)


def loader(split, augment=False, batch=None):
    cfg = config()["train"]
    workers = int(os.environ.get("FLOWERS_DATA_WORKERS", cfg["workers"]))
    assert workers >= 0
    extra = {}
    if workers and os.environ.get("FLOWERS_SPAWN_WORKERS") == "1":
        # Avoid forking after CUDA/pin-memory threads have initialized. Worker
        # count, per-epoch RNG seeds, augmentation and dataset stay unchanged.
        extra["multiprocessing_context"] = "spawn"
        extra["persistent_workers"] = os.environ.get("FLOWERS_PERSISTENT_WORKERS") == "1"
    return DataLoader(dataset(split, augment), batch_size=batch or cfg["batch_size"],
                      shuffle=augment, num_workers=workers, pin_memory=True, **extra)


def prepare():
    (ROOT / "logs").mkdir(exist_ok=True)
    sets, ids = {}, {}
    for split, expected in config()["split_sizes"].items():
        ds = dataset(split, download=True)
        assert len(ds) == expected, (split, len(ds), expected)
        ids[split] = [p.name for p in ds._image_files]
        sets[split] = {"count": len(ds), "classes": len(set(ds._labels)),
                       "class_counts": np.bincount(ds._labels, minlength=102).tolist()}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        assert not set(ids[a]) & set(ids[b]), "dataset split overlap"
    hashes = {p.name: digest(p) for p in (ROOT / "data" / "flowers-102").glob("*.mat")}
    save_json(ROOT / "logs" / "dataset.json", {"splits": sets, "ids": ids, "metadata_sha256": hashes})
    print(json.dumps({"dataset": sets, "metadata_sha256": hashes}), flush=True)


def arch_for(name, arch_file=None):
    if arch_file:
        arch = json.loads(Path(arch_file).read_text())
        return arch.get("arch", arch)
    presets = {"ofa-min": (2, 3, 3), "ofa-mid": (3, 4, 5), "ofa-max": (4, 6, 7)}
    if name in presets:
        d, e, k = presets[name]
        return {"d": [d] * 5, "e": [e] * 20, "ks": [k] * 20}
    return None


def build_model(name, arch=None, pretrained=True):
    if name == "mbv3-small":
        net = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None)
        net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, 102)
    elif name == "mcunet-in2":
        from mcunet.model_zoo import build_model as mcunet_build
        from mcunet.tinynas.nn.modules.layers import LinearLayer
        net, _, _ = mcunet_build(name, pretrained=pretrained)
        net.classifier = LinearLayer(net.classifier.in_features, 102, dropout_rate=0.2)
    elif name.startswith("ofa"):
        from ofa.model_zoo import ofa_net
        from ofa.utils.layers import LinearLayer
        supernet = ofa_net(config()["ofa_id"], pretrained=pretrained)
        arch = arch or arch_for(name)
        if arch is None:
            raise ValueError("Explicit OFA architecture required")
        supernet.set_active_subnet(**arch)
        net = supernet.get_active_subnet(preserve_weight=pretrained)
        net.classifier = LinearLayer(net.classifier.in_features, 102, dropout_rate=0.2)
        del supernet
    else:
        raise ValueError(name)
    return net


def static_profile(net):
    """FP32 host structure accounting, NOT MCU peak-memory estimation."""
    records, handles = [], []
    def hook(name):
        def collect(m, ins, out):
            if not isinstance(out, torch.Tensor):
                return
            x = ins[0]
            macs = 0
            if isinstance(m, nn.Conv2d):
                macs = out.numel() * (m.in_channels // m.groups) * math.prod(m.kernel_size)
            elif isinstance(m, nn.Linear):
                macs = out.numel() * m.in_features
            records.append({"name": name, "type": type(m).__name__, "input": list(x.shape),
                "output": list(out.shape), "macs": macs, "output_elements": out.numel(),
                "depthwise": isinstance(m, nn.Conv2d) and m.groups == m.in_channels and m.groups > 1})
        return collect
    for name, m in net.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            handles.append(m.register_forward_hook(hook(name)))
    net.eval()
    with torch.no_grad():
        output = net(torch.zeros(1, 3, config()["image_size"], config()["image_size"], device=next(net.parameters()).device))
    for h in handles:
        h.remove()
    assert list(output.shape) == [1, 102]
    macs = sum(r["macs"] for r in records)
    return {"parameters": sum(p.numel() for p in net.parameters()),
            "parameter_bytes_fp32": sum(p.numel()*p.element_size() for p in net.parameters()),
            "macs": macs, "depthwise_mac_fraction": sum(r["macs"] for r in records if r["depthwise"]) / max(1, macs),
            "largest_output_elements": max(r["output_elements"] for r in records),
            "scope": "host_structure_only_not_mcu_peak_ram_or_flash", "layers": records}


@torch.no_grad()
def evaluate(net, data_loader, device, max_batches=None):
    net.eval()
    correct = total = 0
    loss_sum = 0.0
    per_class_n = torch.zeros(102, dtype=torch.long)
    per_class_ok = torch.zeros(102, dtype=torch.long)
    for i, (x, y) in enumerate(data_loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        logits = net(x)
        loss_sum += nn.functional.cross_entropy(logits, y, reduction="sum").item()
        ok = logits.argmax(1).eq(y)
        correct += ok.sum().item()
        total += y.numel()
        per_class_n += torch.bincount(y.cpu(), minlength=102)
        per_class_ok += torch.bincount(y[ok].cpu(), minlength=102)
    present = per_class_n > 0
    return {"top1": correct / total, "loss": loss_sum / total, "count": total,
            "mean_class_accuracy": (per_class_ok[present] / per_class_n[present]).mean().item()}


def train(args):
    cfg = config()
    run = ROOT / "runs" / args.run
    run.mkdir(parents=True, exist_ok=False)
    seed = args.seed if args.seed is not None else cfg["seed"]
    seed_all(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda" and not args.allow_cpu:
        raise RuntimeError("Training requires CUDA; use --allow-cpu only for explicit smoke tests")
    arch = arch_for(args.model, args.arch)
    head_epochs = args.head_epochs if args.head_epochs is not None else cfg["train"]["head_epochs"]
    ft_epochs = args.epochs if args.epochs is not None else cfg["train"]["finetune_epochs"]
    protocol = {"config": cfg, "model": args.model, "arch": arch, "seed": seed,
                "head_epochs": head_epochs, "finetune_epochs": ft_epochs, "device": device,
                "torch": torch.__version__, "code_sha256": digest(__file__),
                "dataset_manifest_sha256": digest(ROOT / "logs" / "dataset.json"),
                "test_used": False, "started": time.time(), "max_batches": args.max_batches}
    protocol["dataloader_start_method"] = "spawn" if os.environ.get("FLOWERS_SPAWN_WORKERS") == "1" else "platform_default"
    protocol["persistent_data_workers"] = os.environ.get("FLOWERS_PERSISTENT_WORKERS") == "1"
    protocol["actual_data_workers"] = int(os.environ.get("FLOWERS_DATA_WORKERS", cfg["train"]["workers"]))
    if protocol["actual_data_workers"] == 0:
        protocol["dataloader_start_method"] = "in_process"
        protocol["persistent_data_workers"] = False
    for filename in ["ofa.commit", "mcunet.commit"]:
        p = ROOT / "logs" / filename
        if p.exists():
            protocol[filename] = p.read_text().strip()
    save_json(run / "protocol.json", protocol)
    net = build_model(args.model, arch).to(device)
    save_json(run / "structure.json", static_profile(net))
    if args.model.startswith("ofa"):
        from ofa.imagenet_classification.elastic_nn.utils import set_running_statistics
        set_running_statistics(net, loader("train", False))
    train_loader, val_loader = loader("train", True), loader("val")
    best, best_loss = -1.0, float("inf")
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    elapsed_start = time.time()
    for phase, epochs in [("head", head_epochs), ("full", ft_epochs)]:
        for n, p in net.named_parameters():
            p.requires_grad = phase == "full" or n.startswith("classifier.")
        groups = {}
        for n, p in net.named_parameters():
            if not p.requires_grad:
                continue
            lr = cfg["train"]["head_lr"] if n.startswith("classifier.") else cfg["train"]["backbone_lr"]
            wd = 0.0 if p.ndim <= 1 or n.endswith(".bias") else cfg["train"]["weight_decay"]
            groups.setdefault((lr, wd), []).append(p)
        opt = torch.optim.AdamW([{"params": ps, "lr": lr, "weight_decay": wd} for (lr, wd), ps in groups.items()])
        for epoch in range(epochs):
            if phase == "full":
                factor = min(1.0, (epoch + 1) / 3) * 0.5 * (1 + math.cos(math.pi * epoch / max(1, epochs)))
                for g, ((base_lr, _), _) in zip(opt.param_groups, groups.items()):
                    g["lr"] = base_lr * factor
            net.train()
            # Keep pretrained/recalibrated running statistics; affine weights can train.
            for m in net.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.eval()
            loss_total, count = 0.0, 0
            for batch_idx, (x, y) in enumerate(train_loader):
                if args.max_batches is not None and batch_idx >= args.max_batches:
                    break
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device, enabled=device == "cuda"):
                    logits = net(x)
                    loss = nn.functional.cross_entropy(logits, y, label_smoothing=cfg["train"]["label_smoothing"])
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite loss")
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                loss_total += loss.item() * len(y)
                count += len(y)
            metrics = evaluate(net, val_loader, device, args.max_batches)
            entry = {"phase": phase, "epoch": epoch + 1, "train_loss": loss_total / count,
                     "validation": metrics, "elapsed_s": time.time() - elapsed_start}
            with open(run / "history.jsonl", "a") as f:
                f.write(json.dumps(entry) + "\n")
            print(json.dumps(entry), flush=True)
            if metrics["top1"] > best or (metrics["top1"] == best and metrics["loss"] < best_loss):
                best, best_loss = metrics["top1"], metrics["loss"]
                state = {"state_dict": {k: v.detach().cpu().clone() for k, v in net.state_dict().items()},
                         "model": args.model, "arch": arch, "metrics": metrics, "phase": phase,
                         "epoch": epoch + 1, "protocol": protocol}
                torch.save(state, run / "best.pt")
                save_json(run / "best.json", entry)
    best_state = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    net.load_state_dict(best_state["state_dict"], strict=True)
    reloaded = evaluate(net, val_loader, device, args.max_batches)
    assert reloaded["top1"] == best_state["metrics"]["top1"]
    save_json(run / "summary.json", {"status": "complete", "scope": "smoke" if args.max_batches else "flowers_validation",
        "validation": reloaded, "checkpoint_sha256": digest(run / "best.pt"),
        "target_met": reloaded["top1"] >= cfg["quality"]["validation_top1_target"],
        "elapsed_s": time.time() - elapsed_start, "test_used": False, "hardware_validated": False})


def native_activations(net, prefix=""):
    changes = []
    for name, module in list(net.named_children()):
        qualified = prefix + name
        if type(module).__name__ in {"Hswish", "Hsigmoid"} and type(module).__module__.startswith(("ofa.", "mcunet.")):
            new = nn.Hardswish() if type(module).__name__ == "Hswish" else nn.Hardsigmoid()
            setattr(net, name, new)
            changes.append(qualified)
        else:
            changes.extend(native_activations(module, qualified + "."))
    return changes


def fuse_global_pooling(model):
    """Fuse spatial ReduceMean pairs into the equivalent ONNX GlobalAveragePool."""
    import onnx
    from onnx import numpy_helper, helper
    graph = model.graph
    constants = {v.name: numpy_helper.to_array(v) for v in graph.initializer}
    for node in graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value":
                    constants[node.output[0]] = numpy_helper.to_array(attr.t)
    consumers = {}
    producers = {name: node for node in graph.node for name in node.output}
    for node in graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)
    def axes(node):
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        if attrs.get("keepdims", 1) != 1:
            return None
        values = attrs.get("axes")
        if values is None and len(node.input) > 1:
            values = constants.get(node.input[1])
        return set(int(x) for x in values) if values is not None else None
    skip, replacements, names = set(), {}, []
    outputs = {v.name for v in graph.output}
    for node in graph.node:
        if node.op_type != "ReduceMean" or axes(node) not in ({2}, {3}):
            continue
        previous = producers.get(node.input[0])
        if previous is None or previous.op_type != "ReduceMean":
            continue
        if axes(previous) not in ({2}, {3}) or axes(previous) == axes(node):
            continue
        if len(consumers.get(previous.output[0], [])) != 1 or previous.output[0] in outputs:
            continue
        skip.add(previous.name)
        replacements[node.name] = helper.make_node("GlobalAveragePool", [previous.input[0]],
                                                  list(node.output), name=node.name + "_global_pool")
        names.append(node.name)
    nodes = [replacements.get(node.name, node) for node in graph.node if node.name not in skip]
    del graph.node[:]
    graph.node.extend(nodes)
    return model, names


def export(args):
    import onnx
    import onnxruntime as ort
    run = ROOT / "runs" / args.run
    out = run / ("export-native-pool" if args.native_pooling else "export-native" if args.native_activations else "export")
    out.mkdir(exist_ok=False)
    state = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    seed_all(state["protocol"]["seed"])
    net = build_model(state["model"], state["arch"], pretrained=False).eval()
    net.load_state_dict(state["state_dict"], strict=True)
    ds = dataset("train")
    x = torch.stack([ds[int(i)][0] for i in np.linspace(0, len(ds)-1, 16, dtype=int)])
    with torch.no_grad():
        original = net(x).numpy()
    replacements = native_activations(net) if args.native_activations else []
    with torch.no_grad():
        expected = net(x).numpy()
    np.testing.assert_allclose(expected, original, rtol=1e-3, atol=1e-4)
    path = out / "model.onnx"
    torch.onnx.export(net, x[:1], path, opset_version=18, input_names=["input"], output_names=["logits"],
                      do_constant_folding=True)
    pooling = []
    if args.native_pooling:
        converted, pooling = fuse_global_pooling(onnx.load(str(path)))
        onnx.save(converted, str(path))
    onnx.checker.check_model(onnx.load(str(path)))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    actual = np.concatenate([session.run(None, {"input": item[None].numpy()})[0] for item in x])
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-4)
    np.savez(out / "golden.npz", inputs=x.numpy(), fp32_logits=expected)
    save_json(out / "report.json", {"onnx_check": "passed", "fp32_parity": "passed",
        "max_abs_error": float(np.max(np.abs(actual-expected))), "images": len(x),
        "native_activation_replacements": replacements,
        "native_pooling_replacements": pooling,
        "activation_rewrite_max_error": float(np.max(np.abs(original-expected))),
        "onnx_bytes": path.stat().st_size, "sha256": digest(path),
        "scope": "host_fp32_export_only", "esp32_validated": False, "quantized": False})
    print((out / "report.json").read_text(), flush=True)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    tr = sub.add_parser("train")
    tr.add_argument("--model", required=True, choices=["mbv3-small", "mcunet-in2", "ofa-min", "ofa-mid", "ofa-max", "ofa"])
    tr.add_argument("--run", required=True)
    tr.add_argument("--arch")
    tr.add_argument("--head-epochs", type=int)
    tr.add_argument("--epochs", type=int)
    tr.add_argument("--seed", type=int)
    tr.add_argument("--max-batches", type=int)
    tr.add_argument("--allow-cpu", action="store_true")
    ex = sub.add_parser("export")
    ex.add_argument("--run", required=True)
    ex.add_argument("--native-activations", action="store_true")
    ex.add_argument("--native-pooling", action="store_true")
    args = p.parse_args()
    os.chdir(ROOT)
    if args.cmd == "prepare":
        prepare()
    elif args.cmd == "train":
        train(args)
    elif args.cmd == "export":
        export(args)


if __name__ == "__main__":
    main()
