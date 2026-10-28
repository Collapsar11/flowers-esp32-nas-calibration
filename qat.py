"""Pilot QAT: fixed valid S3 INT8 scales, training-only labels and teacher logits."""
import argparse
import json
import math
import sys
import time
import pipeline as p
sys.path.insert(0, str(p.ROOT / "upstream/esp-ppq"))
import torch
from esp_ppq.executor import TorchExecutor
from quantize import load_quant_graph, legalize_integer_scales


@torch.no_grad()
def validate(executor):
    correct = count = 0
    for x, y in p.loader("val"):
        logits = executor.forward(x.cuda())[0]
        assert logits.shape == (len(y), 102), logits.shape
        correct += logits.argmax(1).cpu().eq(y).sum().item()
        count += len(y)
    return correct/count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--name", default="qat-v1")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--source-file", default="quantized_graph.pt")
    parser.add_argument("--lr", type=float, default=5e-5)
    args = parser.parse_args()
    p.seed_all(p.config()["seed"])
    run = p.ROOT / "runs" / args.run
    out = run / args.name
    out.mkdir(exist_ok=False)
    source_file = run / args.source / args.source_file
    graph = load_quant_graph(source_file)
    # The deployment ONNX is static batch 1. Generalize only a leading batch
    # dimension for minibatch QAT; batch-1 inference remains identical.
    for op in graph.operations.values():
        if op.type == "Reshape" and op.inputs[1].is_parameter:
            shape = op.inputs[1].value
            if isinstance(shape, torch.Tensor) and shape.ndim == 1 and len(shape) == 2 and shape.tolist() == [1, -1]:
                op.inputs[1].value = torch.tensor([0, -1], dtype=shape.dtype)
    executor = TorchExecutor(graph=graph, device="cuda")
    params, names, seen = [], [], set()
    for op in graph.operations.values():
        if op.type not in {"Conv", "Gemm", "MatMul"}:
            continue
        for var in op.inputs[1:]:
            if not var.is_parameter or not isinstance(var.value, torch.Tensor) or not var.value.is_floating_point():
                continue
            if id(var.value) in seen:
                continue
            seen.add(id(var.value))
            var.value.requires_grad_(True)
            params.append(var.value)
            names.append(var.name)
    state = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    teacher = p.build_model(state["model"], state["arch"], pretrained=False).cuda().eval()
    teacher.load_state_dict(state["state_dict"])
    for param in teacher.parameters():
        param.requires_grad_(False)
    optimizer = torch.optim.Adam(params, lr=args.lr)
    p.save_json(out / "protocol.json", {"source_graph_sha256": p.digest(source_file),
        "epochs": args.epochs, "lr": args.lr, "schedule": "cosine_to_10_percent", "trainable_parameters": names, "quantization_scales": "fixed",
        "loss": "0.5 cross_entropy + 0.5 KD(T=2)", "test_used": False})
    best = validate(executor)
    print(json.dumps({"initial_validation_top1": best}), flush=True)
    torch.save(graph, out / "best-graph.pt")
    start = time.time()
    for epoch in range(args.epochs):
        for group in optimizer.param_groups:
            group["lr"] = args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * epoch / args.epochs)))
        total_loss, count = 0.0, 0
        for x, y in p.loader("train", True):
            x, y = x.cuda(), y.cuda()
            optimizer.zero_grad(set_to_none=True)
            logits = executor.forward_with_gradient(x)[0]
            assert logits.shape == (len(y), 102), logits.shape
            with torch.no_grad():
                teacher_logits = teacher(x)
            ce = torch.nn.functional.cross_entropy(logits, y, label_smoothing=0.05)
            kd = torch.nn.functional.kl_div(torch.log_softmax(logits/2, dim=1),
                torch.softmax(teacher_logits/2, dim=1), reduction="batchmean") * 4
            loss = 0.5 * (ce + kd)
            if not torch.isfinite(loss):
                raise RuntimeError("QAT loss is not finite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            total_loss += loss.item()*len(y)
            count += len(y)
        accuracy = validate(executor)
        record = {"epoch": epoch+1, "train_loss": total_loss/count, "validation_top1": accuracy,
                  "elapsed_s": time.time()-start}
        with (out / "history.jsonl").open("a") as f:
            f.write(json.dumps(record)+"\n")
        print(json.dumps(record), flush=True)
        if accuracy > best:
            best = accuracy
            torch.save(graph, out / "best-graph.pt")
    p.save_json(out / "summary.json", {"best_validation_top1": best, "test_used": False,
        "target_met": best >= p.config()["quality"]["validation_top1_target"], "epochs": args.epochs,
        "firmware_verified": False, "graph_sha256": p.digest(out / "best-graph.pt")})


if __name__ == "__main__":
    main()
