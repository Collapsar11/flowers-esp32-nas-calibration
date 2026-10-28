"""Localize export/quantizer errors on training images only."""
import json
import sys
import pipeline as p
sys.path.insert(0, str(p.ROOT / "upstream/esp-ppq"))
import numpy as np
import torch
import onnxruntime as ort
from esp_ppq.api import load_onnx_graph
from esp_ppq.executor import TorchExecutor

p.seed_all(123)
run = p.ROOT / "runs/pilot-mbv3-small-v1"
state = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
net = p.build_model(state["model"], state["arch"], pretrained=False).eval()
net.load_state_dict(state["state_dict"])
x, y = next(iter(p.loader("train", batch=8)))
with torch.no_grad():
    ref = net(x).numpy()
results = {}
for suffix in ["export/model.onnx", "int8-esp32s3-legal-v2/quant-input.onnx"]:
    filename = str(run / suffix)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(filename, sess_options=options, providers=["CPUExecutionProvider"])
    ov = np.concatenate([session.run(None, {"input": v[None].numpy()})[0] for v in x])
    graph = load_onnx_graph(filename)
    executor = TorchExecutor(graph=graph, device="cpu")
    with torch.no_grad():
        pv = np.concatenate([executor.forward(v[None])[0].numpy() for v in x])
    results[suffix] = {"ort_max_error": float(abs(ref-ov).max()), "ppq_float_max_error": float(abs(ref-pv).max()),
        "pytorch_classes": ref.argmax(1).tolist(), "ppq_classes": pv.argmax(1).tolist(),
        "labels": y.tolist(), "ppq_minmax": [float(pv.min()), float(pv.max())]}
p.save_json(run / "quant-float-diagnostic.json", results)
print(json.dumps(results, indent=2))
