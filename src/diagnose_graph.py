"""Inspect a saved quantized graph without recalibration or any test-set access."""
import sys
import json
import pipeline as p
sys.path.insert(0, str(p.ROOT / "upstream/esp-ppq"))
import torch
from esp_ppq.executor import TorchExecutor
from esp_ppq.quantization.analyse import graphwise_error_analyse
from esp_ppq.core import QuantizationStates

p.seed_all(123)
path = p.ROOT / "runs/pilot-ofa-min-v1/int8-esp32s3-legal-v1/quantized_graph.pt"
graph = torch.load(path, map_location="cpu", weights_only=False)
# PPQ serializes edges as names; restore the same links as its NativeImporter.
for op in graph.operations.values():
    op.inputs[:] = [graph.variables[name] for name in op.inputs]
    op.outputs[:] = [graph.variables[name] for name in op.outputs]
for var in graph.variables.values():
    var.dest_ops[:] = [graph.operations[name] for name in var.dest_ops]
    if var.source_op is not None:
        var.source_op = graph.operations[var.source_op]
graph._graph_inputs = {name: graph.variables[name] for name in graph._graph_inputs}
graph._graph_outputs = {name: graph.variables[name] for name in graph._graph_outputs}
ds = p.dataset("train")
xs = [ds[i][0].unsqueeze(0) for i in [0, 100, 300, 500, 700, 1000]]
results = {}
for device in ["cpu", "cuda"]:
    executor = TorchExecutor(graph=graph, device=device)
    with torch.no_grad():
        outputs = [executor.forward(x.to(device))[0].detach().cpu() for x in xs]
    results[device] = [{"class": int(o.argmax()), "min": float(o.min()), "max": float(o.max()),
                       "logits": o.flatten().tolist()} for o in outputs]
errors = graphwise_error_analyse(graph, "cpu", xs[:2], collate_fn=lambda x: x, steps=2, verbose=False)
results["cumulative_error_in_topological_order"] = [
    {"op": op.name, "type": op.type, "error": errors[op.name]}
    for op in graph.topological_sort() if op.name in errors]
p.save_json(path.parent / "diagnostic.json", results)
print(json.dumps({k: [{kk: vv for kk,vv in o.items() if kk != "logits"} for o in v]
                  for k,v in results.items()}, indent=2))
