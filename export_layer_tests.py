"""Export intermediate golden tensors to locate the first runtime mismatch.

Uses training calibration input only. Diagnostic payloads are not storage metrics.
"""
import argparse
import json
import re
import shutil
import sys
import pipeline as p
from quantize import load_quant_graph
sys.path.insert(0, str(p.ROOT / "upstream/esp-ppq"))
from esp_ppq.api.espdl_interface import generate_test_value, get_target_platform
import esp_ppq.lib as pfl

parser = argparse.ArgumentParser()
parser.add_argument("--run", required=True)
parser.add_argument("--quant-dir", required=True)
parser.add_argument("--name", required=True)
parser.add_argument("--layers", type=int, default=8)
args = parser.parse_args()
p.seed_all(p.config()["seed"])
source = p.ROOT / "runs" / args.run / args.quant_dir
out = source.parent / args.name
out.mkdir(exist_ok=False)
graph = load_quant_graph(source / "quantized_graph.pt")
report = json.loads((source / "report.json").read_text())
names = re.findall(r"^\s+%(.+?) = \w+\[", (source / "model.info").read_text(), re.MULTILINE)
names = [name for name in names if name in graph.variables][:args.layers]
assert names, "No surviving intermediate tensors identified"
calibration = json.loads((source / "calibration.json").read_text())
x = p.dataset("train")[calibration["indices"][0]][0].unsqueeze(0)
values = generate_test_value(graph, "cpu", [x], output_names=names)
exporter = pfl.Exporter(platform=get_target_platform("esp32s3", report["quantization_bits"]))
exporter.export(file_path=str(out / "model-test.espdl"), graph=graph.copy(copy_value=True),
                values_for_test=values, export_config=False)
shutil.copy2(source / "model.espdl", out / "model.espdl")
p.save_json(out / "diagnostic.json", {"source": args.quant_dir, "tensor_names": names,
    "input_split": "train", "purpose": "locate_runtime_divergence", "accuracy_evaluation": False})
print(json.dumps({"exported_layers": names, "path": str(out)}, indent=2))
