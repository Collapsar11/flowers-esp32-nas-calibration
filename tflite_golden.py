"""One deterministic validation input per class for MCU numerical verification."""
import argparse
import os
from pathlib import Path
import json
import hashlib
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument("--run", required=True)
parser.add_argument("--quant-dir", required=True)
args = parser.parse_args()
run = Path(__file__).resolve().parent / "runs" / args.run
out = run / args.quant_dir
val = np.load(run / "tflite-data/val.npy", mmap_mode="r")
labels = np.load(run / "tflite-data/val-labels.npy")
indices = [int(np.flatnonzero(labels == c)[0]) for c in range(102)]
runtime = tf.lite.Interpreter(model_path=str(out / "model.tflite"), num_threads=4,
    experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF)
runtime.allocate_tensors()
inp, output = runtime.get_input_details()[0], runtime.get_output_details()[0]
scale, zero = inp["quantization"]
inputs, outputs = [], []
for index in indices:
    q = np.clip(np.round(val[index] / scale + zero), -128, 127).astype(np.int8)[None]
    runtime.set_tensor(inp["index"], q)
    runtime.invoke()
    inputs.append(q)
    outputs.append(runtime.get_tensor(output["index"]))
inputs, outputs = np.concatenate(inputs), np.concatenate(outputs)
np.savez(out / "golden-int8.npz", inputs=inputs, outputs=outputs)
(out / "golden-protocol.json").write_text(json.dumps({"split": "val", "indices": indices,
    "count": 102, "per_class": 1, "purpose": "functional_parity_not_test_accuracy",
    "production_sha256": hashlib.sha256((out / "model.tflite").read_bytes()).hexdigest(),
    "golden_inputs_sha256": hashlib.sha256(inputs.tobytes()).hexdigest(),
    "golden_outputs_sha256": hashlib.sha256(outputs.tobytes()).hexdigest()}, indent=2)+"\n")
print("Exported 102 class-stratified golden samples", flush=True)
