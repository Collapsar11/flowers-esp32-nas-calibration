"""Alternative full INT8, per-channel-weight export; no MCU success inferred."""
import argparse
from collections import Counter
import hashlib
import importlib
import json
import os
import shutil
from pathlib import Path
from execution import preserve_cpu_affinity
preserve_cpu_affinity()
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
import numpy as np
import tensorflow as tf
from onnx2tf import convert

parser = argparse.ArgumentParser()
parser.add_argument("--run", required=True)
parser.add_argument("--tag", default="tflite-int8-v2")
parser.add_argument("--mcu-activations", action="store_true")
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
run = root / "runs" / args.run
out = run / args.tag
out.mkdir(exist_ok=False)
shutil.copy2(run / "export-native-pool/model.onnx", out / "source.onnx")
if args.mcu_activations:
    # TFLM lacks RELU_0_TO_1. Express HardSigmoid through supported ReLU6.
    import onnx
    from onnx import helper, numpy_helper
    model = onnx.load(str(out / "source.onnx"))
    nodes = []
    for node in model.graph.node:
        if node.op_type != "HardSigmoid":
            nodes.append(node)
            continue
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        base = node.name + "_mcu"
        constants = {"alpha": attrs.get("alpha", 0.2)*6, "beta": attrs.get("beta", 0.5)*6,
                     "zero": 0, "six": 6, "one_sixth": 1/6}
        for name, value in constants.items():
            model.graph.initializer.append(numpy_helper.from_array(np.array(value, dtype=np.float32), base+"_"+name))
        nodes.extend([helper.make_node("Mul", [node.input[0], base+"_alpha"], [base+"_scaled"], name=base+"_scale"),
            helper.make_node("Add", [base+"_scaled", base+"_beta"], [base+"_shifted"], name=base+"_shift"),
            helper.make_node("Clip", [base+"_shifted", base+"_zero", base+"_six"], [base+"_relu6"], name=base+"_clip"),
            helper.make_node("Mul", [base+"_relu6", base+"_one_sixth"], list(node.output), name=base+"_output")])
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    onnx.checker.check_model(model)
    onnx.save(model, str(out / "source.onnx"))
# This pinned converter fetches example images even with custom inputs.
# Supply our already-normalized training sample instead; no external sample download.
importlib.import_module("onnx2tf.onnx2tf").download_test_image_data = lambda: np.load(
    run / "tflite-data/train.npy", mmap_mode="r")[:1].copy()
convert(input_onnx_file_path=str(out / "source.onnx"),
        output_folder_path=str(out / "saved_model"), output_signaturedefs=True, non_verbose=True,
        custom_input_op_name_np_data_path=[["input", str(run / "tflite-data/train.npy")]])
golden = np.load(run / "export-native-pool/golden.npz")
fp_file = out / "saved_model/source_float32.tflite"
fp = tf.lite.Interpreter(model_path=str(fp_file), num_threads=4)
fp.allocate_tensors()
fp_in, fp_out = fp.get_input_details()[0], fp.get_output_details()[0]
observed = []
for x in golden["inputs"]:
    fp.set_tensor(fp_in["index"], x.transpose(1, 2, 0)[None])
    fp.invoke()
    observed.append(fp.get_tensor(fp_out["index"]))
observed = np.concatenate(observed)
np.testing.assert_allclose(observed, golden["fp32_logits"], rtol=1e-3, atol=1e-4)
cal = np.load(run / "tflite-data/train.npy", mmap_mode="r")
def representative():
    for x in cal:
        yield [x[None]]
converter = tf.lite.TFLiteConverter.from_saved_model(str(out / "saved_model"))
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
binary = converter.convert()
(out / "model.tflite").write_bytes(binary)
runtime = tf.lite.Interpreter(model_content=binary, num_threads=4,
    experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF)
runtime.allocate_tensors()
inp, output = runtime.get_input_details()[0], runtime.get_output_details()[0]
assert inp["dtype"] == np.int8 and output["dtype"] == np.int8
assert not any(np.issubdtype(t["dtype"], np.floating) for t in runtime.get_tensor_details())
val = np.load(run / "tflite-data/val.npy", mmap_mode="r")
labels = np.load(run / "tflite-data/val-labels.npy")
scale, zero = inp["quantization"]
correct = 0
first_inputs, first_outputs = [], []
for i, (x, y) in enumerate(zip(val, labels)):
    q = np.clip(np.round(x / scale + zero), -128, 127).astype(np.int8)[None]
    runtime.set_tensor(inp["index"], q)
    runtime.invoke()
    result = runtime.get_tensor(output["index"])
    correct += int(result.argmax() == y)
    if i < 16:
        first_inputs.append(q)
        first_outputs.append(result)
    if (i+1) % 100 == 0:
        print(json.dumps({"validated": i+1, "int8_top1": correct/(i+1)}), flush=True)
np.savez(out / "golden-int8.npz", inputs=np.concatenate(first_inputs), outputs=np.concatenate(first_outputs))
fp_accuracy = json.loads((run / "summary.json").read_text())["validation"]["top1"]
report = {"runtime": "tflite_builtin_reference", "quantization_bits": 8,
    "fp32_parity_max_error": float(np.abs(observed-golden["fp32_logits"]).max()),
    "int8_top1": correct/len(val), "quantization_drop": fp_accuracy-correct/len(val),
    "validation_count": len(val), "production_model_bytes": len(binary),
    "production_sha256": hashlib.sha256(binary).hexdigest(),
    "per_channel_tensor_count": sum(len(t["quantization_parameters"]["scales"]) > 1 for t in runtime.get_tensor_details()),
    "operators": dict(Counter(op["op_name"] for op in runtime._get_ops_details())),
    "test_used": False, "firmware_verified": False, "real_hardware_validated": False}
(out / "report.json").write_text(json.dumps(report, indent=2)+"\n")
print(json.dumps(report, indent=2), flush=True)
