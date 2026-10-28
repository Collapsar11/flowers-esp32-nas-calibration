"""Read/patch the verified TFLite graph; run in the isolated TensorFlow environment.

Only weight and bias buffers change during QAT export. Operators, shapes, all
quantization parameters and other constants retain the verified deployment ABI.
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from execution import preserve_cpu_affinity
preserve_cpu_affinity()
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import flatbuffers
import numpy as np
import tensorflow as tf
from tensorflow.lite.python import schema_py_generated as s

tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(1)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def interpreter(path):
    obj = tf.lite.Interpreter(model_path=str(path), num_threads=4,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF,
        experimental_preserve_all_tensors=True)
    obj.allocate_tensors()
    return obj


def extract(source, out):
    out.mkdir(exist_ok=False)
    quant_path = source / "model.tflite"
    float_path = source / "saved_model/source_float32.tflite"
    qi, fi = interpreter(quant_path), interpreter(float_path)
    qm = s.Model.GetRootAsModel(quant_path.read_bytes(), 0)
    fm = s.Model.GetRootAsModel(float_path.read_bytes(), 0)
    qg, fg = qm.Subgraphs(0), fm.Subgraphs(0)
    qops, fops = qi._get_ops_details(), fi._get_ops_details()
    assert len(qops) == len(fops) == qg.OperatorsLength()
    qd = {x["index"]: x for x in qi.get_tensor_details()}
    fd = {x["index"]: x for x in fi.get_tensor_details()}
    tensors = {str(k): {"shape": d["shape"].tolist(),
        "dtype": str(np.dtype(d["dtype"])),
        "scales": d["quantization_parameters"]["scales"].tolist(),
        "zeros": d["quantization_parameters"]["zero_points"].tolist(),
        "axis": int(d["quantization_parameters"]["quantized_dimension"])} for k, d in qd.items()}
    constants, ops = {}, []
    producer_map = {int(fi.get_input_details()[0]["index"]): int(qi.get_input_details()[0]["index"])}
    options = {"CONV_2D": s.Conv2DOptions, "DEPTHWISE_CONV_2D": s.DepthwiseConv2DOptions,
        "FULLY_CONNECTED": s.FullyConnectedOptions, "ADD": s.AddOptions, "MUL": s.MulOptions,
        "MEAN": s.ReducerOptions}
    for idx, (qo, fo) in enumerate(zip(qops, fops)):
        name = qo["op_name"]
        assert name == fo["op_name"], (idx, name, fo["op_name"])
        qins, fins = qo["inputs"].tolist(), fo["inputs"].tolist()
        assert len(qins) == len(fins)
        for qt, ft in zip(qins, fins):
            if qt < 0:
                assert ft < 0
                continue
            assert qd[qt]["shape"].tolist() == fd[ft]["shape"].tolist()
            if ft in producer_map:
                assert producer_map[ft] == qt, (idx, qt, ft, "edge mismatch")
            else:
                value = fi.get_tensor(ft)
                if str(qt) in constants:
                    assert np.array_equal(constants[str(qt)], value)
                constants[str(qt)] = value
        for qt, ft in zip(qo["outputs"], fo["outputs"]):
            assert qd[qt]["shape"].tolist() == fd[ft]["shape"].tolist()
            producer_map[int(ft)] = int(qt)
        opt = {}
        if name in options:
            for graph in [qg, fg]:
                table = graph.Operators(idx).BuiltinOptions()
                obj = options[name]()
                obj.Init(table.Bytes, table.Pos)
                values = {attr: int(getattr(obj, attr)()) for attr in ["Padding", "StrideH", "StrideW",
                    "DilationHFactor", "DilationWFactor", "DepthMultiplier", "FusedActivationFunction", "KeepDims"]
                    if hasattr(obj, attr)}
                if opt:
                    assert values == opt, (idx, values, opt)
                opt = values
        ops.append({"name": name, "inputs": qins, "outputs": qo["outputs"].tolist(), "options": opt})
    meta = {"source_model": str(quant_path.resolve()), "source_sha256": sha(quant_path),
        "fp32_source_sha256": sha(float_path), "input": int(qi.get_input_details()[0]["index"]),
        "output": int(qi.get_output_details()[0]["index"]), "tensors": tensors, "ops": ops,
        "test_used": False}
    write(out / "graph.json", meta)
    np.savez(out / "float-constants.npz", **constants)
    print(json.dumps({"extracted_ops": len(ops), "source_sha256": meta["source_sha256"]}), flush=True)


def patch(graph_dir, weights, dest):
    meta = json.loads((graph_dir / "graph.json").read_text())
    source = Path(meta["source_model"])
    assert sha(source) == meta["source_sha256"]
    model = s.ModelT.InitFromObj(s.Model.GetRootAsModel(source.read_bytes(), 0))
    arrays = np.load(weights)
    changed = {}
    for op in meta["ops"]:
        if op["name"] not in {"CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED"}:
            continue
        for tid in op["inputs"][1:]:
            if tid < 0:
                continue
            key = str(tid)
            d = meta["tensors"][key]
            value = arrays[key]
            assert list(value.shape) == d["shape"] and np.isfinite(value).all()
            scales, zeros = np.array(d["scales"]), np.array(d["zeros"])
            shape = [1] * value.ndim
            shape[d["axis"]] = len(scales)
            dtype = np.dtype(d["dtype"])
            assert dtype in (np.dtype("int8"), np.dtype("int32"))
            info = np.iinfo(dtype)
            q = np.clip(np.rint(value / scales.reshape(shape)) + zeros.reshape(shape),
                info.min, info.max).astype(dtype)
            buffer_id = model.subgraphs[0].tensors[tid].buffer
            raw = q.tobytes()
            assert buffer_id not in changed or changed[buffer_id] == raw
            changed[buffer_id] = raw
            model.buffers[buffer_id].data = np.frombuffer(raw, dtype=np.uint8)
    builder = flatbuffers.Builder(0)
    builder.Finish(model.Pack(builder), file_identifier=b"TFL3")
    dest.write_bytes(builder.Output())
    # Independently verify all operator tables and quantization metadata unchanged.
    patched = s.ModelT.InitFromObj(s.Model.GetRootAsModel(dest.read_bytes(), 0))
    original = s.ModelT.InitFromObj(s.Model.GetRootAsModel(source.read_bytes(), 0))
    for bid in changed:
        patched.buffers[bid].data = original.buffers[bid].data
    b1, b2 = flatbuffers.Builder(0), flatbuffers.Builder(0)
    b1.Finish(patched.Pack(b1), file_identifier=b"TFL3")
    b2.Finish(original.Pack(b2), file_identifier=b"TFL3")
    assert bytes(b1.Output()) == bytes(b2.Output()), "export modified something other than trained weights"
    write(dest.with_suffix(".audit.json"), {"passed": True, "changed_buffers": len(changed),
        "only_weight_bias_buffers_changed": True, "source_sha256": meta["source_sha256"],
        "production_sha256": sha(dest)})


def evaluate(model, data, dest, run):
    it = interpreter(model)
    inp, output = it.get_input_details()[0], it.get_output_details()[0]
    assert inp["dtype"] == output["dtype"] == np.int8
    details = it.get_tensor_details()
    assert all(not np.issubdtype(d["dtype"], np.floating) for d in details)
    x, y = np.load(data / "val.npy", mmap_mode="r"), np.load(data / "val-labels.npy")
    scale, zero = inp["quantization"]
    correct, predictions = 0, []
    started = time.monotonic()
    for image, label in zip(x, y):
        qx = np.clip(np.rint(image[None] / scale) + zero, -128, 127).astype(np.int8)
        it.set_tensor(inp["index"], qx)
        it.invoke()
        pred = int(it.get_tensor(output["index"]).argmax())
        correct += pred == int(label)
        predictions.append(pred)
        if len(predictions) % 255 == 0:
            print(json.dumps({"integer_validation_progress": len(predictions), "total": len(y),
                "elapsed_s": time.monotonic() - started}), flush=True)
    fp32 = json.loads((run / "summary.json").read_text())["validation"]["top1"]
    report = {"runtime": "tflite_builtin_reference", "quantization_bits": 8,
        "int8_top1": correct / len(y), "quantization_drop": fp32 - correct / len(y),
        "original_fp32_top1": fp32, "validation_count": len(y), "test_used": False,
        "production_model_bytes": model.stat().st_size, "production_sha256": sha(model),
        "all_tensors_integer": True, "firmware_verified": False, "real_hardware_validated": False}
    write(dest, report)
    np.save(dest.with_suffix(".predictions.npy"), np.array(predictions))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("extract")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("patch")
    p.add_argument("--graph", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--dest", type=Path, required=True)
    p = sub.add_parser("evaluate")
    for key in ["model", "data", "dest", "run"]:
        p.add_argument("--" + key, type=Path, required=True)
    a = ap.parse_args()
    if a.command == "extract":
        extract(a.source, a.out)
    elif a.command == "patch":
        patch(a.graph, a.weights, a.dest)
    else:
        evaluate(a.model, a.data, a.dest, a.run)
