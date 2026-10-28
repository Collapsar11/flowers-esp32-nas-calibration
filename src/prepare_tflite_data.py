"""Reuse exactly the train calibration IDs and preprocessing for another runtime."""
import argparse
import json
import numpy as np
import pipeline as p

parser = argparse.ArgumentParser()
parser.add_argument("--run", required=True)
args = parser.parse_args()
run = p.ROOT / "runs" / args.run
out = run / "tflite-data"
out.mkdir(exist_ok=False)
cal_file = p.ROOT / "runs/pilot-ofa-min-v1/int8-esp32s3-native-pool-v1/calibration.json"
cal = json.loads(cal_file.read_text())
for split, ids in [("train", cal["indices"]), ("val", list(range(1020)))]:
    ds = p.dataset(split)
    array = np.lib.format.open_memmap(out / (split + ".npy"), mode="w+", dtype=np.float32,
        shape=(len(ids), p.config()["image_size"], p.config()["image_size"], 3))
    labels = []
    for row, index in enumerate(ids):
        x, y = ds[index]
        array[row] = x.permute(1, 2, 0).numpy()
        labels.append(y)
    array.flush()
    np.save(out / (split + "-labels.npy"), np.array(labels, dtype=np.int64))
p.save_json(out / "protocol.json", {"calibration": cal, "validation_count": 1020,
    "layout": "NHWC", "normalization": "same ImageNet normalization as pipeline.py",
    "test_used": False})
print(str(out), flush=True)
