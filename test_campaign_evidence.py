"""Synthetic rejection tests only; fixtures are not experimental evidence."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import campaign
from verification import batch_checks


class BatchEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "model.tflite").write_bytes(b"model-fixture")
        self.gold = {"inputs": np.arange(204, dtype=np.int16).reshape(102, 2),
                     "outputs": np.arange(102, dtype=np.int16).reshape(102, 1)}
        np.savez(self.root / "golden-int8.npz", **self.gold)
        self.manifest = {"sdkconfig_sha256": hashlib.sha256(b"sdk").hexdigest(), "batches": []}
        self.aggregate = {"validation_internal_peak_delta": 0, "validation_psram_peak_delta": 100,
                          "tensor_arena_reserved_bytes": 100, "tensor_arena_used_bytes": 90}
        for index in range(2):
            directory = self.root / str(index)
            directory.mkdir()
            flash_bytes = b"model-fixture" + bytes([index])
            (directory / "flash_image.bin").write_bytes(flash_bytes)
            (directory / "sdkconfig").write_bytes(b"sdk")
            binary_sha = hashlib.sha256(flash_bytes).hexdigest()
            manifest = {"production_model_sha256": "model", "flash_image_sha256": binary_sha,
                        "sdkconfig_sha256": self.manifest["sdkconfig_sha256"]}
            for key, values in self.gold.items():
                manifest["golden_" + key + "_sha256"] = hashlib.sha256(values[index*51:(index+1)*51].tobytes()).hexdigest()
            result = dict(self.aggregate, functional_pass=True, golden_count=51,
                          integer_mismatches=0, max_integer_error=0, flash_image_sha256=binary_sha)
            (directory / "manifest.json").write_text(json.dumps(manifest))
            (directory / "qemu-report.json").write_text(json.dumps(result))
            self.manifest["batches"].append({"directory": str(index), "offset": index*51,
                "count": 51, "flash_image_sha256": binary_sha})

    def check(self):
        return batch_checks(self.root, self.root, self.manifest, self.aggregate, "model")

    def test_complete_coverage(self):
        self.assertTrue(all(self.check().values()))

    def test_missing_binary(self):
        (self.root / "1/flash_image.bin").unlink()
        self.assertFalse(self.check()["batch_artifacts"])

    def test_tampered_binary(self):
        (self.root / "1/flash_image.bin").write_bytes(b"changed")
        self.assertFalse(self.check()["flash_identity"])

    def test_wrong_embedded_model_even_with_consistent_flash_hashes(self):
        directory = self.root / "1"
        wrong = b"different-model"
        sha = hashlib.sha256(wrong).hexdigest()
        (directory / "flash_image.bin").write_bytes(wrong)
        for filename in ["manifest.json", "qemu-report.json"]:
            file = directory / filename
            value = json.loads(file.read_text())
            value["flash_image_sha256"] = sha
            file.write_text(json.dumps(value))
        self.manifest["batches"][1]["flash_image_sha256"] = sha
        self.assertTrue(self.check()["flash_identity"])
        self.assertFalse(self.check()["embedded_model_identity"])

    def test_overlapping_coverage(self):
        self.manifest["batches"][1]["offset"] = 0
        self.assertFalse(self.check()["batch_coverage"])

    def test_wrong_model(self):
        file = self.root / "1/manifest.json"
        value = json.loads(file.read_text())
        value["production_model_sha256"] = "different"
        file.write_text(json.dumps(value))
        self.assertFalse(self.check()["batch_model_identity"])

    def test_hidden_integer_error(self):
        file = self.root / "1/qemu-report.json"
        value = json.loads(file.read_text())
        value["integer_mismatches"] = 1
        file.write_text(json.dumps(value))
        self.assertFalse(self.check()["batch_strict_parity"])

    def test_underreported_memory(self):
        self.aggregate["validation_psram_peak_delta"] = 99
        self.assertFalse(self.check()["batch_memory_aggregation"])

    def test_altered_golden(self):
        self.gold["outputs"][60] += 1
        np.savez(self.root / "golden-int8.npz", **self.gold)
        self.assertFalse(self.check()["batch_golden_identity"])


class SelectionTests(unittest.TestCase):
    def test_distinct_deterministic_accuracy_and_resource_selection(self):
        states = [dict(id=str(i), int8_top1=.90 + i / 1000, macs=100+i,
            production_model_bytes=200+i, psram_peak_bytes=300+2*i,
            depthwise_mac_fraction=i/100, arch={"d": [2+i%3]*5}) for i in range(30)]
        original = copy.deepcopy(states)
        chosen = campaign.select_diverse(states)
        self.assertEqual(len({s["id"] for s in chosen}), 15)
        self.assertEqual([s["id"] for s in chosen[:5]], [str(i) for i in range(29, 24, -1)])
        self.assertEqual(chosen, campaign.select_diverse(list(reversed(states))))
        self.assertEqual(states, original)


if __name__ == "__main__":
    unittest.main()
