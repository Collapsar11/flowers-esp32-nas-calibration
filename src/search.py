"""Pretrained OFA NSGA-II and equal-budget random search, with no hardware labels."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys

import pipeline as p
import torch
from deap import base, creator, tools

creator.create("ResourceFitness", base.Fitness, weights=(1.0, -1.0, -1.0, -1.0))
creator.create("Architecture", dict, fitness=creator.ResourceFitness)


def canonical(arch):
    result = copy.deepcopy(arch)
    assert len(result["d"]) == 5 and len(result["e"]) == len(result["ks"]) == 20
    for s, depth in enumerate(result["d"]):
        assert depth in p.config()["depth"]
        for i in range(4):
            j = s * 4 + i
            assert result["e"][j] in p.config()["expand"]
            assert result["ks"][j] in p.config()["kernel"]
            if i >= depth:
                result["e"][j] = result["ks"][j] = 3
    return {k: result[k] for k in ("d", "e", "ks")}


def identity(arch):
    return hashlib.sha256(json.dumps(canonical(arch), sort_keys=True).encode()).hexdigest()[:16]


def sample(rng):
    cfg = p.config()
    return canonical({"d": [rng.choice(cfg["depth"]) for _ in range(5)],
        "e": [rng.choice(cfg["expand"]) for _ in range(20)],
        "ks": [rng.choice(cfg["kernel"]) for _ in range(20)]})


def offspring(a, b, rng):
    cfg = p.config()
    child = {}
    for field, choices in [("d", cfg["depth"]), ("e", cfg["expand"]), ("ks", cfg["kernel"])]:
        child[field] = []
        for av, bv in zip(a[field], b[field]):
            value = rng.choice([av, bv])
            if rng.random() < cfg["search"]["mutation_probability"]:
                value = rng.choice(choices)
            child[field].append(value)
    # Construct a fresh individual. No parent fitness or checkpoint is inherited.
    return canonical(child)


def proxy(arch, directory):
    """Topology liveness estimate, explicitly not ESP-DL allocator or scratch memory."""
    import onnx
    net = p.build_model("ofa", arch, pretrained=False).eval()
    structure = p.static_profile(net)
    path = directory / "topology.onnx"
    with torch.no_grad():
        torch.onnx.export(net, torch.zeros(1, 3, p.config()["image_size"], p.config()["image_size"]),
            path, opset_version=18, input_names=["input"], output_names=["logits"])
    model = onnx.shape_inference.infer_shapes(onnx.load(path))
    graph = model.graph
    constants = {v.name for v in graph.initializer}
    sizes = {}
    for value in list(graph.input) + list(graph.value_info) + list(graph.output):
        tt = value.type.tensor_type
        dims = [d.dim_value for d in tt.shape.dim]
        if any(d <= 0 for d in dims):
            raise ValueError(f"Unknown shape in memory proxy: {value.name}")
        sizes[value.name] = math.prod(dims) * (8 if tt.elem_type == onnx.TensorProto.INT64 else 1)
    last_use = {}
    for i, node in enumerate(graph.node):
        for name in node.input:
            last_use[name] = i
    for value in graph.output:
        last_use[value.name] = len(graph.node)
    live = {x.name for x in graph.input if x.name not in constants}
    peak = sum(sizes[n] for n in live)
    for i, node in enumerate(graph.node):
        for name in node.output:
            if name not in sizes:
                raise ValueError(f"Missing shape: {name}")
            live.add(name)
        peak = max(peak, sum(sizes[n] for n in live))
        live = {n for n in live if last_use.get(n, -1) > i}
    weights = sum(math.prod(v.dims) * (4 if len(v.dims) <= 1 else 1) for v in graph.initializer)
    result = {"macs": structure["macs"], "parameters": structure["parameters"],
        "depthwise_mac_fraction": structure["depthwise_mac_fraction"],
        "int8_weight_bytes_estimate": weights,
        "storage_proxy_bytes": math.ceil(weights * 1.1) + 65536,
        "live_activation_bytes_estimate": peak,
        "runtime_memory_proxy_bytes": 2 * peak + 262144,
        "scope": "screening_estimate_excludes_target_allocator_scratch_and_application",
        "memory_margin_rule": "2x live activation plus 256 KiB; verify using firmware",
        "is_real_hardware_label": False}
    p.save_json(directory / "proxy.json", result)
    path.unlink()  # Only this temporary topology export; trained exports are retained.
    return result


def run_command(command, logfile):
    with logfile.open("a") as f:
        subprocess.run(command, stdout=f, stderr=subprocess.STDOUT, check=True, cwd=p.ROOT)


def search(args):
    cfg = p.config()
    if args.smoke:
        cfg["search"].update(population=4, generations=1, unique_budget=8,
                             head_epochs=1, finetune_epochs=1, full_training_shortlist=4)
    p.seed_all(cfg["seed"])
    directory = p.ROOT / "runs" / args.run
    directory.mkdir(exist_ok=True)
    plan = {"method": args.method, "seed": cfg["seed"], "config": cfg,
            "code_sha256": p.digest(__file__), "test_used": False, "hardware_labels_used": 0,
            "scope": "software_smoke_only" if args.smoke else "formal_search"}
    plan_path = directory / "plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != plan:
            raise ValueError("Cannot resume with a changed search protocol")
    else:
        p.save_json(plan_path, plan)
    if cfg["hardware"]["target"] != "esp32s3":
        raise ValueError("Confirm target before constrained search")
    for key in ["flash_budget_bytes", "psram_budget_bytes", "internal_ram_budget_bytes"]:
        if not cfg["hardware"][key]:
            raise ValueError(f"Missing {key}")
    # Validate the search family's deployment path before launching the expensive search.
    pilots = [p.ROOT / "runs" / f"pilot-ofa-{s}-v1/summary.json" for s in ("min", "mid", "max")]
    if not args.smoke and (not all(x.exists() for x in pilots) or not any(json.loads(x.read_text())["target_met"] for x in pilots)):
        raise RuntimeError("Search not launched: finish OFA pilots and meet the validation quality floor")
    from verification import deployment_gate
    if not args.smoke:
        accepted = {}
        for pilot in pilots:
            gate = deployment_gate(pilot.parent)
            if gate["passed"]:
                accepted[pilot.parent.name] = gate
        if not accepted:
            raise RuntimeError("Search not launched: no OFA pilot passes INT8, memory, and QEMU deployment gates")
        p.save_json(directory / "deployment-precondition.json", {"accepted_pilots": accepted,
            "runtime": cfg["hardware"].get("runtime"), "real_hardware_validated": False})
    rng = random.Random(cfg["seed"])
    excluded = {identity(p.arch_for("ofa-" + s)) for s in ("min", "mid", "max")}
    seen, evaluated = set(excluded), []
    population = []
    n = cfg["search"]["population"]
    for generation in range(cfg["search"]["generations"] + 1):
        children = []
        while len(children) < n:
            if generation == 0 or args.method == "random":
                arch = sample(rng)
            else:
                # DEAP's crowded tournament; population has NSGA-II crowding distances.
                # DEAP uses module-global random. Isolate it from model building
                # so cached/resumed profiling cannot alter the search sequence.
                previous_random_state = random.getstate()
                random.setstate(rng.getstate())
                try:
                    parents = tools.selTournamentDCD(population, 4)
                    rng.setstate(random.getstate())
                finally:
                    random.setstate(previous_random_state)
                arch = offspring(parents[0], parents[1], rng)
            ident = identity(arch)
            if ident in seen:
                continue
            seen.add(ident)
            candidate = directory / ident
            candidate.mkdir(exist_ok=True)
            archpath = candidate / "arch.json"
            p.save_json(archpath, arch)
            proxypath = candidate / "proxy.json"
            metrics = json.loads(proxypath.read_text()) if proxypath.exists() else proxy(arch, candidate)
            feasible = metrics["storage_proxy_bytes"] <= cfg["hardware"]["flash_budget_bytes"] and metrics["runtime_memory_proxy_bytes"] <= cfg["hardware"]["psram_budget_bytes"]
            runname = f"{args.run}/short-{ident}"
            summary = p.ROOT / "runs" / runname / "summary.json"
            if feasible and not summary.exists():
                run_command([sys.executable, 'src/pipeline.py', "train", "--model", "ofa", "--arch", str(archpath),
                    "--run", runname, "--head-epochs", str(cfg["search"]["head_epochs"]),
                    "--epochs", str(cfg["search"]["finetune_epochs"])] +
                    (["--max-batches", "2"] if args.smoke else []), candidate / "training.log")
            quality = json.loads(summary.read_text())["validation"]["top1"] if feasible else -1.0
            individual = creator.Architecture(arch)
            individual.fitness.values = (quality, metrics["macs"], metrics["storage_proxy_bytes"], metrics["runtime_memory_proxy_bytes"])
            children.append(individual)
            record = {"id": ident, "arch": arch, "generation": generation, "short_run": runname,
                      "validation_top1": quality, "proxy_feasible": feasible, "proxy": metrics}
            evaluated.append(record)
            p.save_json(candidate / "evaluation.json", record)
            p.save_json(directory / "evaluated.json", evaluated)
            print(json.dumps({"evaluated": len(evaluated), "id": ident, "validation_top1": quality}), flush=True)
        population = tools.selNSGA2(population + children, n)
        p.save_json(directory / f"population-{generation}.json", [identity(i) for i in population])
    assert len(evaluated) == cfg["search"]["unique_budget"]
    # Full training shortlist: high quality plus resource-diverse candidates.
    feasible = [r for r in evaluated if r["proxy_feasible"]]
    by_quality = sorted(feasible, key=lambda r: r["validation_top1"], reverse=True)
    shortlist = by_quality[:min(20, cfg["search"]["full_training_shortlist"])]
    rest = [r for r in feasible if r not in shortlist]
    for record in sorted(rest, key=lambda r: r["proxy"]["macs"]):
        if len(shortlist) >= cfg["search"]["full_training_shortlist"]:
            break
        shortlist.append(record)
    p.save_json(directory / "full-training-shortlist.json", shortlist)
    p.save_json(directory / "stage-status.json", {"status": "software_smoke_complete" if args.smoke else "search_complete_requires_full_training_and_int8_verification",
        "evaluations": len(evaluated), "shortlist_count": len(shortlist), "formal_candidates_locked": False})


def self_test():
    rng = random.Random(123)
    a = p.arch_for("ofa-min")
    b = copy.deepcopy(a)
    b["e"][3] = 6
    assert identity(a) == identity(b), "Inactive genes must not create duplicate architectures"
    seen = set()
    for _ in range(100):
        child = offspring(a, sample(rng), rng)
        individual = creator.Architecture(child)
        assert not individual.fitness.valid
        seen.add(identity(child))
    assert len(seen) > 90
    print("PASS: canonical uniqueness, legal OFA genes, variation, invalidated offspring fitness")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--method", choices=["nsga2", "random"], default="nsga2")
    parser.add_argument("--run")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        if not args.run:
            parser.error("--run is required")
        search(args)
