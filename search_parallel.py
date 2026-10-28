"""Resume the frozen NSGA-II protocol; only independent evaluations run in parallel.

Generation membership, RNG consumption, fitness ordering and selection are identical
to search.py. Before doing work, replay verifies every existing evaluated prefix.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import queue
import random
import subprocess
import sys
import time
from execution import preserve_cpu_affinity, environment
os.environ.setdefault("FLOWERS_PRESERVE_AFFINITY", "1")
preserve_cpu_affinity()
import search as s
import pipeline as p


def generations(cfg, method):
    rng = random.Random(cfg["seed"])
    seen = {s.identity(p.arch_for("ofa-" + size)) for size in ("min", "mid", "max")}
    population = []
    for generation in range(cfg["search"]["generations"] + 1):
        arches = []
        while len(arches) < cfg["search"]["population"]:
            if generation == 0 or method == "random":
                arch = s.sample(rng)
            else:
                previous = random.getstate()
                random.setstate(rng.getstate())
                try:
                    parents = s.tools.selTournamentDCD(population, 4)
                    rng.setstate(random.getstate())
                finally:
                    random.setstate(previous)
                arch = s.offspring(parents[0], parents[1], rng)
            ident = s.identity(arch)
            if ident in seen:
                continue
            seen.add(ident)
            arches.append(arch)
        records = yield generation, arches
        children = []
        for arch, record in zip(arches, records):
            assert s.identity(arch) == record["id"]
            ind = s.creator.Architecture(arch)
            metrics = record["proxy"]
            ind.fitness.values = (record["validation_top1"], metrics["macs"],
                metrics["storage_proxy_bytes"], metrics["runtime_memory_proxy_bytes"])
            children.append(ind)
        population = s.tools.selNSGA2(population + children, cfg["search"]["population"])
        yield [s.identity(ind) for ind in population]


def replay(directory, cfg, method):
    existing_file = directory / "evaluated.json"
    existing = json.loads(existing_file.read_text()) if existing_file.exists() else []
    generator = generations(cfg, method)
    current = next(generator)
    offset = 0
    while True:
        generation, arches = current
        for index, arch in enumerate(arches):
            if offset + index >= len(existing):
                return {"passed": True, "verified_prefix_count": len(existing), "next_generation": generation}
            record = existing[offset + index]
            assert record["id"] == s.identity(arch) and record["arch"] == arch and record["generation"] == generation
        records = existing[offset:offset + len(arches)]
        ids = generator.send(records)
        population_file = directory / ("population-%d.json" % generation)
        if population_file.exists():
            assert json.loads(population_file.read_text()) == ids
        offset += len(arches)
        try:
            current = next(generator)
        except StopIteration:
            return {"passed": True, "verified_prefix_count": len(existing), "search_complete": True}


def worker(runname, ident, generation):
    cfg = p.config()
    directory = p.ROOT / "runs" / runname
    candidate = directory / ident
    with (candidate / "evaluation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        result = candidate / "evaluation.json"
        if result.exists():
            return
        arch = json.loads((candidate / "arch.json").read_text())
        assert s.identity(arch) == ident
        p.seed_all(cfg["seed"])
        proxy_file = candidate / "proxy.json"
        metrics = json.loads(proxy_file.read_text()) if proxy_file.exists() else s.proxy(arch, candidate)
        feasible = (metrics["storage_proxy_bytes"] <= cfg["hardware"]["flash_budget_bytes"] and
            metrics["runtime_memory_proxy_bytes"] <= cfg["hardware"]["psram_budget_bytes"])
        shortname = runname + "/short-" + ident
        short = p.ROOT / "runs" / shortname
        if feasible and not (short / "summary.json").exists():
            # The old sequential evaluator must have stopped before migration.
            if short.exists():
                raise RuntimeError(f"Incomplete training directory requires inspection: {short}")
            s.run_command([sys.executable, "pipeline.py", "train", "--model", "ofa",
                "--arch", str(candidate / "arch.json"), "--run", shortname,
                "--head-epochs", str(cfg["search"]["head_epochs"]),
                "--epochs", str(cfg["search"]["finetune_epochs"])], candidate / "training.log")
        quality = json.loads((short / "summary.json").read_text())["validation"]["top1"] if feasible else -1.0
        p.save_json(result, {"id": ident, "arch": arch, "generation": generation, "short_run": shortname,
            "validation_top1": quality, "proxy_feasible": feasible, "proxy": metrics})


def run(args):
    cfg = p.config()
    directory = p.ROOT / "runs" / args.run
    plan = json.loads((directory / "plan.json").read_text())
    assert plan["config"] == cfg and plan["code_sha256"] == p.digest(p.ROOT / "search.py")
    assert plan["scope"] == "formal_search" and plan["test_used"] is False
    checked = replay(directory, cfg, plan["method"])
    print(json.dumps(checked), flush=True)
    if args.replay_only:
        return
    from verification import deployment_gate
    assert deployment_gate(p.ROOT / "runs/pilot-ofa-min-v1")["passed"]
    with (directory / "parallel-executor.lock").open("a") as executor_lock:
        fcntl.flock(executor_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        p.save_json(directory / "execution.json", {"executor": "search_parallel.py",
            "executor_sha256": p.digest(__file__), "logical_protocol": "unchanged search.py",
            "replay": checked, "gpu_slots": args.gpus, "test_used": False,
            "started_unix": time.time(), "original_plan_sha256": p.digest(directory / "plan.json")})
        slots = queue.Queue()
        for index, gpu in enumerate(args.gpus):
            slots.put((gpu, list(range(40 + 4 * index, 44 + 4 * index))))

        def execute(arch, generation):
            ident = s.identity(arch)
            candidate = directory / ident
            candidate.mkdir(exist_ok=True)
            path = candidate / "arch.json"
            if path.exists():
                assert json.loads(path.read_text()) == arch
            else:
                p.save_json(path, arch)
            if (candidate / "evaluation.json").exists():
                return json.loads((candidate / "evaluation.json").read_text())
            gpu, cpus = slots.get()
            try:
                env = environment(gpu)
                with (candidate / "executor.log").open("a") as log:
                    subprocess.run([sys.executable, "supervise.py", "--cpus", ",".join(map(str, cpus)),
                        "--descendants", "--", sys.executable,
                        "search_parallel.py", "--run", args.run, "--worker", ident,
                        "--generation", str(generation)], env=env, stdout=log, stderr=subprocess.STDOUT,
                        check=True, cwd=p.ROOT)
                return json.loads((candidate / "evaluation.json").read_text())
            finally:
                slots.put((gpu, cpus))

        old = json.loads((directory / "evaluated.json").read_text())
        evaluated = []
        gen = generations(cfg, plan["method"])
        current = next(gen)
        with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
            while True:
                generation, arches = current
                p.save_json(directory / ("planned-%d.json" % generation), arches)
                futures = [pool.submit(execute, arch, generation) for arch in arches]
                records = []
                for future in futures:  # Preserve logical order, independent of finish order.
                    record = future.result()
                    if len(evaluated) < len(old):
                        assert record == old[len(evaluated)]
                    records.append(record)
                    evaluated.append(record)
                    p.save_json(directory / "evaluated.json", evaluated)
                    print(json.dumps({"evaluated": len(evaluated), "id": record["id"],
                        "validation_top1": record["validation_top1"]}), flush=True)
                ids = gen.send(records)
                p.save_json(directory / ("population-%d.json" % generation), ids)
                try:
                    current = next(gen)
                except StopIteration:
                    break
        assert len(evaluated) == cfg["search"]["unique_budget"]
        feasible = [r for r in evaluated if r["proxy_feasible"]]
        shortlist = sorted(feasible, key=lambda r: r["validation_top1"], reverse=True)[:20]
        for record in sorted([r for r in feasible if r not in shortlist], key=lambda r: r["proxy"]["macs"]):
            if len(shortlist) >= cfg["search"]["full_training_shortlist"]:
                break
            shortlist.append(record)
        p.save_json(directory / "full-training-shortlist.json", shortlist)
        p.save_json(directory / "stage-status.json", {"status": "search_complete_requires_full_training_and_int8_verification",
            "evaluations": len(evaluated), "shortlist_count": len(shortlist), "formal_candidates_locked": False})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--gpus", type=int, nargs="+", default=[1, 2, 7])
    ap.add_argument("--replay-only", action="store_true")
    ap.add_argument("--worker")
    ap.add_argument("--generation", type=int)
    a = ap.parse_args()
    if a.worker:
        worker(a.run, a.worker, a.generation)
    else:
        run(a)
