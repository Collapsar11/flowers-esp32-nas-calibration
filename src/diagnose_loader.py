"""Bounded loader diagnostic; does not train or modify experiment results."""
import argparse
import multiprocessing as mp
import os
import sys


def noop():
    pass


if __name__ == "__main__" and os.environ.get("FLOWERS_WORKER_START_METHOD") == "forkserver":
    mp.set_forkserver_preload([])
    proc = mp.get_context("forkserver").Process(target=noop)
    proc.start()
    proc.join()
    print("prewarmed forkserver", flush=True)

import pipeline as p

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    print("imported", flush=True)
    p.seed_all(20261027)
    if p.torch.cuda.is_available():
        print("cuda", p.torch.zeros(1, device="cuda").item(), flush=True)
    data = p.dataset("val")
    print("dataset", len(data), flush=True)
    extra = {"multiprocessing_context": os.environ.get("FLOWERS_WORKER_START_METHOD", "spawn")} if args.workers else {}
    loader = p.torch.utils.data.DataLoader(data, batch_size=16,
        num_workers=args.workers, pin_memory=True, **extra)
    print("before iterator", flush=True)
    if os.environ.get("FLOWERS_TRACE_LOADER") == "1":
        def trace(frame, event, arg):
            filename = frame.f_code.co_filename
            if event == "line" and ("multiprocessing/" in filename or "data/dataloader.py" in filename):
                print(filename, frame.f_lineno, flush=True)
            return trace
        sys.settrace(trace)
    iterator = iter(loader)
    sys.settrace(None)
    print("iterator ready", flush=True)
    print("batch", next(iterator)[0].shape, flush=True)
