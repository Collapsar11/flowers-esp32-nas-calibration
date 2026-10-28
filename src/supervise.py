"""Keep an owned job on its assigned CPUs even while imports hold Python's GIL."""
import argparse
import os
from pathlib import Path
import subprocess
import time


def constrain(pid, cpus, descendants=False, gpu_map=None):
    proc = Path('/proc') / str(pid)
    try:
        if gpu_map:
            entries = (proc / 'environ').read_bytes().split(b'\0')
            gpu = next((e.split(b'=', 1)[1].decode() for e in entries if e.startswith(b'CUDA_VISIBLE_DEVICES=')), None)
            cpus = gpu_map.get(gpu, cpus)
        for task in (proc / 'task').iterdir():
            tid = int(task.name)
            if set(os.sched_getaffinity(tid)) != cpus:
                os.sched_setaffinity(tid, cpus)
        if descendants:
            for child in (proc / 'task' / str(pid) / 'children').read_text().split():
                constrain(int(child), cpus, True, gpu_map)
    except (FileNotFoundError, ProcessLookupError):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpus', required=True)
    parser.add_argument('--pid', type=int)
    parser.add_argument('--descendants', action='store_true')
    parser.add_argument('--gpu-cpus', action='append', default=[], help='Existing owned worker allocation GPU:CPU,CPU,...')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cpus = set(map(int, args.cpus.split(',')))
    gpu_map = {entry.split(':')[0]: set(map(int, entry.split(':')[1].split(','))) for entry in args.gpu_cpus}
    os.sched_setaffinity(0, cpus)
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    child = None if args.pid else subprocess.Popen(command)
    pid = args.pid or child.pid
    while Path('/proc', str(pid)).exists():
        constrain(pid, cpus, args.descendants, gpu_map)
        if child and child.poll() is not None:
            break
        status = Path('/proc', str(pid), 'status')
        try:
            if '\nState:\tZ' in status.read_text():
                break
        except FileNotFoundError:
            break
        time.sleep(2)
    if child:
        raise SystemExit(child.wait())


if __name__ == '__main__':
    main()
