"""Evaluate the tail of the fixed shortlist using the same candidate locks/policy."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import json
import queue
import time
import campaign as c


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--search', default='nas-tflite-v1')
    parser.add_argument('--run', default='nas-final-v1')
    parser.add_argument('--gpus', type=int, nargs='+', default=[7, 2, 5])
    parser.add_argument('--count', type=int, default=9)
    parser.add_argument('--pending', action='store_true', help='Claim only candidates without a state file, in reverse shortlist order')
    parser.add_argument('--label', default='extra-workers')
    parser.add_argument('--cpu-start', type=int, default=88)
    args = parser.parse_args()
    out = c.ROOT / 'runs' / args.run
    assert args.label.replace('-', '').isalnum()
    with (out / (args.label + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        records = c.read(c.ROOT / 'runs' / args.search / 'full-training-shortlist.json')
        if args.pending:
            records = [r for r in reversed(records) if not (out / 'states' / (r['id'] + '.json')).exists()][:args.count]
        else:
            records = records[-args.count:]
        assert len({r['id'] for r in records}) == len(records) <= args.count
        c.write(out / (args.label + '.json'), {'code_sha256': c.digest(__file__),
            'candidate_ids': [r['id'] for r in records], 'gpus': args.gpus,
            'started_unix': time.time(), 'policy': 'same frozen shortlist and candidate function; no additional architectures or changed training budgets'})
        slots = queue.Queue()
        for index, gpu in enumerate(args.gpus):
            slots.put((gpu, list(range(args.cpu_start + 4*index, args.cpu_start + 4*index + 4))))
        def work(record):
            gpu, cpus = slots.get()
            try:
                return c.candidate(args, record, gpu, cpus)
            finally:
                slots.put((gpu, cpus))
        results = []
        with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
            for future in as_completed([pool.submit(work, record) for record in records]):
                results.append(future.result())
                c.write(out / (args.label + '-results.json'), results)
        print(json.dumps({'finished': len(results)}), flush=True)


if __name__ == '__main__':
    main()
