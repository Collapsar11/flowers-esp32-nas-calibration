"""Read-only independent audit of six physical labels and the nine prediction lock.

Creates a new audit artifact only after every check passes. No device access,
training changes, held-out measurements, or calls to held-out evaluation.
"""
import argparse
import math
import os
from pathlib import Path
import statistics
import time
import numpy as np
from measurement_io import ROOT, design, read, read_receipt, sha, write_new
from strict6 import verify_lock

BASE = ROOT / 'results/strict6-physical-v1'
BOARD = os.environ['FLOWERS_BOARD_MAC']


def independent_ridge(rows, labels, dimensions, queries):
    def vector(row):
        return [math.log(row['macs']), math.log(row['activation_io_bytes']),
                row['depthwise_mac_fraction']][:dimensions]
    x = np.array([vector(r) for r in rows])
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale == 0] = 1
    z = (x - mean) / scale
    y = np.log(labels)
    # Augmented least squares is independent of the production normal-equation solve.
    a = np.vstack([z, np.eye(dimensions)])
    b = np.concatenate([y - y.mean(), np.zeros(dimensions)])
    w = np.linalg.lstsq(a, b, rcond=None)[0]
    return {r['id']: float(np.exp(y.mean() + ((vector(r) - mean) / scale) @ w))
            for r in queries}


def audit():
    protocol, features = design()
    lock_path = BASE / 'calibration/lock.json'
    frozen = verify_lock(lock_path, False)
    spec = read(BASE / 'operations/spec.json')
    assert spec['source_sha256'] == sha(ROOT / 'run_physical_support.py')
    assert spec['support_ids'] == protocol['support_ids']
    assert spec['query_ids_prohibited'] == protocol['query_ids']
    for name, digest in spec['analysis_source_sha256'].items():
        assert sha(ROOT / name) == digest
    assert frozen['source'] == 'physical' and frozen['physical_calibration_network_count'] == 6
    assert frozen['query_labels_accessed'] is False
    assert set(frozen['predictions']) == set(protocol['query_ids'])
    for ident in protocol['query_ids']:
        assert not (BASE / ident).exists(), 'query acquisition directory must be absent'
    assert not list(BASE.rglob('failure.json')), 'failed attempts require separate explicit review'
    paths = sorted(BASE.glob('*/**/receipt.json'))
    assert len(paths) == 36
    nonces, runtime, boot_methods = set(), set(), {}
    groups, provenance, total = {}, [], {'memory': 0, 'timing': 0, 'warmup': 0}
    for path in paths:
        receipt, manifest = read_receipt(path)
        ident = manifest['id']
        assert ident in protocol['support_ids'] and receipt['source'] == 'physical'
        assert receipt['board_serial'] == BOARD and receipt['silicon_latency_valid'] is True
        assert receipt['ended_unix'] < frozen['created_unix']
        assert receipt['nonce'] not in nonces
        nonces.add(receipt['nonce']); runtime.add(manifest['runtime_fingerprint'])
        text = (path.parent / 'serial.log').read_text()
        assert text.count('ESP-ROM:esp32s3') == 1
        assert receipt['loaded_image_sha256_declared'] == manifest['artifact_hashes']['flash_image.bin']
        assert manifest['production_sha256'] == features[ident]['production_sha256']
        kind, session = receipt['kind'], receipt['session']
        assert (kind, session) not in groups.setdefault(ident, {})
        groups[ident][kind, session] = receipt
        done = receipt['parsed']['done']
        assert done['integer_mismatches'] == 0
        total[kind] += done['count']; total['warmup'] += done['warmup']
        method = receipt.get('reset_method', 'manual EN/RESET (original pilot)')
        if 'reset_method' in receipt:
            assert receipt['capture_source_sha256'] == spec['source_sha256']
        boot_methods[method] = boot_methods.get(method, 0) + 1
        provenance.append({'path': str(path.relative_to(BASE)), 'sha256': sha(path)})
    assert len(runtime) == 1 and runtime == {frozen['runtime_fingerprint']}
    assert total == {'memory': 1836, 'timing': 3672, 'warmup': 180}
    observed = []
    for ident in protocol['support_ids']:
        sessions = groups[ident]
        assert set(sessions) == {(k, s) for k in ['memory', 'timing'] for s in range(3)}
        medians = [statistics.median(sessions['timing', s]['parsed']['duration_ms']) for s in range(3)]
        row = {'id': ident, 'latency_ms': statistics.median(medians), 'session_medians_ms': medians}
        for name in ['psram_peak_bytes', 'internal_heap_peak_bytes']:
            row[name] = max(sessions['memory', s]['parsed']['memory'][name] for s in range(3))
        assert row == next(r for r in frozen['support_observations']['networks'] if r['id'] == ident)
        assert features[ident]['model_bytes'] <= 12 * 2**20
        assert features[ident]['validation_top1'] >= .90
        assert row['psram_peak_bytes'] <= 6 * 2**20
        assert row['internal_heap_peak_bytes'] <= 192 * 2**10
        observed.append(row)
    x = [features[r['id']] for r in observed]
    y = [r['latency_ms'] for r in observed]
    queries = [features[i] for i in protocol['query_ids']]
    independent = {name: independent_ridge(x, y, d, queries) for name, d in
                   [('log_macs_ridge', 1), ('structure_ridge_primary', 3)]}
    proportional = statistics.median(t / r['macs'] for t, r in zip(y, x))
    offsets = {k: statistics.median(r[k] - features[r['id']]['qemu_' + k] for r in observed)
               for k in ['psram_peak_bytes', 'internal_heap_peak_bytes']}
    assert offsets == frozen['memory_offsets_bytes']
    assert proportional == frozen['scaled_macs_ms_per_mac']
    for ident, entry in frozen['predictions'].items():
        assert entry['raw_macs']['latency_ms'] is None
        assert math.isclose(entry['scaled_macs']['latency_ms'], proportional * features[ident]['macs'], rel_tol=1e-12)
        for name in independent:
            assert math.isclose(entry[name]['latency_ms'], independent[name][ident], rel_tol=1e-12)
        for method, prediction in entry.items():
            for k in offsets:
                expect = max(0, features[ident]['qemu_' + k] + (0 if method == 'raw_macs' else offsets[k]))
                assert prediction[k] == expect
    # Independently reconstruct decisions with no query ground truth.
    for method, decision in frozen['decisions'].items():
        for budget, selected in [(None, decision['fastest'])] + [
                (r['budget_ms'], r['id']) for r in decision['accuracy_under_budgets']]:
            if method == 'raw_macs' and budget is not None:
                assert selected is None
                continue
            pool = []
            for ident in protocol['query_ids']:
                f, p = features[ident], frozen['predictions'][ident][method]
                if (f['validation_top1'] >= .90 and f['model_bytes'] <= 12 * 2**20
                    and p['psram_peak_bytes'] <= 6 * 2**20 and p['internal_heap_peak_bytes'] <= 192 * 2**10
                    and (budget is None or p['latency_ms'] <= budget)):
                    pool.append(ident)
            def cost(i):
                return features[i]['macs'] if method == 'raw_macs' else frozen['predictions'][i][method]['latency_ms']
            key = (lambda i: (cost(i), i)) if budget is None else (lambda i: (-features[i]['validation_top1'], cost(i), i))
            assert selected == (min(pool, key=key) if pool else None)
    # Verify board identity and success for each authorized flash operation.
    flash_seconds = read(ROOT / 'results/esp32-first-board-v1/flash-result.json')['load_seconds']
    flash_records = []
    for ident in protocol['support_ids'][1:]:
        directory = BASE / 'operations' / ident
        record = read(directory / 'flash.json')
        log = (directory / 'flash.log').read_text()
        assert record['exit_code'] == 0 and record['board_mac'] == BOARD
        assert record['log_sha256'] == sha(directory / 'flash.log')
        assert 'MAC: ' + BOARD in log and 'Hash of data verified.' in log
        image = Path(record['command'][-1])
        assert image.parent.name == ident
        assert sha(image) == read(image.parent / 'manifest.json')['artifact_hashes']['flash_image.bin']
        flash_seconds += record['wall_seconds']; flash_records.append(record)
    backup = read(ROOT / 'results/esp32-first-board-v1/original-backup.json')
    assert backup['verified_against_device']
    assert sha(ROOT / 'results/esp32-first-board-v1' / backup['file']) == backup['sha256']
    capture_wall = sum(r['ended_unix'] - r['started_unix'] for group in groups.values() for r in group.values())
    interrupted, interrupted_wall = [], 0.0
    for recovery_path in sorted((BASE / 'operations').glob('recovery-*.json')):
        recovery = read(recovery_path)
        assert recovery['protocol_unchanged'] is True
        assert recovery['capture_source_sha256'] == spec['source_sha256']
        assert recovery['source_sha256'] == sha(ROOT / 'resume_physical_support.py')
        for attempt in recovery['interrupted_attempts']:
            assert attempt['id'] in protocol['support_ids']
            assert attempt['included_in_calibration'] is False
            archive = ROOT / attempt['archive']
            assert not (archive / 'receipt.json').exists()
            for name, digest in attempt['files_sha256'].items():
                assert sha(archive / name) == digest
            failure = read(archive / 'failure.json')
            assert failure == attempt['failure'] and failure['nonce'] not in nonces
            interrupted_wall += failure['ended_unix'] - failure['started_unix']
            interrupted.append(attempt)
    return {'passed': True, 'audited_unix': time.time(), 'source': 'physical', 'simulation_only': False,
            'board_mac': BOARD, 'support_networks': 6, 'physical_query_networks': 0,
            'sessions': 36, 'memory_sessions': 18, 'timing_sessions': 18,
            'invocations': total, 'integer_output_mismatches': 0,
            'reset_methods': boot_methods, 'runtime_fingerprint': next(iter(runtime)),
            'independent_aggregation_regression_and_selection_verified': True,
            'lock_sha256': sha(lock_path), 'audit_source_sha256': sha(Path(__file__)),
            'support_observations': observed, 'memory_offsets_bytes': offsets,
            'predicted_query_networks': 9, 'query_labels_accessed': False,
            'heldout_accuracy_metrics': None,
            'interrupted_attempts_excluded_and_preserved': interrupted,
            'cost': {'measured_network_count_reduction': .60, 'flash_wall_seconds_six_networks': flash_seconds,
                     'capture_wall_seconds_36_sessions': capture_wall,
                     'interrupted_capture_wall_seconds': interrupted_wall,
                     'fit_and_selection_wall_seconds': frozen['fit_and_selection_wall_seconds'],
                     'measured_wall_time_saving': None,
                     'note': 'Capture wall includes host/reset/log overhead. 60% is network count reduction only; no nine-query measurements or time-saving counterfactual.'},
            'original_backup_preserved': True, 'receipts': provenance,
            'interpretation': 'Six physical calibration labels; predictions and decisions locked for nine unmeasured query networks. No held-out error/ranking/regret claim.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default=str(BASE / 'audit.json'))
    args = parser.parse_args()
    result = audit()
    write_new(args.out, result)
    print({'passed': result['passed'], 'lock_sha256': result['lock_sha256'], 'audit': args.out})
