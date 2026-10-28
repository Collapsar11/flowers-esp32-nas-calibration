"""Independently audit held-out acquisition and reported errors without refitting."""
import math
import os
from pathlib import Path
import statistics
import time

from measurement_io import ROOT, design, read, read_receipt, sha, write_new
from strict6 import verify_lock

BASE = ROOT / 'results/strict6-physical-query-v1'
LOCK = ROOT / 'results/strict6-physical-v1/calibration/lock.json'
LOCK_SHA = 'f0698cd1fb0cb8623b919c1a413acb9b0190ad258b49c13d16c9c1734f3a9fc3'
BOARD = os.environ['FLOWERS_BOARD_MAC']


def audit():
    frozen = verify_lock(LOCK, False)
    assert sha(LOCK) == LOCK_SHA
    protocol, features = design()
    spec = read(BASE / 'operations/spec.json')
    result = read(BASE / 'evaluation.json')
    assert spec['source_sha256'] == sha(ROOT / 'run_physical_query.py')
    assert spec['prediction_lock_sha256'] == result['prediction_lock_sha256'] == LOCK_SHA
    assert result['source'] == 'physical' and result['simulation_only'] is False
    assert spec['query_ids'] == protocol['query_ids']
    assert spec['acquisition_order'][0] == frozen['decisions']['structure_ridge_primary']['fastest']
    assert sha(ROOT / 'results/nas-final-test-v1/test-metrics.csv') == spec['test_metrics_sha256']
    groups, nonces, provenance = {}, set(), []
    support_nonces = {read(LOCK.parent / p['path'])['nonce'] for p in frozen['support_observations']['receipts']}
    counts = {'memory': 0, 'timing': 0, 'warmup': 0}
    paths = sorted(BASE.glob('*/**/receipt.json'))
    assert len(paths) == 54
    assert not list(BASE.rglob('failure.json')), 'Review and preserve failed attempts separately before audit'
    for path in paths:
        r, m = read_receipt(path)
        ident = m['id']
        assert ident in protocol['query_ids']
        assert r['source'] == 'physical' and r['board_serial'] == BOARD
        assert r['silicon_latency_valid'] is True
        assert r['started_unix'] > frozen['created_unix']
        assert r['prediction_lock_sha256'] == LOCK_SHA
        assert r['capture_source_sha256'] == spec['source_sha256']
        assert r['nonce'] not in nonces | support_nonces
        nonces.add(r['nonce'])
        assert (path.parent / 'serial.log').read_text().count('ESP-ROM:esp32s3') == 1
        assert m['runtime_fingerprint'] == frozen['runtime_fingerprint']
        assert m['production_sha256'] == features[ident]['production_sha256']
        assert r['loaded_image_sha256_declared'] == m['artifact_hashes']['flash_image.bin']
        key = r['kind'], r['session']
        assert key not in groups.setdefault(ident, {})
        groups[ident][key] = r
        counts[r['kind']] += r['parsed']['done']['count']
        counts['warmup'] += r['parsed']['done']['warmup']
        provenance.append({'path': str(path.relative_to(BASE)), 'sha256': sha(path)})
    assert counts == {'memory': 2754, 'timing': 5508, 'warmup': 270}
    observations = {}
    for ident in protocol['query_ids']:
        sessions = groups[ident]
        assert set(sessions) == {(k, s) for k in ['memory', 'timing'] for s in range(3)}
        medians = [statistics.median(sessions['timing', s]['parsed']['duration_ms']) for s in range(3)]
        obs = {'id': ident, 'latency_ms': statistics.median(medians), 'session_medians_ms': medians}
        for name in ['psram_peak_bytes', 'internal_heap_peak_bytes']:
            obs[name] = max(sessions['memory', s]['parsed']['memory'][name] for s in range(3))
        assert obs == next(row for row in result['query_observations']['networks'] if row['id'] == ident)
        observations[ident] = obs
    for method in ['scaled_macs', 'log_macs_ridge', 'structure_ridge_primary']:
        errors = [abs(frozen['predictions'][i][method]['latency_ms'] - observations[i]['latency_ms']) for i in protocol['query_ids']]
        apes = [100 * e / observations[i]['latency_ms'] for i, e in zip(protocol['query_ids'], errors)]
        for key, value in [('mape_percent', statistics.mean(apes)), ('mae_ms', statistics.mean(errors)),
                           ('rmse_ms', math.sqrt(statistics.mean(e * e for e in errors))),
                           ('maximum_ape_percent', max(apes)), ('median_ape_percent', statistics.median(apes))]:
            assert math.isclose(result['metrics'][method][key], value, rel_tol=1e-12, abs_tol=1e-10)
    # Independently check ranking metrics and frozen fastest selections.
    def average_ranks(values):
        return [1 + sum(v < x for v in values) + (sum(v == x for v in values) - 1) / 2 for x in values]
    ids = protocol['query_ids']
    actual = [observations[i]['latency_ms'] for i in ids]
    for method in frozen['decisions']:
        predicted = [features[i]['macs'] if method == 'raw_macs' else frozen['predictions'][i][method]['latency_ms'] for i in ids]
        x, y = average_ranks(predicted), average_ranks(actual)
        mx, my = statistics.mean(x), statistics.mean(y)
        rho = sum((a-mx)*(b-my) for a,b in zip(x,y)) / math.sqrt(sum((a-mx)**2 for a in x)*sum((b-my)**2 for b in y))
        assert math.isclose(rho, result['metrics'][method]['spearman'], abs_tol=1e-12)
        concordant = discordant = tied_pred = tied_actual = 0
        for a in range(len(ids)):
            for b in range(a):
                dx, dy = predicted[a]-predicted[b], actual[a]-actual[b]
                if dx*dy > 0: concordant += 1
                elif dx*dy < 0: discordant += 1
                elif dx == 0 and dy != 0: tied_pred += 1
                elif dx != 0 and dy == 0: tied_actual += 1
        tau = (concordant-discordant)/math.sqrt((concordant+discordant+tied_pred)*(concordant+discordant+tied_actual))
        agreement = (concordant+.5*tied_pred)/(concordant+discordant+tied_pred)
        assert math.isclose(tau, result['metrics'][method]['kendall_tau_b'], abs_tol=1e-12)
        assert math.isclose(agreement, result['metrics'][method]['pairwise_ordering_agreement'], abs_tol=1e-12)
        selected = frozen['decisions'][method]['fastest']
        assert result['decisions'][method]['fastest']['chosen'] == selected
        for before, after in zip(frozen['decisions'][method]['accuracy_under_budgets'], result['decisions'][method]['accuracy_under_budgets']):
            assert (after.get('chosen') if before['applicable'] else after['id']) == before['id']
        checks = [(None, result['decisions'][method]['fastest'])]
        if method != 'raw_macs':
            checks += [(entry['budget_ms'], entry) for entry in result['decisions'][method]['accuracy_under_budgets']]
        for budget, score in checks:
            pool = [i for i in ids if features[i]['validation_top1'] >= .90
                    and features[i]['model_bytes'] <= 12*2**20
                    and observations[i]['psram_peak_bytes'] <= 6*2**20
                    and observations[i]['internal_heap_peak_bytes'] <= 192*2**10
                    and (budget is None or observations[i]['latency_ms'] <= budget)]
            if not pool:
                assert score['status'] == 'no_feasible_oracle'
                continue
            key = (lambda i: (observations[i]['latency_ms'], i)) if budget is None else (lambda i: (-features[i]['validation_top1'], observations[i]['latency_ms'], i))
            oracle = min(pool, key=key)
            assert score['oracle'] == oracle
            chosen = score['chosen']
            if chosen is None: assert score['status'] == 'no_selection'
            elif chosen not in pool: assert score['status'] == 'budget_violation'
            else:
                regret = (observations[chosen]['latency_ms']/observations[oracle]['latency_ms']-1) if budget is None else 100*(features[oracle]['validation_top1']-features[chosen]['validation_top1'])
                assert score['status'] == 'feasible' and score['selection_agreement'] == (chosen == oracle)
                assert math.isclose(score['regret'], regret, abs_tol=1e-12)
    for name, limit in [('psram_peak_bytes', 6*2**20), ('internal_heap_peak_bytes', 192*2**10)]:
        for label, method in [('qemu_baseline', 'raw_macs'), ('six_label_residual_correction', 'structure_ridge_primary')]:
            predicted = [frozen['predictions'][i][method][name] for i in ids]
            actual_memory = [observations[i][name] for i in ids]
            differences = [p-y for p,y in zip(predicted, actual_memory)]
            metrics = result['memory'][name][label]
            assert metrics['mae_bytes'] == statistics.mean(abs(d) for d in differences)
            assert metrics['maximum_absolute_error_bytes'] == max(abs(d) for d in differences)
            assert metrics['maximum_underprediction_bytes'] == max(0, -min(differences))
            categories = {'true_feasible': 0, 'false_feasible': 0, 'false_infeasible': 0, 'true_infeasible': 0}
            for p, y in zip(predicted, actual_memory):
                category = ('true_feasible' if y <= limit else 'false_feasible') if p <= limit else ('false_infeasible' if y <= limit else 'true_infeasible')
                categories[category] += 1
            assert categories == metrics['feasibility_counts']
    flash_seconds = 0
    for ident in ids:
        directory = BASE / 'operations' / ident
        r = read(directory / 'flash.json')
        log = (directory / 'flash.log').read_text()
        assert r['exit_code'] == 0 and r['board_mac'] == BOARD
        assert sha(directory / 'flash.log') == r['log_sha256']
        assert 'MAC: ' + BOARD in log and 'Hash of data verified.' in log
        image = Path(r['command'][-1])
        assert image.parent.name == ident
        assert sha(image) == read(image.parent / 'manifest.json')['artifact_hashes']['flash_image.bin']
        assert r['ended_unix'] < min(v['started_unix'] for v in groups[ident].values())
        flash_seconds += r['wall_seconds']
    capture_seconds = sum(r['ended_unix']-r['started_unix'] for group in groups.values() for r in group.values())
    selected = frozen['decisions']['structure_ridge_primary']['fastest']
    check = read(BASE / 'selected-model-check.json')
    assert check['id'] == selected and check['observed'] == observations[selected]
    assert check['selection_unchanged'] and check['prediction_lock_sha256'] == LOCK_SHA
    backup = read(ROOT / 'results/esp32-first-board-v1/original-backup.json')
    assert sha(ROOT / 'results/esp32-first-board-v1' / backup['file']) == backup['sha256']
    return {'passed': True, 'audited_unix': time.time(), 'source': 'physical',
            'networks': 9, 'sessions': 54, 'invocations': counts, 'integer_mismatches': 0,
            'prediction_lock_unchanged': True, 'prediction_lock_sha256': LOCK_SHA,
            'no_refitting': True, 'selected_id_unchanged': selected, 'observations': observations,
            'audit_source_sha256': sha(Path(__file__)), 'evaluation_sha256': sha(BASE / 'evaluation.json'),
            'cost': {'query_flash_wall_seconds': flash_seconds, 'query_valid_capture_wall_seconds': capture_seconds,
                     'cost_scope': 'Held-out evaluation overhead; not calibration or selection labels'},
            'original_backup_preserved': True, 'receipts': provenance}


if __name__ == '__main__':
    result = audit()
    write_new(BASE / 'audit.json', result)
    print({'passed': result['passed'], 'sessions': result['sessions']})
