"""Compact, read-only progress; incomplete models never count as accepted."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import argparse
import time

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def collect():
    campaign = ROOT / 'runs/nas-final-v1'
    states = [read(path) for path in sorted((campaign / 'states').glob('*.json'))]
    full, qat, errors = [], [], []
    for state in states:
        run = ROOT / 'runs' / state['run']
        summary = read(run / 'summary.json')
        if summary:
            full.append(summary['validation']['top1'])
        if state['stage'].startswith('qat_round_'):
            index = state['stage'].rsplit('_', 1)[-1]
            history = run / ('qat-tflite-round' + index) / 'history.jsonl'
            rows = history.read_text().splitlines() if history.exists() else []
            latest = {}
            for row in reversed(rows):
                try:
                    latest = json.loads(row)
                    break
                except json.JSONDecodeError:
                    continue
            qat.append({'id': state['id'], 'round': int(index),
                'epoch': latest.get('epoch', 0), 'fake_top1': latest.get('fake_quant_top1')})
        if state['stage'] in {'failed_execution', 'rejected_quality'}:
            errors.append({k: state.get(k) for k in ['id', 'stage', 'error', 'reason']})
    accepted = [s for s in states if s['stage'] == 'accepted']
    selection = read(campaign / 'selected-networks.json')
    return {'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'search': read(ROOT / 'runs/nas-tflite-v1/stage-status.json'),
        'controller_status': read(campaign / 'status.json'),
        'stages': dict(Counter(s['stage'] for s in states)), 'full_training_complete': len(full),
        'fp32_range': [min(full), max(full)] if full else None, 'qat_progress': qat,
        'accepted': len(accepted),
        'accepted_int8_range': [min(s['int8_top1'] for s in accepted), max(s['int8_top1'] for s in accepted)] if accepted else None,
        'accepted_resource_ranges': {key: [min(s[key] for s in accepted), max(s[key] for s in accepted)]
            for key in ['macs', 'production_model_bytes', 'psram_peak_bytes', 'internal_heap_peak_bytes', 'quantization_drop']} if accepted else {},
        'errors': errors, 'selected_count': selection['count'] if selection else 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    while True:
        result = collect()
        if args.watch:
            epochs = [r['epoch'] for r in result['qat_progress']]
            compact = {k: result[k] for k in ['timestamp_utc', 'stages', 'full_training_complete',
                'fp32_range', 'accepted', 'accepted_int8_range', 'errors', 'selected_count']}
            compact['qat_epoch_range'] = [min(epochs), max(epochs)] if epochs else None
            compact['controller_stage'] = result['controller_status']['stage']
            print(json.dumps(compact), flush=True)
        else:
            print(json.dumps(result, indent=2))
        if not args.watch or result['selected_count'] == 15:
            break
        time.sleep(30)
