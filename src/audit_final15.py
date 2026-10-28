"""Independent completion audit of training, integer predictions and emulator evidence."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import campaign as c
from verification import deployment_gate


def history(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def audit(state, cfg, bundle=None):
    run = c.ROOT / 'runs' / state['run']
    training = c.read(run / 'protocol.json')
    summary = c.read(run / 'summary.json')
    arch = c.read(c.ROOT / 'runs/nas-tflite-v1' / state['id'] / 'arch.json')
    records = history(run / 'history.jsonl')
    quant = run / state['quant_directory']
    provenance = c.read(quant / 'training-provenance.json')
    qat = run / state['qat_directory']
    qp = c.read(qat / 'protocol.json')
    epoch = provenance['selected_epoch']
    exported = qat / ('epoch-%03d' % epoch)
    predictions = np.load(exported / 'report.predictions.npy')
    labels = np.load(run / 'tflite-data/val-labels.npy')
    measured_top1 = float(np.mean(predictions == labels))
    sdk = run / state['firmware_directory'] / 'sdkconfig'
    sdk_text = sdk.read_text()
    calibration = c.read(run / 'tflite-data/protocol.json')
    splits = c.read(c.ROOT / 'logs/dataset.json')['ids']
    calibration_files = set(calibration['calibration']['image_ids'])
    checks = {
        'accepted_state': state['stage'] == 'accepted',
        'architecture_identity': training['arch'] == arch == state['arch'] and
            hashlib.sha256(json.dumps(arch, sort_keys=True).encode()).hexdigest()[:16] == state['id'],
        'full_training_budget': training['head_epochs'] == 5 and training['finetune_epochs'] == 50 and
            training['max_batches'] is None and len(records) == 55 and
            [r['epoch'] for r in records if r['phase'] == 'head'] == list(range(1, 6)) and
            [r['epoch'] for r in records if r['phase'] == 'full'] == list(range(1, 51)),
        'complete_validation_every_training_epoch': all(r['validation']['count'] == 1020 for r in records),
        'fp32_checkpoint_identity': c.digest(run / 'best.pt') == summary['checkpoint_sha256'],
        'fp32_reference': summary['validation']['count'] == 1020 and summary['validation']['top1'] == state['fp32_top1'],
        'qat_training_budget': qp['epochs'] == 50 and [r['epoch'] for r in history(qat / 'history.jsonl')] == list(range(1, 51)),
        'qat_graph_parity': qp['full_fp32_pair_check']['count'] == 1020 and qp['full_fp32_pair_check']['max_abs_error'] < .002,
        'qat_checkpoint_identity': c.digest(qat / 'best.pt') == provenance['checkpoint_sha256'],
        'integer_predictions': predictions.shape == labels.shape == (1020,) and
            bool(np.all((predictions >= 0) & (predictions < 102))) and abs(measured_top1 - state['int8_top1']) < 1e-12,
        'integer_prediction_model_identity': c.digest(exported / 'model.tflite') == c.digest(quant / 'model.tflite') == state['production_sha256'],
        'accuracy': measured_top1 >= cfg['quality']['validation_top1_target'] and
            summary['validation']['top1'] - measured_top1 <= cfg['quality']['max_quantization_drop'],
        'calibration_train_only': calibration['calibration']['split'] == 'train' and
            calibration['calibration']['count'] == len(calibration_files) == 510 and
            calibration_files <= set(splits['train']) and not calibration_files.intersection(splits['val'] + splits['test']),
        'test_not_used': summary['test_used'] is False and training['test_used'] is False and
            qp['test_used'] is False and calibration['test_used'] is False and state['test_used'] is False,
        'target_sdk_configuration': all(line in sdk_text.splitlines() for line in [
            'CONFIG_ESPTOOLPY_FLASHSIZE="16MB"', 'CONFIG_SPIRAM_MODE_OCT=y',
            'CONFIG_SPIRAM_SPEED=80', 'CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ=240']),
        'not_physical_measurements': state['real_hardware_validated'] is False and state['latency_label_ms'] is None,
    }
    if state['qat_directory'].endswith('round2'):
        first = run / 'qat-tflite-round1'
        checks['first_qat_round_complete'] = c.read(first / 'protocol.json')['epochs'] == 50 and len(history(first / 'history.jsonl')) == 50
    gate = deployment_gate(run)
    checks['fresh_deployment_gate'] = gate['passed']
    if bundle:
        checks['bundle_model_identity'] = c.digest(bundle / 'model.tflite') == state['production_sha256']
        checks['bundle_architecture_identity'] = c.read(bundle / 'architecture.json') == arch
    return {'id': state['id'], 'passed': all(checks.values()), 'checks': checks,
        'deployment_gate': gate, 'fp32_top1': state['fp32_top1'], 'int8_top1': measured_top1,
        'production_sha256': state['production_sha256'], 'sdkconfig_sha256': c.digest(sdk),
        'model_bytes': state['production_model_bytes'], 'psram_bytes': state['psram_peak_bytes'],
        'internal_heap_bytes': state['internal_heap_peak_bytes']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate')
    args = parser.parse_args()
    out = c.ROOT / 'runs/nas-final-v1'
    cfg = c.read(c.ROOT / 'configs/config.json')
    if args.candidate:
        states = [c.read(out / 'states' / (args.candidate + '.json'))]
    else:
        manifest = c.read(out / 'selected-networks.json')
        states = manifest['networks']
        assert manifest['status'] == 'locked' and manifest['count'] == len(states) == 15
        assert len({s['id'] for s in states}) == len({s['production_sha256'] for s in states}) == 15
        assert manifest['config'] == cfg and manifest['test_used_for_selection'] is False
    rows = []
    for index, state in enumerate(states, 1):
        bundle = None if args.candidate else out / 'final15' / ('%02d-%s' % (index, state['id']))
        row = audit(state, cfg, bundle)
        rows.append(row)
        print(json.dumps({'id': row['id'], 'passed': row['passed'],
            'failed_checks': [k for k, v in row['checks'].items() if not v]}), flush=True)
    consistent_sdk = len({r['sdkconfig_sha256'] for r in rows}) == 1
    report = {'passed': all(r['passed'] for r in rows) and consistent_sdk, 'count': len(rows),
        'same_sdk_configuration': consistent_sdk, 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'audit_code_sha256': c.digest(__file__), 'models': rows, 'real_hardware_validated': False}
    filename = ('preflight-audit-' + args.candidate + '.json') if args.candidate else 'final-audit.json'
    c.write(out / filename, report)
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
