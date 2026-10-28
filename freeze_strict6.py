"""Freeze a label-free six-support/nine-query design. No board access."""
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/strict6-esp32-v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = OUT / 'public-features.json'
    public = json.loads(source.read_text())
    rows = sorted(public['networks'], key=lambda x: x['id'])
    assert len(rows) == len({r['id'] for r in rows}) == 15
    features = [
        'log_macs', 'log_activation_io_bytes', 'depthwise_mac_fraction',
        'log_model_bytes', 'log_qemu_psram_peak_bytes',
    ]
    vectors = [[math.log(r['macs']), math.log(r['activation_io_bytes']),
                r['depthwise_mac_fraction'], math.log(r['model_bytes']),
                math.log(r['qemu_psram_peak_bytes'])] for r in rows]
    low = [min(v[j] for v in vectors) for j in range(5)]
    span = [max(v[j] for v in vectors) - low[j] or 1.0 for j in range(5)]
    normalized = [[(v[j]-low[j])/span[j] for j in range(5)] for v in vectors]
    chosen = [min(range(15), key=lambda i: (rows[i]['macs'], rows[i]['id']))]
    chosen += [min(range(15), key=lambda i: (-rows[i]['macs'], rows[i]['id']))]
    trace = [{'id': rows[i]['id'], 'reason': why} for i, why in zip(chosen, ['minimum MACs', 'maximum MACs'])]
    while len(chosen) < 6:
        distance = {i: min(sum((normalized[i][j]-normalized[k][j])**2 for j in range(5))
                           for k in chosen) for i in range(15) if i not in chosen}
        nxt = min(distance, key=lambda i: (-distance[i], rows[i]['id']))
        chosen.append(nxt)
        trace.append({'id': rows[nxt]['id'], 'minimum_squared_distance': distance[nxt]})
    support = [rows[i]['id'] for i in chosen]
    query = [r['id'] for r in public['networks'] if r['id'] not in support]
    result = {
        'protocol_version': 'strict6-esp32-v1',
        'frozen_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'design_frozen_no_physical_measurements',
        'physical_board_accessed': False,
        'selection_sha256': public['selection_sha256'],
        'public_features_sha256': sha(source), 'selector_code_sha256': sha(Path(__file__)),
        'support_count': 6, 'query_count': 9, 'support_ids': support, 'query_ids': query,
        'selection': {
            'algorithm': 'minimum and maximum MACs, then greedy max-min squared Euclidean distance; lexical ID tie break',
            'features': features, 'normalization': 'min-max over public features of the locked 15',
            'low': low, 'span': span, 'trace': trace,
            'image_test_results_used': False, 'hardware_labels_used': False,
            'validation_accuracy_used_for_support_selection': False,
        },
        'target': {
            'chip': 'ESP32-S3-N16R8', 'board_configuration': 'bread-compact-wifi-mpu6050',
            'cpu_mhz': 240, 'psram_mhz': 80, 'psram_mode': 'Octal SPI',
            'flash_bytes': 16*2**20, 'psram_bytes': 8*2**20,
            'runtime': 'TFLite Micro + ESP-NN; Flash weights; PSRAM tensor arena',
            'esp_idf_commit': 'fcae32885b0296b32044cb99ecbdc50d98dddb83',
            'scope': 'standalone inference harness; full board application integration unverified',
        },
        'unchanged_gates': {'int8_validation_top1_min': .90, 'validation_quantization_drop_max': .01,
            'production_model_bytes_max': 12*2**20, 'model_related_psram_peak_bytes_max': 6*2**20,
            'incremental_internal_heap_peak_bytes_max': 192*2**10,
            'note': 'internal heap is incremental, not total SRAM; image test results never reselect the pool'},
        'latency_models': {
            'raw_macs': {'labels': 0, 'output': 'ranking only; no millisecond error metrics'},
            'scaled_macs': {'labels': 6, 'formula': 'MACs * median(support_latency_ms / support_MACs)'},
            'log_macs_ridge': {'labels': 6, 'features': ['log_macs'], 'target': 'log_latency_ms', 'alpha': 1.0},
            'structure_ridge_primary': {'labels': 6,
                'features': ['log_macs', 'log_activation_io_bytes', 'depthwise_mac_fraction'],
                'target': 'log_latency_ms', 'alpha': 1.0},
            'ridge_specification': {
                'scaler': 'mean and population standard deviation fitted on six support rows only; zero std -> 1',
                'objective': 'sum_i (log(y_i)-b-x_i*w)^2 + alpha*sum_j w_j^2; intercept unpenalized',
                'inverse_transform': 'exp(b+x*w), no post-hoc bias correction',
                'hyperparameter_search': False,
                'support_diagnostic': 'leave-one-out with scaler refit inside each fold; not held-out-nine evidence',
            },
        },
        'memory_and_storage': {
            'psram_baseline': 'existing QEMU model-related peak bytes',
            'psram_calibration': 'max(0, QEMU_peak + median(real_peak-QEMU_peak on same six supports))',
            'internal_heap_baseline': 'existing QEMU incremental internal heap peak bytes',
            'internal_heap_calibration': 'same median residual offset, clipped at zero',
            'storage': 'exact production .tflite file bytes; no learned size predictor',
            'firmware_flash': 'report linked production image and partitions separately; exclude diagnostic golden payloads',
            'additional_system_memory': 'record static DRAM/IRAM, reserved stacks and high-water marks, baseline/minimum free heap and largest blocks separately',
            'uncertainty': 'no claimed confidence bound from six architectures; point estimates and support residuals only',
        },
        'future_measurement': {
            'execution_authorized_now': False,
            'input': 'same existing 102 class-stratified validation inputs, batch 1, 160x160, integer input',
            'sessions_per_network': 3, 'warmup_invokes_per_session': 10,
            'timed_passes_per_session': 2, 'timed_inputs_per_pass': 102,
            'timed_invokes_per_network': 612, 'session_definition': 'independent cold boot',
            'input_order': 'seed 20261028 + session_index (0,1,2); one permutation per session repeated twice, identical across networks',
            'timer': 'esp_timer_get_time around Invoke only; exclude preprocessing, input copies and serial output',
            'latency_label': 'median of three per-session median latency_ms',
            'also_report': ['per-session mean/p50/p95', 'all invocation durations', 'inter-session spread'],
            'execution': 'fixed CPU 240 MHz and PSRAM 80 MHz; pin inference to core 1; disable DFS/light sleep/Wi-Fi/Bluetooth; other core idle except required OS tasks',
            'memory_pass': 'separate from timed pass; local minimum-free heap monitor spans arena allocation, AllocateTensors and Invoke; log after interval',
            'arena': 'direct allocation of frozen arena_used_bytes rounded to 16-byte alignment plus 4096-byte headroom; exclude old 6 MiB probe from production measurement',
            'memory_label': 'maximum model-related peak delta over three independent memory sessions; SPIRAM and INTERNAL|8BIT separately',
            'required_before_board': 'implement timing harness and revalidate all 15 in host/QEMU with unchanged model hashes; pin firmware/model/sdkconfig/compiler/TFLM/ESP-NN hashes and clock/cache settings',
            'failure': 'retain failed network ID and record failure/infeasibility; no silent replacement, imputed labels, or query measurement for training',
        },
        'lock_sequence': [
            'Current: freeze fifteen model identities, label-free six/nine split and analysis protocol; no physical board access.',
            'Future after board-stage authorization: collect real labels for exactly the six support architectures.',
            'Fit fixed models; save scaler/coefficients, all nine predictions, support-only derived budgets and all decisions with SHA256 and timestamp.',
            'Only after prediction lock, optionally measure nine query networks as independent evaluation; never refit or select methods using their labels.',
        ],
        'evaluation': {
            'latency_primary': 'MAPE over nine query architectures; each architecture equal weight',
            'latency_secondary': ['MAE ms', 'RMSE ms', 'median and maximum absolute percentage error',
                'Spearman rank correlation with average ranks', 'Kendall tau-b', 'pairwise ordering agreement; predicted ties receive 0.5, oracle ties excluded'],
            'ram': ['MAE bytes', 'maximum absolute error bytes', 'maximum underprediction bytes', 'feasibility confusion counts'],
            'zero_memory_labels': 'percentage errors undefined for zero; report NA, never epsilon-denominator MAPE',
            'decision_pool': 'nine held-out networks; original INT8 validation accuracy and unchanged memory/storage gates',
            'primary_decision': 'predict fastest feasible query network; ID breaks equal predicted latencies',
            'latency_regret': '(actual chosen latency - actual oracle minimum feasible latency) / oracle minimum feasible latency',
            'secondary_decision': 'maximize original INT8 validation accuracy under latency budgets; ties: lower predicted latency then lexical ID',
            'latency_budgets': '0.75, 1.0, 1.25 times median of six support latency labels; freeze before any query reveal; sensitivity analysis, not application SLA',
            'accuracy_regret': 'oracle maximum feasible validation accuracy minus chosen validation accuracy, in percentage points',
            'invalid_decision': 'chosen infeasible -> failure and budget violation, regret NA; no feasible oracle or no selected model -> explicitly report NA/no selection',
            'oracle': 'all measured query labels, evaluation only; never a six-label deployable method',
            'six_only_limit': 'with only six real measurements there is no measured nine-query error, rank correlation or regret; show unverified predictions and support LOOCV only',
            'budget_claim': 'calibration/decision uses 6 rather than 15 distinct networks: 60% fewer; a complete independent 9-query audit makes total experimental measurements 15, not 6',
            'honesty': 'no guaranteed improvement; report if same-budget MAC-only baselines match or beat structure Ridge',
        },
        'interpretation': 'few-shot device-specific calibration, not meta-learning from six samples; existing MCU NAS already includes memory/runtime constraints',
        'sources': [
            'https://arxiv.org/abs/2007.10319',
            'https://papers.nips.cc/paper_files/paper/2021/hash/e3251075554389fe91d17a794861d47b-Abstract.html',
            'https://docs.espressif.com/projects/esp-idf/en/v5.4.2/esp32s3/api-reference/system/esp_timer.html',
        ],
    }
    target = OUT / 'protocol.json'
    assert not target.exists(), 'Refuse to overwrite frozen protocol'
    target.write_text(json.dumps(result, indent=2) + '\n')
    template = {'status': 'unmeasured', 'protocol_sha256': sha(target),
                'support': [{'id': ident, 'latency_ms': None, 'psram_peak_bytes': None,
                             'internal_heap_peak_bytes': None, 'sessions': []} for ident in support],
                'query_hardware_labels': 'intentionally absent until predictions and decisions are locked'}
    (OUT/'support-measurements.template.json').write_text(json.dumps(template, indent=2)+'\n')
    print(json.dumps({'support': support, 'query': query, 'protocol_sha256': sha(target)},indent=2))


if __name__ == '__main__':
    main()
