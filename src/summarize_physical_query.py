"""Produce a read-only-data summary after the held-out physical audit passes."""
import csv
from pathlib import Path

from measurement_io import ROOT, design, read, sha, write_new
from strict6 import verify_lock

BASE = ROOT / 'results/strict6-physical-query-v1'
LOCK = ROOT / 'results/strict6-physical-v1/calibration/lock.json'


def main():
    audit = read(BASE / 'audit.json')
    result = read(BASE / 'evaluation.json')
    frozen = verify_lock(LOCK, False)
    assert audit['passed'] and audit['evaluation_sha256'] == sha(BASE / 'evaluation.json')
    assert audit['prediction_lock_sha256'] == sha(LOCK)
    protocol, features = design()
    with (ROOT / 'results/nas-final-test-v1/test-metrics.csv').open() as stream:
        test = {r['id']: r for r in csv.DictReader(stream)}
    rows = []
    for ident in protocol['query_ids']:
        observed = audit['observations'][ident]
        primary = frozen['predictions'][ident]['structure_ridge_primary']
        rows.append({'id': ident, 'selected': ident == audit['selected_id_unchanged'],
                     'observed_latency_ms': observed['latency_ms'],
                     'primary_prediction_ms': primary['latency_ms'],
                     'primary_absolute_percentage_error': 100*abs(primary['latency_ms']-observed['latency_ms'])/observed['latency_ms'],
                     'observed_psram_peak_bytes': observed['psram_peak_bytes'],
                     'predicted_psram_peak_bytes': primary['psram_peak_bytes'],
                     'observed_internal_heap_increment_bytes': observed['internal_heap_peak_bytes'],
                     'model_file_bytes': features[ident]['model_bytes'],
                     'validation_top1_percent': 100*features[ident]['validation_top1'],
                     'test_top1_percent': 100*float(test[ident]['int8_test_top1'])})
    support = read(ROOT / 'results/strict6-physical-v1/audit.json')['cost']
    query = audit['cost']
    support_components = support['flash_wall_seconds_six_networks'] + support['capture_wall_seconds_36_sessions']
    query_components = query['query_flash_wall_seconds'] + query['query_valid_capture_wall_seconds']
    selected = audit['selected_id_unchanged']
    selected_capture = sum(read(p)['ended_unix']-read(p)['started_unix'] for p in (BASE / selected).glob('*/receipt.json'))
    selected_flash = read(BASE / 'operations' / selected / 'flash.json')['wall_seconds']
    summary = {'source': 'physical', 'simulation_only': False, 'audit_passed': True,
               'prediction_lock_sha256': sha(LOCK), 'calibration_networks': 6, 'heldout_networks': 9,
               'query_labels_used_for_fitting': 0, 'query_labels_used_for_reselection': 0,
               'selected_model': read(BASE / 'selected-model-check.json'), 'networks': rows,
               'latency_metrics': result['metrics'], 'memory_metrics': result['memory'],
               'selection_evaluation': result['decisions'],
               'cost': {'support_flash_and_valid_capture_seconds': support_components,
                        'query_evaluation_flash_and_valid_capture_seconds': query_components,
                        'all15_flash_and_valid_capture_seconds': support_components+query_components,
                        'selected_acceptance_flash_and_valid_capture_seconds': selected_flash+selected_capture,
                        'six_label_fit_and_selection_seconds': frozen['fit_and_selection_wall_seconds'],
                        'support_interrupted_attempt_seconds': support['interrupted_capture_wall_seconds'],
                        'measurement_count_reduction_for_decision': .60,
                        'selected_acceptance_in_addition_to_six': 1,
                        'nine_query_measurements_are_offline_evaluation_cost': True,
                        'end_to_end_wall_time_saving': None,
                        'scope_note': 'Component accounting only; shared setup, firmware builds, downloads, backup, identify/verify and waits are not included. Do not call this an end-to-end speedup.'},
               'limits': ['Single board, fixed runtime and clock configuration, nine held-out architectures.',
                          'Invoke latency excludes input copying, reference comparison and serial logging.',
                          'Internal heap is incremental allocation, not total SRAM; no application integration.',
                          'Golden-output agreement is functional correctness, not full-test classification accuracy.',
                          'No application latency SLA has been specified; frozen budgets are sensitivity analyses.',
                          'Test accuracy does not drive selection; the chosen model is below 90% test Top-1.'],
               'evidence_hashes': {n: sha(BASE / n) for n in ['audit.json', 'evaluation.json', 'selected-model-check.json']}}
    write_new(BASE / 'summary.json', summary)
    lines = ['\n## 九网络实板评估最终结果（已完成）\n',
             '固定 6 个实板标签的预测锁保持不变。9 个查询网络完成 54 个独立启动，原始日志及独立审计均通过；查询标签只用于评估，没有重新拟合或重新选型。\n',
             '| 查询网络 | 主方法预测 ms | 实测 ms | 绝对百分比误差 | 实测 PSRAM bytes | 测试 Top-1 |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in rows:
        lines.append(f"| `{r['id']}`{'（已选）' if r['selected'] else ''} | {r['primary_prediction_ms']:.3f} | {r['observed_latency_ms']:.3f} | {r['primary_absolute_percentage_error']:.3f}% | {r['observed_psram_peak_bytes']:,} | {r['test_top1_percent']:.4f}% |")
    lines += ['\n| 方法 | MAPE | MAE ms | Spearman | Kendall tau-b |', '| --- | ---: | ---: | ---: | ---: |']
    for method, m in result['metrics'].items():
        mape = f"{m['mape_percent']:.4f}%" if 'mape_percent' in m else '不适用（仅排序）'
        mae = f"{m['mae_ms']:.3f}" if 'mae_ms' in m else '不适用'
        lines.append(f"| `{method}` | {mape} | {mae} | {m['spearman']:.4f} | {m['kendall_tau_b']:.4f} |")
    lines += ['\n| 方法 | 锁定最快选型 | 实测最优网络 | 选型状态 | 相对延迟 regret |', '| --- | --- | --- | --- | ---: |']
    for method, d in result['decisions'].items():
        s = d['fastest']
        regret = '不适用' if s['regret'] is None else f"{100*s['regret']:.4f}%"
        lines.append(f"| `{method}` | `{s['chosen']}` | `{s.get('oracle')}` | {s['status']} | {regret} |")
    lines += ['\n内存误差采用字节单位，内部堆可能为零，不计算其百分比误差。\n',
              '| 指标 | 方法 | MAE bytes | 最大低估 bytes | 误判为可行的数量 |', '| --- | --- | ---: | ---: | ---: |']
    for resource, methods in result['memory'].items():
        for method, m in methods.items():
            lines.append(f"| {resource} | {method} | {m['mae_bytes']:.3f} | {m['maximum_underprediction_bytes']:.3f} | {m['feasibility_counts']['false_feasible']} |")
    lines += ['\n所有方法在揭示查询实测前已选中同一网络，因此这些固定选择不能展示主方法相对基线的选型优势。预测误差是否改善应以本节表格为准。\n',
              '选中网络的完整测试集 Top-1 仍为 89.4942%；硬件功能和内存通过不改变测试精度不满 90% 的事实。延迟为纯 Invoke 耗时，没有定义实际应用 SLA。\n',
              f'成本分项：六网络烧录及有效采集合计 {support_components:.3f} s；九网络离线验证烧录及有效采集合计 {query_components:.3f} s；已选网络单独验收占 {selected_flash+selected_capture:.3f} s（已包含在九网络中，不重复计费）。另有六网络阶段中断尝试 {support["interrupted_capture_wall_seconds"]:.3f} s。上述不包含构建、下载、备份、识别和等待，不能作为端到端加速比。\n',
              '本轮 54 个会话包含 2,754 次内存模式推理、5,508 次正式计时推理、270 次预热，正式输出比较未发现 INT8 元素差异。板上保留最后一个查询网络 `31977ff8dec9d005` 的测量固件；原固件备份未改动。\n',
              '完整结果：`results/strict6-physical-query-v1/summary.json`；原冻结分析结果：`evaluation.json`；独立审计：`audit.json`；全部记录索引：`query-receipts.json`。\n']
    readme = ROOT / 'README.md'
    text = readme.read_text()
    assert '## 九网络实板评估最终结果（已完成）' not in text
    text = text.replace('## 九网络实板独立评估（2026-10-28，进行中）', '## 九网络实板独立评估（启动记录；最终结果见下方）')
    readme.write_text(text + '\n'.join(lines))
    print({'summary': str(BASE / 'summary.json'), 'audit_passed': True, 'selected': selected})


if __name__ == '__main__':
    main()
