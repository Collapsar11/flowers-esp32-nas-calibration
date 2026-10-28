"""Reproduce frozen six-support results from published aggregates, without hardware."""
import json
import math
from pathlib import Path
import statistics

import strict6 as s
from measurement_io import design, read

BASE = Path(__file__).resolve().parents[1] / 'results/published'


def same(actual, expected, path='root'):
    if isinstance(expected, dict):
        assert set(actual) == set(expected), path
        for k in expected:
            same(actual[k], expected[k], path+'.'+k)
    elif isinstance(expected, list):
        assert len(actual) == len(expected), path
        for i, (a, e) in enumerate(zip(actual, expected)):
            same(a, e, path+'.'+str(i))
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        assert math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10), (path, actual, expected)
    else:
        assert actual == expected, (path, actual, expected)


def main():
    protocol, features = design()
    support = read(BASE/'support-observations.json')
    frozen = read(BASE/'frozen-predictions.json')
    assert len(support) == 6
    assert [r['id'] for r in support] == frozen['support_ids'] == protocol['support_ids']
    assert set(protocol['support_ids']).isdisjoint(protocol['query_ids'])
    assert frozen['query_labels_accessed'] is False
    rows = [features[r['id']] for r in support]
    latency = [r['latency_ms'] for r in support]
    models = {name:s.fit_ridge(rows,latency,d) for name,d in [('log_macs_ridge',1),('structure_ridge_primary',3)]}
    same(models, frozen['models'])
    scale = statistics.median(t/r['macs'] for t,r in zip(latency,rows))
    same(scale, frozen['scaled_macs_ms_per_mac'])
    offsets = {k:statistics.median(r[k]-features[r['id']]['qemu_'+k] for r in support)
               for k in ['psram_peak_bytes','internal_heap_peak_bytes']}
    same(offsets, frozen['memory_offsets_bytes'])
    predictions = {}
    for ident in protocol['query_ids']:
        row = features[ident]
        predictions[ident] = {}
        for method in s.METHODS:
            ms = None if method=='raw_macs' else scale*row['macs'] if method=='scaled_macs' else s.predict(models[method],row)
            predictions[ident][method] = {'latency_ms':ms, **{
                k:float(max(0,row['qemu_'+k]+(0 if method=='raw_macs' else offsets[k]))) for k in offsets}}
    same(predictions, frozen['predictions'])
    budgets = [factor*statistics.median(latency) for factor in [.75,1.,1.25]]
    same(budgets, frozen['latency_budgets_ms'])
    decisions = {method:{'fastest':s.choose(predictions,features,method),
        'accuracy_under_budgets':[{'budget_ms':b,'id':s.choose(predictions,features,method,b),
                                 'applicable':method!='raw_macs'} for b in budgets]} for method in s.METHODS}
    same(decisions, frozen['decisions'])

    # Query labels are first loaded after fitting and reproducing all decisions.
    query = read(BASE/'query-observations.json')
    expected = read(BASE/'evaluation.json')
    actual = {r['id']:r for r in query}
    ids = protocol['query_ids']
    assert len(query)==9 and set(actual)==set(ids)
    metrics, evaluated = {}, {}
    for method in s.METHODS:
        pred = [features[i]['macs'] if method=='raw_macs' else predictions[i][method]['latency_ms'] for i in ids]
        target = [actual[i]['latency_ms'] for i in ids]
        metrics[method] = s.rank_metrics(pred,target) if method=='raw_macs' else s.latency_metrics(pred,target)
        d = decisions[method]
        evaluated[method] = {'fastest':s.decision_score(d['fastest'],actual,features),
            'accuracy_under_budgets':[dict(b,status='not_applicable_raw_macs') if not b['applicable']
              else s.decision_score(b['id'],actual,features,b['budget_ms']) for b in d['accuracy_under_budgets']]}
    memory = {}
    for key,limit in [('psram_peak_bytes',6*2**20),('internal_heap_peak_bytes',192*2**10)]:
        memory[key] = {name:s.memory_metrics([predictions[i][method][key] for i in ids],
            [actual[i][key] for i in ids],limit) for name,method in
            [('qemu_baseline','raw_macs'),('six_label_residual_correction','structure_ridge_primary')]}
    same(metrics,expected['metrics'])
    same(memory,expected['memory'])
    same(evaluated,expected['decisions'])
    print(json.dumps({'passed':True,'scope':'numerical reproduction from published aggregates',
        'support_labels_used':6,'heldout_networks':9,'hardware_accessed':False,
        'primary_mape_percent':metrics['structure_ridge_primary']['mape_percent'],
        'scaled_macs_mape_percent':metrics['scaled_macs']['mape_percent'],
        'selected':decisions['structure_ridge_primary']['fastest']},indent=2))


if __name__ == '__main__':
    main()
