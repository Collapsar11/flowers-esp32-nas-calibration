"""Exactly-six calibration, immutable prediction lock, then nine-label evaluation.

No device access. --simulation is required for QEMU and never creates silicon evidence.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
import numpy as np
from measurement_io import ROOT, PROTOCOL_SHA, collect, design, read, sha, write_new

METHODS=['raw_macs','scaled_macs','log_macs_ridge','structure_ridge_primary']


def vector(row, dimensions):
    return [math.log(row['macs']),math.log(row['activation_io_bytes']),row['depthwise_mac_fraction']][:dimensions]


def fit_ridge(rows, labels, dimensions):
    x=np.asarray([vector(r,dimensions) for r in rows],dtype=np.float64)
    y=np.log(np.asarray(labels,dtype=np.float64))
    mean=x.mean(0); scale=x.std(0); scale[scale==0]=1
    z=(x-mean)/scale; intercept=float(y.mean())
    w=np.linalg.solve(z.T@z+np.eye(dimensions),z.T@(y-intercept))
    return {'dimensions':dimensions,'mean':mean.tolist(),'scale':scale.tolist(),
        'coefficients':w.tolist(),'intercept':intercept,'alpha':1.0,'training_count':len(rows)}


def predict(model, row):
    z=(np.asarray(vector(row,model['dimensions']))-model['mean'])/model['scale']
    value=float(np.exp(model['intercept']+np.dot(z,model['coefficients'])))
    assert math.isfinite(value) and value>0
    return value


def feasible(feature, memory, latency=None, budget=None):
    return (feature['validation_top1']>=.90 and feature['model_bytes']<=12*2**20
        and memory['psram_peak_bytes']<=6*2**20 and memory['internal_heap_peak_bytes']<=192*2**10
        and (budget is None or latency<=budget))


def choose(predictions, features, method, budget=None):
    if method=='raw_macs' and budget is not None:return None
    pool=[]
    for ident, entry in predictions.items():
        p=entry[method]; latency=p['latency_ms']
        if feasible(features[ident],p,latency,budget):pool.append(ident)
    if not pool:return None
    cost=lambda ident:features[ident]['macs'] if method=='raw_macs' else predictions[ident][method]['latency_ms']
    key=(lambda ident:(cost(ident),ident)) if budget is None else (lambda ident:(-features[ident]['validation_top1'],cost(ident),ident))
    return min(pool,key=key)


def file_list(path):
    path=Path(path); value=read(path)
    assert isinstance(value,list) and all(isinstance(x,str) for x in value)
    return [(path.parent/x).resolve() for x in value]


def calibrate(receipts_file, out, simulation=False):
    start=time.perf_counter(); out=Path(out).resolve()
    assert not out.exists(), 'new calibration directory required; never overwrite predictions'
    protocol,features=design()
    support=collect(file_list(receipts_file),protocol['support_ids'],simulation)
    rows=[features[r['id']] for r in support['networks']]
    latency=[r['latency_ms'] for r in support['networks']]
    assert len(rows)==6 and all(math.isfinite(t) and t>0 for t in latency)
    models={name:fit_ridge(rows,latency,d) for name,d in [('log_macs_ridge',1),('structure_ridge_primary',3)]}
    proportional=statistics.median(t/r['macs'] for t,r in zip(latency,rows))
    offsets={k:statistics.median(r[k]-features[r['id']]['qemu_'+k] for r in support['networks'])
        for k in ['psram_peak_bytes','internal_heap_peak_bytes']}
    predictions={}
    for ident in protocol['query_ids']:
        row=features[ident]; entry={}
        for method in METHODS:
            ms=None if method=='raw_macs' else proportional*row['macs'] if method=='scaled_macs' else predict(models[method],row)
            memory={k:float(max(0,row['qemu_'+k]+(0 if method=='raw_macs' else offsets[k]))) for k in offsets}
            entry[method]={'latency_ms':ms,**memory}
        predictions[ident]=entry
    budgets=[factor*statistics.median(latency) for factor in [.75,1.,1.25]]
    decisions={method:{'fastest':choose(predictions,features,method),
        'accuracy_under_budgets':[{'budget_ms':b,'id':choose(predictions,features,method,b),
            'applicable':method!='raw_macs'} for b in budgets]} for method in METHODS}
    loocv={method:[] for method in METHODS if method!='raw_macs'}
    for i in range(6):
        train=[j for j in range(6) if j!=i]
        x=[rows[j] for j in train]; y=[latency[j] for j in train]
        for name,d in [('log_macs_ridge',1),('structure_ridge_primary',3)]:
            loocv[name].append({'id':rows[i]['id'],'observed_ms':latency[i], 'predicted_ms':predict(fit_ridge(x,y,d),rows[i])})
        loocv['scaled_macs'].append({'id':rows[i]['id'],'observed_ms':latency[i],
            'predicted_ms':rows[i]['macs']*statistics.median(t/r['macs'] for t,r in zip(y,x))})
    # Store portable paths relative to the immutable lock, not machine-specific paths.
    for receipt in support['receipts']: receipt['path']=os.path.relpath(receipt['path'],out)
    frozen={'protocol_sha256':PROTOCOL_SHA,'created_unix':time.time(),
        'source':'qemu' if simulation else 'physical','simulation_only':simulation,
        'silicon_latency_valid':not simulation,'physical_calibration_network_count':0 if simulation else 6,
        'support_ids':protocol['support_ids'],'query_ids':protocol['query_ids'],
        'runtime_fingerprint':support['runtime_fingerprint'],'support_observations':support,
        'models':models,'scaled_macs_ms_per_mac':proportional,'memory_offsets_bytes':offsets,
        'predictions':predictions,'decisions':decisions,'latency_budgets_ms':budgets,
        'support_loocv_diagnostic_not_query_evidence':loocv,
        'fit_and_selection_wall_seconds':time.perf_counter()-start,
        'source_hashes':{name:sha(ROOT/name) for name in ['strict6.py','measurement_io.py']},
        'query_labels_accessed':False,
        'interpretation':'QEMU workflow test only; these values are NOT silicon predictions' if simulation else 'Predictions locked before held-out acquisition; not yet evaluated'}
    write_new(out/'lock.json',frozen)
    write_new(out/'lock-receipt.json',{'sha256':sha(out/'lock.json'),'created_unix':frozen['created_unix'],
        'query_labels_accessed':False})
    return frozen


def verify_lock(path, simulation):
    path=Path(path); lock=read(path)
    assert sha(path)==read(path.parent/'lock-receipt.json')['sha256'], 'prediction lock hash changed'
    assert lock['protocol_sha256']==PROTOCOL_SHA and lock['simulation_only']==simulation
    assert lock['source']==('qemu' if simulation else 'physical')
    for name,digest in lock['source_hashes'].items():assert sha(ROOT/name)==digest, 'analysis source changed after lock'
    for r in lock['support_observations']['receipts']:
        assert sha(path.parent/r['path'])==r['sha256'], 'support evidence changed after lock'
    protocol,_=design()
    assert lock['support_ids']==protocol['support_ids'] and lock['query_ids']==protocol['query_ids']
    return lock


def ranks(values):
    result=np.zeros(len(values)); order=sorted(range(len(values)),key=lambda i:values[i]); i=0
    while i<len(order):
        j=i+1
        while j<len(order) and values[order[j]]==values[order[i]]:j+=1
        for k in order[i:j]:result[k]=(i+j-1)/2+1
        i=j
    return result


def rank_metrics(predicted, actual):
    x,y=ranks(predicted),ranks(actual)
    rho=None if np.std(x)==0 or np.std(y)==0 else float(np.corrcoef(x,y)[0,1])
    c=d=tx=ty=0
    for i in range(len(x)):
        for j in range(i):
            sx=np.sign(x[i]-x[j]);sy=np.sign(y[i]-y[j])
            if sx*sy>0:c+=1
            elif sx*sy<0:d+=1
            elif sx==0 and sy!=0:tx+=1
            elif sx!=0 and sy==0:ty+=1
    den=math.sqrt((c+d+tx)*(c+d+ty))
    return {'spearman':rho,'kendall_tau_b':(c-d)/den if den else None,
        'pairwise_ordering_agreement':(c+.5*tx)/(c+d+tx) if c+d+tx else None,
        'unequal_true_pairs':c+d+tx}


def latency_metrics(predicted,actual):
    pred=np.asarray(predicted); y=np.asarray(actual); diff=np.abs(pred-y); rel=100*diff/y
    return {'mape_percent':float(rel.mean()),'mae_ms':float(diff.mean()),'rmse_ms':float(np.sqrt(np.mean((pred-y)**2))),
        'median_ape_percent':float(np.median(rel)),'maximum_ape_percent':float(rel.max()),
        'per_network_ape_percent':rel.tolist(),**rank_metrics(predicted,actual)}


def memory_metrics(pred,actual,limit):
    delta=np.asarray(pred)-np.asarray(actual)
    return {'mae_bytes':float(np.abs(delta).mean()),'maximum_absolute_error_bytes':float(np.abs(delta).max()),
        'maximum_underprediction_bytes':float(max(0,-delta.min())),
        'percentage_error':None,'percentage_error_reason':'zero-valued internal heap labels are possible; byte metrics used',
        'feasibility_counts':{name:sum(bool(condition(p,y)) for p,y in zip(pred,actual)) for name,condition in {
            'true_feasible':lambda p,y:p<=limit and y<=limit,
            'false_feasible':lambda p,y:p<=limit<y,
            'false_infeasible':lambda p,y:y<=limit<p,
            'true_infeasible':lambda p,y:p>limit and y>limit}.items()}}


def decision_score(chosen, actual, features, budget=None):
    feasible_ids=[ident for ident,r in actual.items() if feasible(features[ident],r,r['latency_ms'],budget)]
    if not feasible_ids:return {'status':'no_feasible_oracle','chosen':chosen,'regret':None}
    key=(lambda i:(actual[i]['latency_ms'],i)) if budget is None else (lambda i:(-features[i]['validation_top1'],actual[i]['latency_ms'],i))
    oracle=min(feasible_ids,key=key)
    result={'chosen':chosen,'oracle':oracle,'budget_ms':budget}
    if chosen is None:return dict(result,status='no_selection',regret=None)
    if chosen not in feasible_ids:return dict(result,status='budget_violation',regret=None,
        actual_latency_ms=actual[chosen]['latency_ms'])
    regret=((actual[chosen]['latency_ms']-actual[oracle]['latency_ms'])/actual[oracle]['latency_ms']) if budget is None else 100*(features[oracle]['validation_top1']-features[chosen]['validation_top1'])
    return dict(result,status='feasible',selection_agreement=chosen==oracle,regret=regret,
        regret_unit='relative_latency' if budget is None else 'accuracy_percentage_points')


def evaluate(lock_path, receipts_file, out, simulation=False):
    lock_path=Path(lock_path); lock=verify_lock(lock_path,simulation)
    protocol,features=design()
    query=collect(file_list(receipts_file),protocol['query_ids'],simulation,after=lock['created_unix'])
    assert query['runtime_fingerprint']==lock['runtime_fingerprint'], 'query runtime differs from support'
    actual={r['id']:r for r in query['networks']}; ids=protocol['query_ids']; metrics={}; decisions={}
    for method in METHODS:
        prediction=[features[i]['macs'] if method=='raw_macs' else lock['predictions'][i][method]['latency_ms'] for i in ids]
        target=[actual[i]['latency_ms'] for i in ids]
        metrics[method]=rank_metrics(prediction,target) if method=='raw_macs' else latency_metrics(prediction,target)
        decision=lock['decisions'][method]
        decisions[method]={'fastest':decision_score(decision['fastest'],actual,features),
            'accuracy_under_budgets':[dict(b,status='not_applicable_raw_macs') if not b['applicable'] else decision_score(b['id'],actual,features,b['budget_ms']) for b in decision['accuracy_under_budgets']]}
    memory={}
    for key,limit in [('psram_peak_bytes',6*2**20),('internal_heap_peak_bytes',192*2**10)]:
        memory[key]={name:memory_metrics([lock['predictions'][i][method][key] for i in ids],[actual[i][key] for i in ids],limit)
            for name,method in [('qemu_baseline','raw_macs'),('six_label_residual_correction','structure_ridge_primary')]}
    result={'protocol_sha256':PROTOCOL_SHA,'prediction_lock_sha256':sha(lock_path),
        'evaluated_unix':time.time(),'simulation_only':simulation,'source':query['source'],
        'silicon_latency_valid':not simulation,'physical_measured_networks':0 if simulation else 15,
        'calibration_architectures':6,'heldout_architectures':9,'query_ids_in_metric_order':ids,
        'metrics':metrics,'memory':memory,'decisions':decisions,'query_observations':query,
        'measurement_count_reduction_for_decision':.60,'measured_wall_time_saving':None,
        'cost_note':'Count reduction is not time reduction. Shared setup, per-model build/load/capture and fitting must be accounted separately.',
        'interpretation':'SIMULATION PIPELINE CHECK ONLY; errors and rankings do not establish ESP32 silicon prediction quality' if simulation else 'Held-out measurements used for evaluation only; no refitting'}
    write_new(out,result); return result


def main():
    ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='command',required=True)
    for name in ['calibrate','evaluate']:
        p=sub.add_parser(name);p.add_argument('--receipts',required=True);p.add_argument('--out',required=True);p.add_argument('--simulation',action='store_true')
        if name=='evaluate':p.add_argument('--lock',required=True)
    args=ap.parse_args()
    if args.command=='calibrate':result=calibrate(args.receipts,args.out,args.simulation)
    else:result=evaluate(args.lock,args.receipts,args.out,args.simulation)
    print(json.dumps({'complete':args.command,'simulation_only':result['simulation_only'],'physical_board_accessed':False}))


if __name__=='__main__':main()
