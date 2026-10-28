"""Re-read every boot log, re-check image provenance and independently rescore predictions."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from measurement_io import ROOT, PROTOCOL_SHA, design, read, read_receipt, sha, write_new
from strict6 import verify_lock


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);a=ap.parse_args()
    run=Path(a.run).resolve();complete=read(run/'complete.json');spec=read(run/'run-spec.json')
    assert complete['passed'] and not complete['physical_board_accessed']
    for path,digest in spec['source_hashes'].items():assert sha(ROOT/path)==digest
    protocol,features=design();lock=verify_lock(run/'calibration/lock.json',True)
    evaluation=read(run/'evaluation.json')
    assert sha(run/'calibration/lock.json')==complete['prediction_lock_sha256']==evaluation['prediction_lock_sha256']
    assert sha(run/'evaluation.json')==complete['evaluation_sha256']
    assert evaluation['simulation_only'] and evaluation['physical_measured_networks']==0
    model_reports=[];tuples=set();nonces=set();all_query_starts=[];all_support_ends=[]
    for ident in protocol['support_ids']+protocol['query_ids']:
        directory=run/'models'/ident;manifest=read(directory/'manifest.json')
        assert manifest['production_sha256']==features[ident]['production_sha256']
        for name,digest in manifest['artifact_hashes'].items():assert sha(directory/name)==digest,(ident,name)
        assert manifest['runtime_fingerprint']==lock['runtime_fingerprint']
        assert manifest['embedded_alignment_bytes']==16
        assert len(manifest['embedded_symbol_addresses'])==3
        assert all(address%16==0 for address in manifest['embedded_symbol_addresses'].values())
        values=[]
        for path in sorted((directory/'qemu').glob('*/receipt.json')):
            receipt,_=read_receipt(path);assert receipt['source']=='qemu'
            assert receipt['nonce'] not in nonces;nonces.add(receipt['nonce'])
            item=(ident,receipt['kind'],receipt['session']);assert item not in tuples;tuples.add(item)
            command=receipt['qemu_command'];assert '-snapshot' in command
            assert command[command.index('-m')+1]=='8M' and command[command.index('-machine')+1]=='esp32s3'
            if ident in protocol['query_ids']:
                assert receipt['started_unix']>lock['created_unix'];all_query_starts.append(receipt['started_unix'])
            else:
                assert receipt['ended_unix']<lock['created_unix'];all_support_ends.append(receipt['ended_unix'])
            if receipt['kind']=='memory':values.append(receipt['parsed']['memory'])
        assert len(values)==3
        model_reports.append({'id':ident,'role':'support' if ident in protocol['support_ids'] else 'query',
            'model_sha256':manifest['production_sha256'],'firmware_sha256':manifest['artifact_hashes']['flash_image.bin'],
            'arena_used_bytes':manifest['arena_used_bytes'],'arena_reserved_bytes':manifest['arena_reserved_bytes'],
            'qemu_psram_peak_bytes':max(v['psram_peak_bytes'] for v in values),
            'qemu_internal_heap_peak_bytes':max(v['internal_heap_peak_bytes'] for v in values),
            'benchmark_firmware_bytes_including_fixtures':manifest['benchmark_firmware_bytes']})
    expected={(i,k,s) for i in features for k in ['timing','memory'] for s in range(3)}
    assert tuples==expected and len(tuples)==90
    assert all(r['qemu_psram_peak_bytes']<=6*2**20 and r['qemu_internal_heap_peak_bytes']<=192*2**10 for r in model_reports)
    # Independent augmented least squares, rather than the production normal-equation solver.
    support=lock['support_observations']['networks'];y=np.log([r['latency_ms'] for r in support])
    for name,d in [('log_macs_ridge',1),('structure_ridge_primary',3)]:
        x=np.asarray([[np.log(features[r['id']]['macs']),np.log(features[r['id']]['activation_io_bytes']),features[r['id']]['depthwise_mac_fraction']][:d] for r in support])
        std=x.std(0);std[std==0]=1;z=(x-x.mean(0))/std
        design_matrix=np.column_stack([np.ones(6),z]);penalty=np.column_stack([np.zeros(d),np.eye(d)])
        beta=np.linalg.lstsq(np.vstack([design_matrix,penalty]),np.r_[y,np.zeros(d)],rcond=None)[0]
        model=lock['models'][name]
        np.testing.assert_allclose(beta,np.r_[model['intercept'],model['coefficients']],rtol=1e-9,atol=1e-9)
        for ident in protocol['query_ids']:
            f=features[ident];v=np.asarray([np.log(f['macs']),np.log(f['activation_io_bytes']),f['depthwise_mac_fraction']][:d])
            prediction=float(np.exp(beta[0]+((v-x.mean(0))/std)@beta[1:]))
            np.testing.assert_allclose(prediction,lock['predictions'][ident][name]['latency_ms'],rtol=1e-9)
    targets={r['id']:r['latency_ms'] for r in evaluation['query_observations']['networks']}
    for name in ['scaled_macs','log_macs_ridge','structure_ridge_primary']:
        actual=np.asarray([targets[i] for i in protocol['query_ids']])
        prediction=np.asarray([lock['predictions'][i][name]['latency_ms'] for i in protocol['query_ids']])
        np.testing.assert_allclose(100*np.mean(np.abs(prediction-actual)/actual),evaluation['metrics'][name]['mape_percent'],rtol=1e-12)
        np.testing.assert_allclose(np.mean(np.abs(prediction-actual)),evaluation['metrics'][name]['mae_ms'],rtol=1e-12)
    result={'passed':True,'audited_unix':time.time(),'protocol_sha256':PROTOCOL_SHA,
        'models':15,'support':6,'query':9,'cold_boot_sessions':90,'timed_invokes':9180,
        'memory_invokes':4590,'warmup_invokes':450,'integer_output_mismatches':0,
        'prediction_lock_after_all_support_before_all_query':max(all_support_ends)<lock['created_unix']<min(all_query_starts),
        'independent_ridge_and_error_rescore_passed':True,'model_and_firmware_hashes_verified':True,
        'embedded_data_alignment_verified':True,
        'simulation_only':True,'physical_board_accessed':False,'physical_measurements':0,
        'models_detail':model_reports}
    write_new(run/'audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='models_detail'},indent=2))


if __name__=='__main__':main()
