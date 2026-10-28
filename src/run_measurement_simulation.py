"""Full server-only rehearsal: six support acquisitions -> lock -> nine query acquisitions."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import json
from pathlib import Path
import time
from measurement_io import ROOT, PROTOCOL_SHA, design, read, sha, write_new
from measure_campaign import build, qemu_session
from strict6 import calibrate, evaluate, verify_lock


def phase(ids, out, workers):
    receipts=[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs=[]
        for ident in ids:
            directory=out/'models'/ident
            build(ident,directory)
            print(json.dumps({'built':ident,'phase':'support' if ident in design()[0]['support_ids'] else 'query'}),flush=True)
            for kind in ['timing','memory']:
                for session in range(3): jobs.append(pool.submit(qemu_session,directory,kind,session))
        for job in as_completed(jobs):receipts.append(job.result())
    return sorted(receipts)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='runs/strict6-simulation-v1');ap.add_argument('--workers',type=int,default=24)
    a=ap.parse_args();out=(ROOT/a.out).resolve();out.mkdir(parents=True,exist_ok=True)
    with (out/'campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        protocol,_=design()
        spec={'protocol_sha256':PROTOCOL_SHA,'physical_board_accessed':False,'source':'qemu',
            'source_hashes':{n:sha(ROOT/n) for n in ['src/measurement_io.py','src/strict6.py','src/measure_campaign.py','src/run_measurement_simulation.py','firmware-measure/main/app_main.cpp','firmware-measure/main/embedded.S.in','firmware-measure/main/CMakeLists.txt','scripts/build/build_measure_firmware.sh']}}
        if (out/'run-spec.json').exists():assert read(out/'run-spec.json')==spec, 'code changed; preserve this run and use a new run directory'
        else:write_new(out/'run-spec.json',spec)
        if (out/'complete.json').exists():
            complete=read(out/'complete.json')
            assert complete['passed'] and complete['evaluation_sha256']==sha(out/'evaluation.json')
            assert complete['prediction_lock_sha256']==sha(out/'calibration/lock.json')
            print(json.dumps(complete),flush=True)
            return
        if not (out/'support-receipts.json').exists():
            receipts=phase(protocol['support_ids'],out,a.workers)
            write_new(out/'support-receipts.json',[str(Path(p).relative_to(out)) for p in receipts])
        if not (out/'calibration/lock.json').exists():calibrate(out/'support-receipts.json',out/'calibration',True)
        verify_lock(out/'calibration/lock.json',True)
        print(json.dumps({'prediction_lock_ready':True,'query_acquisition_can_begin':True}),flush=True)
        if not (out/'query-receipts.json').exists():
            receipts=phase(protocol['query_ids'],out,a.workers)
            write_new(out/'query-receipts.json',[str(Path(p).relative_to(out)) for p in receipts])
        if not (out/'evaluation.json').exists():evaluate(out/'calibration/lock.json',out/'query-receipts.json',out/'evaluation.json',True)
        evaluation=read(out/'evaluation.json')
        assert evaluation['physical_measured_networks']==0
        write_new(out/'complete.json',{'finished_unix':time.time(),'passed':True,'models':15,
            'cold_boot_sessions':90,'timed_invokes':15*3*204,'memory_invokes':15*3*102,
            'source':'qemu','silicon_latency_valid':False,'physical_board_accessed':False,
            'prediction_lock_sha256':sha(out/'calibration/lock.json'),'evaluation_sha256':sha(out/'evaluation.json')})
        print(json.dumps(read(out/'complete.json')),flush=True)


if __name__=='__main__':main()
