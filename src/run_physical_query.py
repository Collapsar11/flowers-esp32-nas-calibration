"""Held-out physical evaluation using an immutable six-support prediction lock.

Nine query models only, selected model first. Never fits or changes a decision.
Keeps original support campaign, analysis sources, firmware, and backup intact.
Incomplete attempts stop for review; no evidence is overwritten.
"""
import datetime
import csv
import fcntl
import os
from pathlib import Path
import json
import shutil
import subprocess
import sys
import time
import uuid
import serial
from esptool.reset import HardReset
from measurement_io import ROOT, PROTOCOL_SHA, design, read, read_receipt, sha, write_new, parse_serial, collect
from strict6 import verify_lock, evaluate

BASE=ROOT/'results/strict6-physical-query-v1'
KIT=ROOT/'results/strict6-simulation-v3-kit/runs/strict6-simulation-v3/models'
PILOT=ROOT/'results/esp32-first-board-v1'
PORT=os.environ['FLOWERS_SERIAL_PORT']
BOARD=os.environ['FLOWERS_BOARD_MAC']
OPS=BASE/'operations'
LOCK=ROOT/'results/strict6-physical-v1/calibration/lock.json'
LOCK_SHA='f0698cd1fb0cb8623b919c1a413acb9b0190ad258b49c13d16c9c1734f3a9fc3'

def event(stage, **values):
    record={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'unix':time.time(),'stage':stage,**values}
    print(json.dumps(record),flush=True)
    with (OPS/'events.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
    temp=OPS/'status.tmp';temp.write_text(json.dumps(record,indent=2)+'\n');temp.replace(OPS/'status.json')


def manifest_for(ident):
    protocol,features=design();assert ident in protocol['query_ids'], 'only frozen query acquisition is authorized'
    frozen=verify_lock(LOCK,False)
    assert sha(LOCK)==LOCK_SHA and time.time()>frozen['created_unix']
    directory=KIT/ident; manifest=read(directory/'manifest.json')
    assert manifest['runtime_fingerprint']==frozen['runtime_fingerprint']
    assert manifest['id']==ident and manifest['protocol_sha256']==PROTOCOL_SHA
    assert manifest['production_sha256']==features[ident]['production_sha256']
    assert sha(directory/'flash_image.bin')==manifest['artifact_hashes']['flash_image.bin']
    return directory,manifest


def esptool(ident, label, arguments):
    assert ident in design()[0]['query_ids']
    verify_lock(LOCK,False);assert sha(LOCK)==LOCK_SHA
    directory=OPS/ident;directory.mkdir(parents=True,exist_ok=True)
    logpath=directory/(label+'.log');resultpath=directory/(label+'.json')
    assert not logpath.exists() and not resultpath.exists(), 'preserve earlier attempt; review before retry'
    cmd=[sys.executable,'-u','-m','esptool','--chip','esp32s3','--port',PORT,
         '--baud','460800','--after','no_reset',*arguments]
    event(label+'_started',id=ident,log=str(logpath.relative_to(ROOT)))
    start=time.time();mono=time.monotonic()
    with logpath.open('x') as log:
        result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
    text=logpath.read_text(errors='replace')
    record={'command':cmd,'started_unix':start,'ended_unix':time.time(),
            'wall_seconds':time.monotonic()-mono,'exit_code':result.returncode,
            'log_sha256':sha(logpath),'id':ident,'board_mac':BOARD}
    write_new(resultpath,record)
    assert result.returncode==0, text[-2000:]
    assert 'MAC: '+BOARD in text and 'Chip is ESP32-S3' in text
    if arguments[0]=='flash_id':assert 'Detected flash size: 16MB' in text
    if arguments[0]=='verify_flash':assert '-- verify OK (digest matched)' in text
    if arguments[0]=='write_flash':assert 'Hash of data verified.' in text
    event(label+'_complete',id=ident,wall_seconds=record['wall_seconds'])
    return record


def capture(ident,kind,session,load_seconds=None):
    directory,manifest=manifest_for(ident)
    folder=BASE/ident/(kind+'-'+str(session));receipt=folder/'receipt.json'
    if receipt.exists():
        r,m=read_receipt(receipt)
        assert r['prediction_lock_sha256']==LOCK_SHA
        assert r['started_unix']>verify_lock(LOCK,False)['created_unix']
        assert r['source']=='physical' and r['board_serial']==BOARD
        assert r['kind']==kind and r['session']==session and m['id']==ident
        assert m['runtime_fingerprint']==manifest['runtime_fingerprint']
        assert r['loaded_image_sha256_declared']==manifest['artifact_hashes']['flash_image.bin']
        event('reused',id=ident,kind=kind,session=session)
        return receipt
    folder.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(directory/'manifest.json',folder/'manifest.json')
    port=serial.Serial(port=None,baudrate=115200,timeout=.2)
    port.dtr=False;port.rts=False;port.port=PORT
    nonce=uuid.uuid4().hex;data=b'';sent=False;started=time.time();reset_unix=None;manual_needed=False
    event('capture_starting',id=ident,kind=kind,session=session)
    try:
        port.open()
        # Hold EN low and clear stale bytes before the deliberate independent reset.
        port.setDTR(False);port.setRTS(True);time.sleep(.1);port.reset_input_buffer()
        reset_unix=time.time();HardReset(port,uses_usb=False)()
        event('reset_sent',id=ident,kind=kind,session=session,reset_method='esptool.HardReset RTS/EN')
        with (folder/'serial.log').open('xb') as log:
            while time.time()-started<1800:
                chunk=port.read(port.in_waiting or 1)
                if not chunk:
                    if not sent and not manual_needed and time.time()-reset_unix>30:
                        event('manual_reset_needed',id=ident,kind=kind,session=session)
                        manual_needed=True
                    continue
                log.write(chunk);log.flush();data+=chunk
                if not sent and b'FLOWERS_READY ' in data:
                    assert data.count(b'ESP-ROM:esp32s3')==1, 'require exactly one clean ROM boot'
                    assert ('FLOWERS_READY '+ident).encode() in data, 'wrong flashed model'
                    port.write(f'RUN {kind} {session} {nonce}\n'.encode());port.flush();sent=True
                    event('measuring',id=ident,kind=kind,session=session)
                if b'FLOWERS_MEASUREMENT_COMPLETE' in data:break
                if any(x in data for x in [b'Guru Meditation',b'abort() was called',b'assert failed',b'FLOWERS_CHECK_FAILED']):break
        text=data.decode(errors='replace')
        assert data.count(b'ESP-ROM:esp32s3')==1
        parsed=parse_serial(text,manifest,kind,session,nonce)
    except BaseException as exc:
        write_new(folder/'failure.json',{'error':repr(exc),'started_unix':started,'ended_unix':time.time(),'nonce':nonce})
        raise
    finally:
        port.close()
    result={'source':'physical','transport':'serial','silicon_latency_valid':True,
        'port':PORT,'board_serial':BOARD,'cold_boot_requested':True,'rom_boot_seen':True,
        'reset_method':'manual fallback requested after RTS reset' if manual_needed else 'USB-UART RTS/EN via esptool.HardReset; DTR/BOOT released',
        'kind':kind,'session':session,'nonce':nonce,'started_unix':started,'ended_unix':time.time(),
        'log_file':'serial.log','log_sha256':sha(folder/'serial.log'),
        'manifest_file':'manifest.json','manifest_sha256':sha(folder/'manifest.json'),'parsed':parsed,
        'loaded_image_sha256_declared':sha(directory/'flash_image.bin'),'load_seconds':load_seconds,
        'capture_source_sha256':sha(Path(__file__)), 'prediction_lock_sha256':LOCK_SHA,
        'provenance_limit':'Board MAC verified through esptool and matching firmware banner; no cryptographic attestation'}
    write_new(receipt,result);read_receipt(receipt)
    summary={'id':ident,'kind':kind,'session':session,'receipt':str(receipt.relative_to(ROOT))}
    if kind=='timing':summary['median_ms']=parsed['median_ms']
    else:summary['psram_peak_bytes']=parsed['memory']['psram_peak_bytes']
    event('capture_complete',**summary)
    return receipt


def main():
    frozen=verify_lock(LOCK,False);assert sha(LOCK)==LOCK_SHA
    protocol,features=design()
    original=read(PILOT/'original-backup.json')
    assert original['verified_against_device'] and sha(PILOT/original['file'])==original['sha256']
    selected=frozen['decisions']['structure_ridge_primary']['fastest']
    test_path=ROOT/'results/nas-final-test-v1/test-metrics.csv'
    with test_path.open() as stream:
        test_row=next(row for row in csv.DictReader(stream) if row['id']==selected)
    assert test_row['production_sha256']==features[selected]['production_sha256']
    test_top1=float(test_row['int8_test_top1'])
    order=[selected]+[i for i in protocol['query_ids'] if i!=selected]
    assert len(order)==9 and len(set(order))==9
    # Remaining images may still be downloading; each must pass the full image
    # hash gate immediately before its device operation.
    for ident in order:
        manifest=read(KIT/ident/'manifest.json')
        assert manifest['id']==ident and manifest['protocol_sha256']==PROTOCOL_SHA
        assert manifest['runtime_fingerprint']==frozen['runtime_fingerprint']
        assert manifest['production_sha256']==features[ident]['production_sha256']
    manifest_for(selected)
    OPS.mkdir(parents=True,exist_ok=True)
    with (OPS/'campaign.lock').open('a') as campaign, (ROOT/'results/strict6-physical-v1/operations/campaign.lock').open('a') as support_campaign:
        fcntl.flock(campaign,fcntl.LOCK_EX|fcntl.LOCK_NB)
        fcntl.flock(support_campaign,fcntl.LOCK_EX|fcntl.LOCK_NB)
        spec={'protocol_sha256':PROTOCOL_SHA,'board_mac':BOARD,'query_ids':protocol['query_ids'],
            'acquisition_order':order,'prediction_lock_sha256':LOCK_SHA,
            'source_sha256':sha(Path(__file__)), 'analysis_source_sha256':frozen['source_hashes'],
            'scope':'Nine held-out models, evaluation only; no refitting or reselection',
            'selected_test_top1_percent':100*test_top1,'test_metrics_sha256':sha(test_path),
            'test_accuracy_is_not_a_selection_input':True}
        if (OPS/'spec.json').exists():assert read(OPS/'spec.json')==spec
        else:write_new(OPS/'spec.json',spec)
        for ident in order:
            directory,manifest=manifest_for(ident)
            missing=[(k,s) for k in ['memory','timing'] for s in range(3)
                     if not (BASE/ident/(k+'-'+str(s))/'receipt.json').exists()]
            load_seconds=None
            if missing:
                assert not any((BASE/ident/(k+'-'+str(s))).exists() for k,s in missing), 'incomplete session requires review'
                if (OPS/ident/'flash.json').exists():
                    old=read(OPS/ident/'flash.json');assert old['exit_code']==0
                    assert sha(OPS/ident/'flash.log')==old['log_sha256']
                    esptool(ident,'resume_verify_'+str(time.time_ns()),['verify_flash','0x0',str(directory/'flash_image.bin')])
                else:
                    esptool(ident,'identify',['flash_id'])
                    written=esptool(ident,'flash',['write_flash','--flash_mode','keep','--flash_freq','keep',
                        '--flash_size','keep','0x0',str(directory/'flash_image.bin')])
                    load_seconds=written['wall_seconds']
            for kind in ['memory','timing']:
                for session in range(3):
                    p=capture(ident,kind,session,load_seconds)
                    if load_seconds is not None and read(p).get('load_seconds')==load_seconds:load_seconds=None
            paths=[BASE/ident/(k+'-'+str(s))/'receipt.json' for k in ['memory','timing'] for s in range(3)]
            observed=collect(paths,[ident],False,after=frozen['created_unix'])['networks'][0]
            if ident==selected and not (BASE/'selected-model-check.json').exists():
                predicted=frozen['predictions'][ident]['structure_ridge_primary']
                write_new(BASE/'selected-model-check.json',{
                    'id':ident,'prediction_lock_sha256':LOCK_SHA,'observed':observed,'prediction':predicted,
                    'latency_ape_percent':100*abs(predicted['latency_ms']-observed['latency_ms'])/observed['latency_ms'],
                    'validation_top1':features[ident]['validation_top1'],'test_top1_percent':100*test_top1,
                    'test_top1_at_least_90_percent':test_top1>=.90,'functional_integer_mismatches':0,
                    'memory_constraints_pass': observed['psram_peak_bytes']<=6*2**20 and observed['internal_heap_peak_bytes']<=192*2**10,
                    'model_storage_constraint_pass':features[ident]['model_bytes']<=12*2**20,
                    'application_latency_sla_defined':False,'selection_unchanged':True})
            event('network_complete',id=ident,observed=observed)
        receipts=[BASE/i/(k+'-'+str(s))/'receipt.json' for i in protocol['query_ids'] for k in ['memory','timing'] for s in range(3)]
        assert len(list(BASE.glob('*/**/receipt.json')))==54
        collect(receipts,protocol['query_ids'],False,after=frozen['created_unix'])
        receipt_list=BASE/'query-receipts.json';paths=[str(p.relative_to(BASE)) for p in receipts]
        if receipt_list.exists():assert read(receipt_list)==paths
        else:write_new(receipt_list,paths)
        if not (BASE/'evaluation.json').exists():evaluate(LOCK,receipt_list,BASE/'evaluation.json',False)
        result=read(BASE/'evaluation.json')
        assert result['prediction_lock_sha256']==LOCK_SHA and not result['simulation_only']
        verify_lock(LOCK,False);assert sha(LOCK)==LOCK_SHA
        event('evaluation_complete',networks=9,receipts=54,metrics=result['metrics'],prediction_lock_unchanged=True)

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        if OPS.exists():event('stopped_for_review',error=repr(exc))
        raise
