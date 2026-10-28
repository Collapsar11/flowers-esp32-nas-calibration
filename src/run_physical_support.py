"""Authorized six-support-only board campaign; never flashes or measures query IDs.

Uses the unchanged firmware and frozen analysis. Each new acquisition starts with
an EN hardware reset via the USB-UART RTS line, using esptool's HardReset.
Existing valid receipts are retained; incomplete attempts require explicit review.
"""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

import serial
from esptool.reset import HardReset
from measurement_io import ROOT, PROTOCOL_SHA, design, read, read_receipt, sha, write_new, parse_serial, collect
from strict6 import calibrate, verify_lock

BASE=ROOT/'results/strict6-physical-v1'
KIT=ROOT/'results/strict6-simulation-v3-kit/runs/strict6-simulation-v3/models'
PILOT=ROOT/'results/esp32-first-board-v1'
PORT=os.environ['FLOWERS_SERIAL_PORT']
BOARD=os.environ['FLOWERS_BOARD_MAC']
OPS=BASE/'operations'


def event(stage, **values):
    record={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'unix':time.time(),'stage':stage,**values}
    print(json.dumps(record),flush=True)
    with (OPS/'events.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
    temp=OPS/'status.tmp';temp.write_text(json.dumps(record,indent=2)+'\n');temp.replace(OPS/'status.json')


def manifest_for(ident):
    protocol,features=design();assert ident in protocol['support_ids'], 'query acquisition is prohibited'
    directory=KIT/ident; manifest=read(directory/'manifest.json')
    assert manifest['id']==ident and manifest['protocol_sha256']==PROTOCOL_SHA
    assert manifest['production_sha256']==features[ident]['production_sha256']
    assert sha(directory/'flash_image.bin')==manifest['artifact_hashes']['flash_image.bin']
    return directory,manifest


def esptool(ident, label, arguments):
    assert ident in design()[0]['support_ids']
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
        'capture_source_sha256':sha(Path(__file__)),
        'provenance_limit':'Board MAC verified through esptool and matching firmware banner; no cryptographic attestation'}
    write_new(receipt,result);read_receipt(receipt)
    summary={'id':ident,'kind':kind,'session':session,'receipt':str(receipt.relative_to(ROOT))}
    if kind=='timing':summary['median_ms']=parsed['median_ms']
    else:summary['psram_peak_bytes']=parsed['memory']['psram_peak_bytes']
    event('capture_complete',**summary)
    return receipt


def verify_initial():
    original=read(PILOT/'original-backup.json')
    assert original['verified_against_device'] and sha(PILOT/original['file'])==original['sha256']
    previous=read(PILOT/'session.json')
    for name,digest in previous['source_sha256'].items():assert sha(ROOT/name)==digest
    protocol,_=design()
    for ident in protocol['support_ids']:manifest_for(ident)
    for path in BASE.glob('*/**/receipt.json'):
        r,m=read_receipt(path);assert m['id'] in protocol['support_ids'] and r['source']=='physical'
    return protocol


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--pilot-memory-one',action='store_true');args=ap.parse_args()
    OPS.mkdir(parents=True,exist_ok=True)
    with (OPS/'campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        protocol=verify_initial();first=protocol['support_ids'][0]
        spec={'protocol_sha256':PROTOCOL_SHA,'board_mac':BOARD,'support_ids':protocol['support_ids'],
            'query_ids_prohibited':protocol['query_ids'],'source_sha256':sha(Path(__file__)),
            'analysis_source_sha256':{n:sha(ROOT/n) for n in ['src/measurement_io.py','src/strict6.py']},
            'scope':'Exactly six support networks; no query flashing, capture, or evaluation'}
        if (OPS/'spec.json').exists():assert read(OPS/'spec.json')==spec
        else:write_new(OPS/'spec.json',spec)
        if args.pilot_memory_one:
            capture(first,'memory',1)
            event('automatic_reset_pilot_passed',id=first)
            return
        for ident in protocol['support_ids']:
            directory,manifest=manifest_for(ident)
            missing=[(k,s) for k in ['memory','timing'] for s in range(3) if not (BASE/ident/(k+'-'+str(s))/'receipt.json').exists()]
            load_seconds=None
            if missing:
                if ident==first:
                    esptool(ident,'verify_existing_flash',['verify_flash','0x0',str(directory/'flash_image.bin')])
                else:
                    esptool(ident,'identify',['flash_id'])
                    written=esptool(ident,'flash',['write_flash','--flash_mode','keep','--flash_freq','keep',
                        '--flash_size','keep','0x0',str(directory/'flash_image.bin')])
                    load_seconds=written['wall_seconds']
            for kind in ['memory','timing']:
                for session in range(3):
                    p=capture(ident,kind,session,load_seconds)
                    if load_seconds is not None and read(p).get('load_seconds')==load_seconds:load_seconds=None
            event('network_complete',id=ident)
        receipts=[BASE/i/(k+'-'+str(s))/'receipt.json' for i in protocol['support_ids'] for k in ['memory','timing'] for s in range(3)]
        collect(receipts,protocol['support_ids'],False)
        receipt_list=BASE/'support-receipts.json';paths=[str(p.relative_to(BASE)) for p in receipts]
        if receipt_list.exists():assert read(receipt_list)==paths
        else:write_new(receipt_list,paths)
        if not (BASE/'calibration/lock.json').exists():calibrate(receipt_list,BASE/'calibration',False)
        frozen=verify_lock(BASE/'calibration/lock.json',False)
        assert frozen['physical_calibration_network_count']==6 and not frozen['query_labels_accessed']
        assert len(list(BASE.glob('*/**/receipt.json')))==36
        for ident in protocol['query_ids']:assert not (BASE/ident).exists()
        event('predictions_locked',support_networks=6,physical_query_networks=0,
              lock_sha256=sha(BASE/'calibration/lock.json'),predicted_networks=len(frozen['predictions']))


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        if OPS.exists():event('stopped_for_review',error=repr(exc))
        raise
