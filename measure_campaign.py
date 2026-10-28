"""Compile immutable measurement firmware and execute QEMU only. Never flashes."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import re
import selectors
import shutil
import subprocess
import sys
import time
import uuid
from measurement_io import ROOT, PROTOCOL_SHA, design, read, sha, write_new, parse_serial


def build(ident, out):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():
        m=read(out/'manifest.json')
        assert m['id']==ident and m['protocol_sha256']==PROTOCOL_SHA
        for name,digest in m['artifact_hashes'].items(): assert sha(out/name)==digest
        assert m['firmware_source_sha256']==sha(ROOT/'firmware-measure/main/app_main.cpp')
        assert m['runtime']['embedding_template_sha256']==sha(ROOT/'firmware-measure/main/embedded.S.in')
        assert m['runtime']['component_cmake_sha256']==sha(ROOT/'firmware-measure/main/CMakeLists.txt')
        return m
    protocol, features=design(); feature=features[ident]
    selected=ROOT/'runs/nas-final-v1/selected-networks.json'
    assert sha(selected)==protocol['selection_sha256']
    state=next(r for r in read(selected)['networks'] if r['id']==ident)
    source=ROOT/'runs'/state['run']/state['quant_directory']
    assert sha(source/'model.tflite')==feature['production_sha256']
    import numpy as np
    golden=np.load(source/'golden-int8.npz'); gp=read(source/'golden-protocol.json')
    assert gp['split']=='val' and gp['count']==102 and len(golden['inputs'])==102
    orders=[]
    for session in range(3):
        order=list(range(102)); random.Random(20261028+session).shuffle(order); orders.append(order)
    used=state['tensor_arena_bytes']; reserved=((used+15)//16)*16+4096
    with (ROOT/'logs/measurement-build.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        fw=ROOT/'firmware-measure'
        shutil.copyfile(source/'model.tflite',fw/'main/model.tflite')
        for key,name in [('inputs','golden-inputs.bin'),('outputs','golden-outputs.bin')]:
            golden[key].tofile(fw/'main'/name)
            assert sha(fw/'main'/name)==gp['golden_'+key+'_sha256']
        header = '\n'.join([
            '#pragma once', '#define MODEL_ID '+json.dumps(ident),
            '#define MODEL_SHA256 '+json.dumps(feature['production_sha256']),
            '#define PROTOCOL_SHA256 '+json.dumps(PROTOCOL_SHA),
            '#define INPUT_SHA256 '+json.dumps(gp['golden_inputs_sha256']),
            '#define FROZEN_ARENA_USED '+str(used),
            'static const unsigned char input_order[3][102] = '+json.dumps(orders).replace('[','{').replace(']','}')+';', ''])
        (fw/'main/measurement_config.h').write_text(header)
        # Pin the exact dependency version already validated in the NAS campaign.
        if not (fw/'dependencies.lock').exists(): shutil.copyfile(ROOT/'firmware-tflite/dependencies.lock',fw/'dependencies.lock')
        if not (fw/'managed_components').exists(): shutil.copytree(ROOT/'firmware-tflite/managed_components',fw/'managed_components')
        started=time.time()
        if (out/'build.log').exists():
            shutil.copyfile(out/'build.log',out/('build-attempt-'+uuid.uuid4().hex+'.log'))
        with (out/'build.log').open('w') as log:
            subprocess.run(['bash','build_measure_firmware.sh'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        for name in ['flash_image.bin','flowers_measure.elf','flowers_measure.bin','flowers_measure.map','size-report.txt','compiler-version.txt']:
            shutil.copyfile(fw/'build'/name,out/name)
        for name in ['sdkconfig','dependencies.lock']: shutil.copyfile(fw/name,out/name)
        (out/'measurement_config.h').write_text(header)
        binary=(out/'flowers_measure.bin').read_bytes()
        assert binary.count((source/'model.tflite').read_bytes())==1
        assert golden['inputs'].tobytes() in binary and golden['outputs'].tobytes() in binary
        symbols=subprocess.check_output(['readelf','-sW',str(out/'flowers_measure.elf')],text=True)
        addresses={}
        for name in ['_binary_model_tflite_start','_binary_golden_inputs_bin_start','_binary_golden_outputs_bin_start']:
            match=re.search(r'\d+:\s+([0-9a-fA-F]+).*\s'+name+r'$',symbols,re.M)
            assert match is not None
            addresses[name]=int(match[1],16)
            assert addresses[name]%16==0, 'misaligned embedded data'
        config=(out/'sdkconfig').read_text()
        for flag in ['CONFIG_ESP_MAIN_TASK_AFFINITY_CPU1=y','CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ_240=y',
            'CONFIG_SPIRAM_MODE_OCT=y','CONFIG_SPIRAM_SPEED_80M=y','CONFIG_ESPTOOLPY_FLASHFREQ_80M=y',
            '# CONFIG_PM_ENABLE is not set']:
            assert flag in config, flag
        sdk_commit=subprocess.check_output(['git','-C','upstream/esp-idf','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        assert sdk_commit==protocol['target']['esp_idf_commit']
        runtime={'sdkconfig_sha256':sha(out/'sdkconfig'),'dependencies_sha256':sha(out/'dependencies.lock'),
            'idf_commit':sdk_commit,'tflm_commit':subprocess.check_output(['git','-C','upstream/esp-tflite-micro','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'compiler_sha256':sha(out/'compiler-version.txt'),'firmware_source_sha256':sha(fw/'main/app_main.cpp'),
            'embedding_template_sha256':sha(fw/'main/embedded.S.in'), 'component_cmake_sha256':sha(fw/'main/CMakeLists.txt')}
        manifest={'id':ident,'protocol_sha256':PROTOCOL_SHA,'production_sha256':feature['production_sha256'],
            'golden_inputs_sha256':gp['golden_inputs_sha256'],'golden_outputs_sha256':gp['golden_outputs_sha256'],
            'input_order':orders,'input_order_algorithm':'Python random.Random(20261028+session).shuffle(range(102))',
            'arena_used_bytes':used,'arena_reserved_bytes':reserved,'runtime':runtime,
            'runtime_fingerprint':hashlib.sha256(json.dumps(runtime,sort_keys=True).encode()).hexdigest(),
            'firmware_source_sha256':runtime['firmware_source_sha256'],'build_seconds':time.time()-started,
            'embedded_symbol_addresses':addresses,'embedded_alignment_bytes':16,
            'production_model_bytes':(source/'model.tflite').stat().st_size,
            'benchmark_firmware_bytes':len(binary),'golden_payload_bytes':golden['inputs'].nbytes+golden['outputs'].nbytes,
            'flash_scope':'benchmark image includes golden fixtures; not production application flash usage',
            'physical_board_accessed':False,
            'artifact_hashes':{p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()}}
        write_new(out/'manifest.json',manifest)
        return manifest


def qemu_session(modeldir, kind, session, timeout=1800):
    modeldir=Path(modeldir); manifest=read(modeldir/'manifest.json')
    folder=modeldir/'qemu'/f'{kind}-{session}'; folder.mkdir(parents=True,exist_ok=True)
    receipt=folder/'receipt.json'
    if receipt.exists():
        from measurement_io import read_receipt
        read_receipt(receipt); return str(receipt)
    if (folder/'serial.log').exists():
        archive=folder/('failed-attempt-'+uuid.uuid4().hex)
        archive.mkdir()
        for name in ['serial.log','failure.json']:
            if (folder/name).exists():shutil.move(str(folder/name),str(archive/name))
    nonce=uuid.uuid4().hex
    binary=ROOT/'toolchains/qemu-20260417/qemu/bin/qemu-system-xtensa'
    cmd=[str(binary),'-nographic','-machine','esp32s3','-m','8M',
        '-global','driver=ssi_psram,property=is_octal,value=true',
        '-global','driver=timer.esp32c3.timg,property=wdt_disable,value=true',
        '-snapshot','-drive',f'file={modeldir.resolve()}/flash_image.bin,if=mtd,format=raw']
    env=dict(os.environ,LD_LIBRARY_PATH=str(ROOT/'toolchains/system-libs/root/usr/lib/x86_64-linux-gnu')+':'+os.environ.get('LD_LIBRARY_PATH',''))
    started=time.time(); proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env=env)
    selector=selectors.DefaultSelector(); selector.register(proc.stdout,selectors.EVENT_READ)
    data=b''; sent=False; reason='timeout'; spontaneous_exit=None
    try:
        with (folder/'serial.log').open('wb') as log:
            while time.time()-started<timeout:
                if not selector.select(timeout=1): continue
                chunk=proc.stdout.read1(65536)
                if not chunk:
                    spontaneous_exit=proc.poll(); reason='process_exited'; break
                log.write(chunk); log.flush(); data+=chunk
                if not sent and b'FLOWERS_READY ' in data:
                    proc.stdin.write(f'RUN {kind} {session} {nonce}\n'.encode()); proc.stdin.flush(); sent=True
                if b'FLOWERS_MEASUREMENT_COMPLETE' in data:reason='complete';break
                if any(s in data for s in [b'Guru Meditation',b'abort() was called',b'assert failed',b'Assertion failed']):reason='firmware_error';break
    finally:
        proc.terminate()
        try:proc.wait(timeout=5)
        except subprocess.TimeoutExpired:proc.kill();proc.wait()
        selector.close()
    try:parsed=parse_serial(data.decode(errors='replace'),manifest,kind,session,nonce)
    except Exception as exc:
        write_new(folder/'failure.json',{'reason':reason,'spontaneous_exit':spontaneous_exit,
            'exit_after_cleanup':proc.returncode,'elapsed':time.time()-started,'error':repr(exc),'nonce':nonce})
        raise
    result={'source':'qemu','transport':'qemu_subprocess','silicon_latency_valid':False,
        'kind':kind,'session':session,'nonce':nonce,'started_unix':started,'ended_unix':time.time(),
        'log_file':'serial.log','log_sha256':sha(folder/'serial.log'),
        'manifest_file':'../../manifest.json','manifest_sha256':sha(modeldir/'manifest.json'),
        'qemu_command':cmd,'qemu_binary_sha256':sha(binary),'parsed':parsed}
    write_new(receipt,result)
    print(json.dumps({'complete':manifest['id'],'mode':kind,'session':session,'source':'qemu'}),flush=True)
    return str(receipt)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('command',choices=['build','qemu']);ap.add_argument('--id',required=True)
    ap.add_argument('--out',required=True);ap.add_argument('--kind',choices=['timing','memory']);ap.add_argument('--session',type=int,choices=range(3))
    args=ap.parse_args()
    if args.command=='build':build(args.id,Path(args.out))
    else:
        if args.kind is None or args.session is None:ap.error('qemu requires --kind and --session')
        assert read(Path(args.out)/'manifest.json')['id']==args.id
        qemu_session(Path(args.out),args.kind,args.session)


if __name__=='__main__':main()
