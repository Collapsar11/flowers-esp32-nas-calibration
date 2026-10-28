"""Future serial acquisition only. Not invoked by server simulation; never flashes.

After separately authorized loading, open the named port and request a manual reset.
A query acquisition requires an already frozen PHYSICAL six-support prediction lock.
"""
import argparse
from pathlib import Path
import shutil
import time
import uuid
from measurement_io import PROTOCOL_SHA, design, parse_serial, read, sha, write_new
from strict6 import verify_lock


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--manifest',required=True);ap.add_argument('--port',required=True)
    ap.add_argument('--board-serial',required=True);ap.add_argument('--kind',choices=['timing','memory'],required=True)
    ap.add_argument('--session',type=int,choices=range(3),required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--prediction-lock');ap.add_argument('--timeout',type=float,default=1800)
    ap.add_argument('--load-seconds',type=float)
    args=ap.parse_args();protocol,features=design();manifest=read(args.manifest)
    assert args.load_seconds is None or args.load_seconds>=0
    assert manifest['protocol_sha256']==PROTOCOL_SHA
    assert manifest['production_sha256']==features[manifest['id']]['production_sha256']
    image=Path(args.manifest).parent/'flash_image.bin'
    assert sha(image)==manifest['artifact_hashes']['flash_image.bin'], 'firmware image differs from build manifest'
    if manifest['id'] in protocol['query_ids']:
        assert args.prediction_lock, 'freeze physical six-support predictions before query capture'
        lock=verify_lock(args.prediction_lock,False)
        assert lock['runtime_fingerprint']==manifest['runtime_fingerprint']
        assert time.time()>lock['created_unix']
    else:assert manifest['id'] in protocol['support_ids']
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(args.manifest,out/'manifest.json')
    # Import/open happens only here, never when importing or running QEMU tools.
    import serial
    port=serial.Serial(port=None,baudrate=115200,timeout=.2)
    port.dtr=False;port.rts=False;port.port=args.port
    nonce=uuid.uuid4().hex;data=b'';sent=False;started=time.time()
    print('Serial acquisition only. Manually reset/power-cycle the loaded board now; no flashing is performed.',flush=True)
    try:
        port.open()
        with (out/'serial.log').open('wb') as log:
            while time.time()-started<args.timeout:
                chunk=port.read(port.in_waiting or 1)
                if not chunk:continue
                log.write(chunk);log.flush();data+=chunk
                if not sent and b'FLOWERS_READY ' in data:
                    assert b'ESP-ROM:esp32s3' in data, 'reset after opening port; do not reuse a running session'
                    port.write(f'RUN {args.kind} {args.session} {nonce}\n'.encode());port.flush();sent=True
                if b'FLOWERS_MEASUREMENT_COMPLETE' in data:break
                if any(x in data for x in [b'Guru Meditation',b'abort() was called',b'assert failed']):break
    finally:port.close()
    parsed=parse_serial(data.decode(errors='replace'),manifest,args.kind,args.session,nonce)
    result={'source':'physical','transport':'serial','silicon_latency_valid':True,
        'port':args.port,'board_serial':args.board_serial,'cold_boot_requested':True,'rom_boot_seen':True,
        'kind':args.kind,'session':args.session,'nonce':nonce,'started_unix':started,'ended_unix':time.time(),
        'log_file':'serial.log','log_sha256':sha(out/'serial.log'),
        'manifest_file':'manifest.json','manifest_sha256':sha(out/'manifest.json'),'parsed':parsed,
        'loaded_image_sha256_declared':sha(image),'load_seconds':args.load_seconds,
        'provenance_limit':'operator-identified board and matching firmware banner; no cryptographic silicon attestation'}
    write_new(out/'receipt.json',result)
    print(f'Captured {manifest["id"]} {args.kind} session {args.session}: {out / "receipt.json"}')


if __name__=='__main__':main()
