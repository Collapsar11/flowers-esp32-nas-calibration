"""Strict acquisition records shared by simulation and future serial capture."""
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
if not __debug__:
    raise RuntimeError('Run without -O: experimental integrity checks must stay enabled')
PROTOCOL_SHA = 'cf36aec24cc79afa03424f708798180b6512b61501239ffe0ccee7ccc44e638b'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidental rewriting of locks or evidence.
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def design():
    base = ROOT / 'results/strict6-esp32-v1'
    if not base.exists():
        base = ROOT / 'runs/strict6-esp32-v1'
    assert sha(base/'protocol.json') == PROTOCOL_SHA, 'frozen protocol changed'
    protocol = read(base/'protocol.json')
    assert sha(base/'public-features.json') == protocol['public_features_sha256']
    assert read(base/'decision-rule-details.json')['protocol_sha256'] == PROTOCOL_SHA
    rows = read(base/'public-features.json')['networks']
    return protocol, {r['id']: r for r in rows}


def parse_serial(text, manifest, kind, session, nonce):
    assert 'FLOWERS_MEASUREMENT_COMPLETE' in text, 'incomplete cold-boot session'
    assert not any(x in text for x in ['Guru Meditation', 'abort() was called', 'assert failed', 'Assertion failed'])
    records = [json.loads(line.split('FLOWERS_JSON ',1)[1]) for line in text.splitlines() if 'FLOWERS_JSON ' in line]
    assert records[0]['event'] == 'begin' and records[-1]['event'] == 'done'
    begin, done = records[0], records[-1]
    assert sum(r['event']=='begin' for r in records)==sum(r['event']=='done' for r in records)==1
    for key, value in {'id':manifest['id'], 'model_sha256':manifest['production_sha256'],
        'protocol_sha256':PROTOCOL_SHA, 'input_sha256':manifest['golden_inputs_sha256'],
        'kind':kind, 'session':session, 'nonce':nonce, 'core':1, 'cpu_hz':240000000,
        'psram_bytes':8388608, 'psram_mhz':80, 'flash_mhz':80, 'hardware_source':'unattested'}.items():
        assert begin[key] == value, (key, begin.get(key), value)
    count = 204 if kind=='timing' else 102
    assert done['count'] == done['class_agreement'] == count and done['integer_mismatches']==0
    assert done['warmup']==(10 if kind=='timing' else 0)
    assert done['arena_used']==manifest['arena_used_bytes'] and done['arena_reserved']==manifest['arena_reserved_bytes']
    assert 0 <= done['stack_free_after_bytes'] <= done['stack_free_before_bytes'] <= 16384
    result = {'begin':begin,'done':done}
    if kind == 'timing':
        invokes = records[1:-1]
        assert len(invokes)==204
        order = manifest['input_order'][session]
        for i, row in enumerate(invokes):
            assert row['event']=='invoke' and row['n']==i and row['input']==order[i%102]
            assert type(row['duration_us']) is int and row['duration_us']>0
        duration = [r['duration_us']/1000 for r in invokes]
        result.update(duration_ms=duration, median_ms=statistics.median(duration),
            mean_ms=statistics.mean(duration), p95_ms=sorted(duration)[math.ceil(.95*len(duration))-1])
    else:
        assert len(records)==3 and records[1]['event']=='memory'
        mem = records[1]
        for k, prefix in [('psram_peak_bytes','psram'),('internal_heap_peak_bytes','internal')]:
            assert type(mem[k]) is int and mem[k]>=0
            assert mem[k]==mem[prefix+'_before']-mem[prefix+'_min_free']
        assert mem['psram_peak_bytes']>=done['arena_reserved']
        result['memory']=mem
    return result


def read_receipt(path):
    path = Path(path)
    receipt = read(path)
    assert receipt['source'] in ['qemu','physical']
    log = path.parent/receipt['log_file']
    manifest_path = path.parent/receipt['manifest_file']
    assert sha(log)==receipt['log_sha256'] and sha(manifest_path)==receipt['manifest_sha256']
    manifest = read(manifest_path)
    assert manifest['protocol_sha256']==PROTOCOL_SHA
    assert manifest['embedded_alignment_bytes']==16
    assert set(manifest['embedded_symbol_addresses'])=={'_binary_model_tflite_start','_binary_golden_inputs_bin_start','_binary_golden_outputs_bin_start'}
    assert all(value%16==0 for value in manifest['embedded_symbol_addresses'].values())
    assert receipt['ended_unix'] >= receipt['started_unix'] > 0
    parsed = parse_serial(log.read_text(errors='replace'), manifest, receipt['kind'], receipt['session'], receipt['nonce'])
    assert parsed==receipt['parsed'], 'receipt does not match raw serial log'
    if receipt['source']=='physical':
        assert receipt['transport']=='serial' and receipt['board_serial'] and receipt['port']
        assert receipt['cold_boot_requested'] is True and receipt['rom_boot_seen'] is True
        assert 'ESP-ROM:esp32s3' in log.read_text(errors='replace'), 'physical session lacks reset/boot evidence'
    else:
        assert receipt['transport']=='qemu_subprocess' and receipt['silicon_latency_valid'] is False
    return receipt, manifest


def collect(receipt_paths, expected_ids, simulation=False, after=None):
    protocol, features = design()
    records = {}; nonces=set(); config_hashes=set(); source='qemu' if simulation else 'physical'
    provenance=[]
    for path in receipt_paths:
        receipt, manifest = read_receipt(path)
        ident=manifest['id']
        assert ident in expected_ids and receipt['source']==source, 'wrong architecture or data source'
        assert manifest['production_sha256']==features[ident]['production_sha256']
        assert receipt['nonce'] not in nonces, 'duplicate boot nonce'
        nonces.add(receipt['nonce'])
        if after is not None:
            assert receipt['started_unix']>after, 'query acquisition began before prediction lock'
        key=(receipt['kind'],receipt['session'])
        assert key not in records.setdefault(ident,{}), 'duplicate cold-boot session'
        records[ident][key]=receipt
        config_hashes.add(manifest['runtime_fingerprint'])
        provenance.append({'path':str(Path(path).resolve()), 'sha256':sha(path)})
    assert set(records)==set(expected_ids) and len(config_hashes)==1, 'missing IDs or mixed runtime configuration'
    output=[]
    for ident in expected_ids:
        sessions=records[ident]
        assert set(sessions)=={(k,s) for k in ['timing','memory'] for s in range(3)}, 'requires three independent boots per mode'
        times=[sessions['timing',s]['parsed']['median_ms'] for s in range(3)]
        output.append({'id':ident,'latency_ms':statistics.median(times), 'session_medians_ms':times,
            **{k:max(sessions['memory',s]['parsed']['memory'][k] for s in range(3)) for k in ['psram_peak_bytes','internal_heap_peak_bytes']}})
    return {'source':source,'silicon_latency_valid':not simulation, 'protocol_sha256':PROTOCOL_SHA,
        'runtime_fingerprint':next(iter(config_hashes)), 'networks':output, 'receipts':provenance}
