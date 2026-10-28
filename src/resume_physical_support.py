"""Resume after a reviewed USB disconnect, retaining the interrupted attempt.

Uses the original frozen capture and calibration functions. Existing successful
receipts are never overwritten; every resumed loaded image is reverified.
"""
import fcntl
from pathlib import Path
import time
import run_physical_support as campaign
from measurement_io import read, sha, write_new, collect


def main():
    c = campaign
    with (c.OPS / 'campaign.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol = c.verify_initial()
        spec = read(c.OPS / 'spec.json')
        assert spec['source_sha256'] == sha(Path(c.__file__))
        recovery_number = len(list(c.OPS.glob('recovery-*.json'))) + 1
        interrupted = []
        for ident in protocol['support_ids']:
            for kind in ['memory', 'timing']:
                for session in range(3):
                    folder = c.BASE / ident / (kind + '-' + str(session))
                    if not (folder / 'failure.json').exists():
                        continue
                    assert not (folder / 'receipt.json').exists()
                    failure = read(folder / 'failure.json')
                    assert 'SerialException' in failure['error'] and 'Device not configured' in failure['error']
                    destination = c.ROOT / 'results/strict6-physical-interrupted-v1' / ident / (
                        kind + '-' + str(session) + '-attempt-' + str(recovery_number))
                    assert not destination.exists()
                    hashes = {p.name: sha(p) for p in folder.iterdir() if p.is_file()}
                    assert set(hashes) == {'manifest.json', 'serial.log', 'failure.json'}
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    folder.rename(destination)
                    interrupted.append({'id': ident, 'kind': kind, 'session': session,
                        'archive': str(destination.relative_to(c.ROOT)), 'files_sha256': hashes,
                        'failure': failure, 'included_in_calibration': False})
        assert interrupted, 'this recovery entry point requires a reviewed interrupted attempt'
        write_new(c.OPS / ('recovery-' + str(recovery_number) + '.json'), {
            'created_unix': time.time(), 'reason': 'USB serial device disconnected during acquisition and reappeared',
            'source_sha256': sha(Path(__file__)), 'capture_source_sha256': sha(Path(c.__file__)),
            'interrupted_attempts': interrupted, 'protocol_unchanged': True,
            'policy': 'Preserve interrupted evidence; verify device and full flashed image; acquire a fresh independent boot and nonce'})
        c.event('recovery_prepared', attempts_preserved=len(interrupted), recovery=recovery_number)
        for ident in protocol['support_ids']:
            directory, manifest = c.manifest_for(ident)
            missing = [(k, s) for k in ['memory', 'timing'] for s in range(3)
                if not (c.BASE / ident / (k + '-' + str(s)) / 'receipt.json').exists()]
            load_seconds = None
            if missing:
                previous_flash = c.OPS / ident / 'flash.json'
                if ident == protocol['support_ids'][0] or previous_flash.exists():
                    if previous_flash.exists():
                        assert read(previous_flash)['exit_code'] == 0
                    c.esptool(ident, 'verify_after_usb_reconnect_' + str(recovery_number),
                              ['verify_flash', '0x0', str(directory / 'flash_image.bin')])
                else:
                    c.esptool(ident, 'identify', ['flash_id'])
                    result = c.esptool(ident, 'flash', ['write_flash', '--flash_mode', 'keep',
                        '--flash_freq', 'keep', '--flash_size', 'keep', '0x0', str(directory / 'flash_image.bin')])
                    load_seconds = result['wall_seconds']
            for kind in ['memory', 'timing']:
                for session in range(3):
                    p = c.capture(ident, kind, session, load_seconds)
                    if load_seconds is not None and read(p).get('load_seconds') == load_seconds:
                        load_seconds = None
            c.event('network_complete', id=ident)
        receipts = [c.BASE / i / (k + '-' + str(s)) / 'receipt.json'
            for i in protocol['support_ids'] for k in ['memory', 'timing'] for s in range(3)]
        collect(receipts, protocol['support_ids'], False)
        listing = c.BASE / 'support-receipts.json'
        paths = [str(p.relative_to(c.BASE)) for p in receipts]
        if listing.exists():
            assert read(listing) == paths
        else:
            write_new(listing, paths)
        if not (c.BASE / 'calibration/lock.json').exists():
            c.calibrate(listing, c.BASE / 'calibration', False)
        frozen = c.verify_lock(c.BASE / 'calibration/lock.json', False)
        assert frozen['physical_calibration_network_count'] == 6 and not frozen['query_labels_accessed']
        assert len(list(c.BASE.glob('*/**/receipt.json'))) == 36
        for ident in protocol['query_ids']:
            assert not (c.BASE / ident).exists()
        c.event('predictions_locked', support_networks=6, physical_query_networks=0,
                lock_sha256=sha(c.BASE / 'calibration/lock.json'), predicted_networks=len(frozen['predictions']))


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        campaign.event('stopped_for_review', error=repr(exc))
        raise
