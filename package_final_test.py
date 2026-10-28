"""Archive final-test evidence, excluding the large decoded image cache."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'runs/nas-final-test-v1'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    summary = json.loads((OUT/'summary.json').read_text())
    protocol = json.loads((OUT/'protocol.json').read_text())
    assert summary['status'] == 'complete' and summary['count'] == 15
    assert summary['selection_unchanged'] and summary['independent_saved_predictions_audit_passed']
    assert sha(ROOT/'runs/nas-final-v1/selected-networks.json') == summary['selection_sha256']
    assert sha(ROOT/'final_test.py') == protocol['code_sha256']
    files = [OUT/name for name in [
        'summary.json', 'test-metrics.csv', 'protocol.json', 'data-protocol.json',
        'evaluation-start.json', 'labels.npy',
    ]]
    files += sorted((OUT/'models').rglob('*.json')) + sorted((OUT/'models').rglob('*.npy'))
    files += sorted((OUT/'logs').glob('*.log'))
    assert len(list((OUT/'models').glob('*/int8.json'))) == 15
    assert len(list((OUT/'models').glob('*/fp32.json'))) == 15
    entries = {str(path.relative_to(OUT)): {'bytes': path.stat().st_size, 'sha256': sha(path)} for path in files}
    entries['source/final_test.py'] = {'bytes': (ROOT/'final_test.py').stat().st_size, 'sha256': sha(ROOT/'final_test.py')}
    assert 'test.npy' not in entries
    manifest = {'selection_sha256': summary['selection_sha256'], 'files': entries,
        'excluded': 'decoded test.npy image cache; original dataset is unchanged on server',
        'physical_board_accessed': False}
    (OUT/'artifact-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    archive = OUT/'flowers102-final-test15.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for path in files:
            bundle.write(path, str(path.relative_to(OUT)))
        bundle.write(ROOT/'final_test.py', 'source/final_test.py')
        bundle.write(OUT/'artifact-manifest.json', 'artifact-manifest.json')
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.testzip() is None
    receipt = {'archive': archive.name, 'bytes': archive.stat().st_size, 'sha256': sha(archive),
        'files': len(entries)+1, 'crc_passed': True, 'selection_sha256': summary['selection_sha256']}
    (OUT/'archive-receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
