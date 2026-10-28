"""Package only the independently audited, selected 15 networks; no deployment."""
import json
from pathlib import Path
import zipfile
import campaign as c


def main():
    root = c.ROOT / 'runs/nas-final-v1'
    selected = c.read(root / 'selected-networks.json')
    audit = c.read(root / 'final-audit.json')
    assert selected['status'] == 'locked' and selected['count'] == 15
    assert audit['passed'] is True and audit['count'] == 15
    assert {r['id'] for r in audit['models']} == {r['id'] for r in selected['networks']}
    files = [(root / name, name) for name in [
        'selected-networks.json', 'selected-networks.csv', 'final-audit.json', 'plan.json']]
    files.append((c.ROOT / 'configs/config.json', 'configs/config.json'))
    for index, state in enumerate(selected['networks'], 1):
        folder = root / 'final15' / ('%02d-%s' % (index, state['id']))
        assert c.digest(folder / 'model.tflite') == state['production_sha256']
        files.extend((path, str(path.relative_to(root))) for path in sorted(folder.rglob('*')) if path.is_file())
    target = root / 'flowers102-esp32s3-15-networks.zip'
    temporary = target.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for source, name in files:
            archive.write(source, name)
    with zipfile.ZipFile(temporary) as archive:
        assert archive.testzip() is None
        assert len([name for name in archive.namelist() if name.endswith('/model.tflite')]) == 15
    temporary.replace(target)
    receipt = {'file': target.name, 'sha256': c.digest(target), 'bytes': target.stat().st_size,
        'selected_manifest_sha256': c.digest(root / 'selected-networks.json'),
        'final_audit_sha256': c.digest(root / 'final-audit.json'), 'model_count': 15,
        'physical_board_flashed': False}
    c.write(root / 'delivery.json', receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
