"""Package audited simulation evidence and six support images; never access hardware."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile
from measurement_io import ROOT, design, read, sha, write_new


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',required=True)
    ap.add_argument('--out',required=True)
    a=ap.parse_args(); run=Path(a.run).resolve(); out=Path(a.out).resolve()
    audit=read(run/'audit.json'); assert audit['passed'] and audit['simulation_only']
    assert audit['physical_measurements']==0
    protocol,_=design(); files={}
    def add(path, name=None):
        path=Path(path); assert path.is_file()
        archive_name=name or str(path.relative_to(ROOT))
        assert archive_name not in files
        files[archive_name]=path
    for name in ['README.md','src/measurement_io.py','src/strict6.py','src/measure_campaign.py',
        'src/run_measurement_simulation.py','src/capture_measurement.py','tests/test_strict6.py',
        'src/audit_measurement_simulation.py','src/package_measurement_kit.py',
        'scripts/build/build_measure_firmware.sh','requirements/measurement.txt']:
        add(ROOT/name)
    for name in ['CMakeLists.txt','sdkconfig.defaults','partitions.csv','main/CMakeLists.txt',
        'main/idf_component.yml','main/app_main.cpp','main/embedded.S.in']:
        add(ROOT/'firmware-measure'/name)
    for name in ['protocol.json','public-features.json','decision-rule-details.json']:
        source=ROOT/'runs/strict6-esp32-v1'/name
        add(source,'results/strict6-esp32-v1/'+name)
    for p in sorted(run.glob('*.json')):add(p)
    for p in sorted((run/'calibration').glob('*.json')):add(p)
    for name in ['unit-tests.log','capture-help.txt']:
        add(run/name)
    for ident in protocol['support_ids']+protocol['query_ids']:
        model=run/'models'/ident; manifest=read(model/'manifest.json')
        add(model/'manifest.json')
        for name in ['size-report.txt','compiler-version.txt','sdkconfig','dependencies.lock','measurement_config.h']:
            assert sha(model/name)==manifest['artifact_hashes'][name]; add(model/name)
        if ident in protocol['support_ids']:
            assert sha(model/'flash_image.bin')==manifest['artifact_hashes']['flash_image.bin']
            add(model/'flash_image.bin')
        for p in sorted((model/'qemu').glob('*/*')):
            if p.name in ['receipt.json','serial.log']:add(p)
    contents={'simulation_only':True,'physical_board_accessed':False,
        'support_firmware_images':protocol['support_ids'],'query_firmware_images_included':False,
        'full_build_artifacts_location':str(run),
        'scope':'All 90 accepted boot logs and receipts, frozen predictions and audit; six support flash images. Full ELF/map/app binaries and nine query flash images remain on the server. Toolchains, dataset and production model source bundle are not included.',
        'files':{name:{'sha256':sha(path),'bytes':path.stat().st_size} for name,path in sorted(files.items())}}
    out.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(out,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for name,path in sorted(files.items()):archive.write(path,name)
        archive.writestr('kit-manifest.json',json.dumps(contents,indent=2)+'\n')
    with zipfile.ZipFile(out) as archive:assert archive.testzip() is None
    # Prove that relative evidence paths survive extraction away from the server tree.
    with tempfile.TemporaryDirectory(prefix='flowers-measure-kit-') as tmp:
        extracted=Path(tmp)
        with zipfile.ZipFile(out) as archive:archive.extractall(extracted)
        replay_run=extracted/run.relative_to(ROOT)
        replay=extracted/'replayed-evaluation.json'
        subprocess.run([sys.executable,str(extracted/'src/strict6.py'),'evaluate','--simulation',
            '--lock',str(replay_run/'calibration/lock.json'),
            '--receipts',str(replay_run/'query-receipts.json'),'--out',str(replay)],check=True)
        expected=read(run/'evaluation.json'); actual=read(replay)
        for key in ['metrics','memory','decisions','prediction_lock_sha256']:
            assert actual[key]==expected[key],key
    report={'archive':out.name,'sha256':sha(out),'bytes':out.stat().st_size,
        'files':len(files)+1,'crc_verified':True,'portable_evaluation_replay_passed':True,
        'simulation_only':True,'physical_board_accessed':False}
    write_new(out.with_suffix('.json'),report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
