"""One final test pass on immutable selected networks; never tunes or selects."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import csv
import fcntl
import hashlib
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'runs/nas-final-test-v1'
SELECTED = ROOT / 'runs/nas-final-v1/selected-networks.json'

def read(path):
    return json.loads(path.read_text())

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024**2), b''):
            h.update(b)
    return h.hexdigest()

def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2) + '\n')
    tmp.replace(path)

def metrics(pred, labels):
    import numpy as np
    counts = np.bincount(labels, minlength=102)
    correct = np.bincount(labels[pred == labels], minlength=102)
    assert pred.shape == labels.shape == (6149,) and np.all(counts > 0)
    return {'count': len(labels), 'correct': int((pred == labels).sum()),
        'top1': float(np.mean(pred == labels)), 'mean_class_accuracy': float(np.mean(correct / counts)),
        'per_class_count': counts.tolist(), 'per_class_correct': correct.tolist()}

def state(ident):
    manifest = read(SELECTED)
    assert sha(SELECTED) == read(OUT / 'protocol.json')['selection_sha256']
    return next(s for s in manifest['networks'] if s['id'] == ident)

def prepare():
    import numpy as np
    import pipeline as p
    ds = p.dataset('test', augment=False, download=False)
    assert len(ds) == 6149
    known = read(ROOT / 'logs/dataset.json')
    ids = [x.name for x in ds._image_files]
    assert ids == known['ids']['test']
    assert not set(ids).intersection(known['ids']['train'] + known['ids']['val'])
    # Prove that this evaluation transform matches the existing validation cache.
    old = np.load(ROOT / 'runs/pilot-ofa-min-v1/tflite-data/val.npy', mmap_mode='r')
    val = p.dataset('val')
    for i in [0, 510, 1019]:
        np.testing.assert_array_equal(val[i][0].permute(1, 2, 0).numpy(), old[i])
    array = np.lib.format.open_memmap(OUT / 'test.npy', mode='w+', dtype=np.float32, shape=(6149,160,160,3))
    labels = []
    for i in range(len(ds)):
        x,y = ds[i]
        array[i] = x.permute(1,2,0).numpy()
        labels.append(y)
        if (i+1) % 1020 == 0:
            print(json.dumps({'prepared':i+1,'count':6149}),flush=True)
    array.flush()
    np.save(OUT / 'labels.npy', np.asarray(labels,dtype=np.int64))
    write(OUT / 'data-protocol.json', {'count':6149,'split':'test','image_ids':ids,
        'class_counts':np.bincount(labels,minlength=102).tolist(),
        'preprocessing':'Resize(183), CenterCrop(160), ToTensor, ImageNet Normalize; NHWC float32; no augmentation',
        'test_array_sha256':sha(OUT / 'test.npy'),'labels_sha256':sha(OUT / 'labels.npy'),
        'dataset_manifest_sha256':sha(ROOT / 'logs/dataset.json'),
        'metadata_sha256':known['metadata_sha256'],'validation_preprocessing_parity':True})

def integer(ident):
    import numpy as np
    import tensorflow as tf
    s = state(ident)
    model = ROOT / 'runs' / s['run'] / s['quant_directory'] / 'model.tflite'
    assert sha(model) == s['production_sha256']
    folder = OUT / 'models' / ident
    folder.mkdir(parents=True,exist_ok=True)
    interpreter = tf.lite.Interpreter(model_path=str(model),num_threads=1,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF)
    interpreter.allocate_tensors()
    inp, out = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    assert inp['dtype'] == out['dtype'] == np.int8
    assert list(inp['shape']) == [1,160,160,3] and list(out['shape']) == [1,102]
    assert not any(np.issubdtype(t['dtype'],np.floating) for t in interpreter.get_tensor_details())
    x = np.load(OUT / 'test.npy',mmap_mode='r')
    labels = np.load(OUT / 'labels.npy')
    outputs = np.empty((6149,102),dtype=np.int8)
    scale, zero = inp['quantization']
    for i, image in enumerate(x):
        # Identical to the frozen QAT checkpoint selection evaluator.
        qx = np.clip(np.rint(image[None] / scale) + zero,-128,127).astype(np.int8)
        interpreter.set_tensor(inp['index'],qx)
        interpreter.invoke()
        outputs[i] = interpreter.get_tensor(out['index'])[0]
        if (i+1) % 1020 == 0:
            print(json.dumps({'id':ident,'integer_test_progress':i+1,'count':6149}),flush=True)
    assert sha(model) == s['production_sha256']
    pred = outputs.argmax(1)
    np.save(folder / 'int8-outputs.npy',outputs)
    np.save(folder / 'int8-predictions.npy',pred)
    write(folder / 'int8.json',{'id':ident,**metrics(pred,labels),'production_sha256':s['production_sha256'],
        'predictions_sha256':sha(folder / 'int8-predictions.npy'),'outputs_sha256':sha(folder / 'int8-outputs.npy'),
        'runtime':'TFLite BUILTIN_REF','tensorflow':tf.__version__,'num_threads':1,
        'input_rounding':'clip(rint(x/scale)+zero,-128,127)','test_used_for_selection':False})

def floating(gpu):
    import numpy as np
    import torch
    import pipeline as p
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    assert torch.cuda.is_available()
    x = np.load(OUT / 'test.npy',mmap_mode='r')
    labels = np.load(OUT / 'labels.npy')
    for saved in read(SELECTED)['networks']:
        s = state(saved['id'])
        folder = OUT / 'models' / s['id']
        folder.mkdir(parents=True,exist_ok=True)
        if (folder / 'fp32.json').exists():
            assert read(folder / 'fp32.json')['checkpoint_sha256'] == sha(ROOT/'runs'/s['run']/'best.pt')
            continue
        run = ROOT/'runs'/s['run']
        expected = read(run/'summary.json')['checkpoint_sha256']
        assert sha(run/'best.pt') == expected
        saved = torch.load(run/'best.pt',map_location='cpu',weights_only=False)
        assert saved['arch'] == s['arch']
        net = p.build_model(saved['model'],saved['arch'],pretrained=False).eval().cuda()
        net.load_state_dict(saved['state_dict'],strict=True)
        outputs = []
        with torch.inference_mode():
            for i in range(0,len(x),32):
                batch = torch.from_numpy(np.array(x[i:i+32],copy=True)).permute(0,3,1,2).contiguous().cuda()
                outputs.append(net(batch).cpu().numpy())
        logits = np.concatenate(outputs)
        pred = logits.argmax(1)
        np.save(folder/'fp32-logits.npy',logits)
        np.save(folder/'fp32-predictions.npy',pred)
        write(folder/'fp32.json',{'id':s['id'],**metrics(pred,labels),'checkpoint_sha256':expected,
            'predictions_sha256':sha(folder/'fp32-predictions.npy'),'logits_sha256':sha(folder/'fp32-logits.npy'),
            'torch':torch.__version__,'precision':'FP32, no autocast, TF32 disabled','test_used_for_selection':False})
        print(json.dumps({'fp32_complete':s['id'],'top1':float(np.mean(pred==labels))}),flush=True)
        del net, saved, outputs, logits
        torch.cuda.empty_cache()

def command(mode, cpus, gpu=-1, ident=None):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1',TF_NUM_INTRAOP_THREADS='1',TF_NUM_INTEROP_THREADS='1',
        FLOWERS_PRESERVE_AFFINITY='1',FLOWERS_DATA_WORKERS='0',TF_USE_LEGACY_KERAS='1')
    py = ROOT / ('.tflite-venv/bin/python' if mode=='integer' else '.venv/bin/python')
    cmd = [str(ROOT/'.venv/bin/python'),'supervise.py','--cpus',','.join(map(str,cpus)),
        '--descendants','--',str(py),'-u',__file__,mode]
    if ident:
        cmd += ['--id',ident]
    log = OUT/'logs'/((ident or mode)+'.log')
    log.parent.mkdir(exist_ok=True)
    with log.open('a') as stream:
        child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            code=child.wait(timeout=7200)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGKILL)
            child.wait()
            raise
        if code:
            raise RuntimeError(f'{mode} {ident} exited {code}; see {log}')

def summarize():
    import numpy as np
    manifest=read(SELECTED)
    protocol=read(OUT/'protocol.json')
    assert sha(SELECTED)==protocol['selection_sha256']
    labels=np.load(OUT/'labels.npy')
    assert sha(OUT/'test.npy')==read(OUT/'data-protocol.json')['test_array_sha256']
    assert sha(OUT/'labels.npy')==read(OUT/'data-protocol.json')['labels_sha256']
    rows=[]
    for s in manifest['networks']:
        folder=OUT/'models'/s['id']
        fp, q = read(folder/'fp32.json'),read(folder/'int8.json')
        for prefix, report, arrayfile in [('fp32',fp,'fp32-logits.npy'),('int8',q,'int8-outputs.npy')]:
            array=np.load(folder/arrayfile)
            pred=np.load(folder/(prefix+'-predictions.npy'))
            assert array.shape==(6149,102) and np.array_equal(array.argmax(1),pred)
            assert sha(folder/(prefix+'-predictions.npy'))==report['predictions_sha256']
            key='logits_sha256' if prefix=='fp32' else 'outputs_sha256'
            assert sha(folder/arrayfile)==report[key]
            for k,v in metrics(pred,labels).items():
                assert report[k]==v
        assert q['production_sha256']==s['production_sha256']==sha(ROOT/'runs'/s['run']/s['quant_directory']/'model.tflite')
        assert fp['checkpoint_sha256']==sha(ROOT/'runs'/s['run']/'best.pt')
        rows.append({'id':s['id'],'validation_int8_top1':s['int8_top1'],'test_count':6149,
            'fp32_test_top1':fp['top1'],'int8_test_top1':q['top1'],
            'fp32_test_mean_class_accuracy':fp['mean_class_accuracy'],
            'int8_test_mean_class_accuracy':q['mean_class_accuracy'],
            'test_quantization_drop':fp['top1']-q['top1'],
            'test_mean_class_quantization_drop':fp['mean_class_accuracy']-q['mean_class_accuracy'],
            'production_sha256':q['production_sha256']})
    write(OUT/'summary.json',{'status':'complete','count':15,'test_images_per_model':6149,
        'selection_sha256':sha(SELECTED),'test_used_for_selection':False,'test_used_for_training':False,
        'physical_board_flashed':False,'real_hardware_validated':False,
        'selection_unchanged':True,'independent_saved_predictions_audit_passed':True,
        'finished_utc':datetime.now(timezone.utc).isoformat(),'networks':rows})
    with (OUT/'test-metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps({'complete':True,'models':15,'int8_test_top1_range':[min(r['int8_test_top1'] for r in rows),max(r['int8_test_top1'] for r in rows)]}),flush=True)

def driver(gpu):
    OUT.mkdir(exist_ok=True)
    with (OUT/'evaluation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        selected=read(SELECTED)
        assert selected['status']=='locked' and len(selected['networks'])==selected['count']==15
        assert read(ROOT/'runs/nas-final-v1/final-audit.json')['passed'] is True
        protocol={'selection_sha256':sha(SELECTED),'audit_sha256':sha(ROOT/'runs/nas-final-v1/final-audit.json'),
            'selection_locked_unix':selected['locked_unix'],
            'model_ids':[s['id'] for s in selected['networks']],
            'production_hashes':{s['id']:s['production_sha256'] for s in selected['networks']},
            'test_count':6149,'test_split':'official Flowers102 test',
            'metrics':['overall_top1','mean_per_class_accuracy','paired_FP32_minus_INT8_test_drop'],
            'integer_runtime':'TFLite BUILTIN_REF','code_sha256':sha(Path(__file__)),
            'model_reselection_allowed':False,'finetuning_allowed':False,'recalibration_allowed':False,
            'physical_board_flashed':False}
        if (OUT/'protocol.json').exists():
            assert read(OUT/'protocol.json')==protocol
        else:
            write(OUT/'protocol.json',protocol)
            write(OUT/'evaluation-start.json',{'started_unix':time.time(),'protocol_sha256':sha(OUT/'protocol.json'),
                'test_data_not_yet_prepared':not (OUT/'data-protocol.json').exists()})
        if not (OUT/'data-protocol.json').exists():
            command('prepare',[72,73,74,75])
        with ThreadPoolExecutor(max_workers=16) as pool:
            jobs=[pool.submit(command,'floating',[76,77,78,79],gpu)]
            for i,s in enumerate(selected['networks']):
                if not (OUT/'models'/s['id']/'int8.json').exists():
                    jobs.append(pool.submit(command,'integer',[40+2*i,41+2*i],-1,s['id']))
            for job in as_completed(jobs):
                job.result()
        summarize()

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['driver','prepare','integer','floating','summarize'])
    ap.add_argument('--id');ap.add_argument('--gpu',type=int,default=7)
    a=ap.parse_args()
    if a.mode=='driver':driver(a.gpu)
    elif a.mode=='prepare':prepare()
    elif a.mode=='integer':integer(a.id)
    elif a.mode=='floating':floating(a.gpu)
    else:summarize()
