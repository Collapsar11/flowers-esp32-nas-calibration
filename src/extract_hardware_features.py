"""Static deployment-graph features only; no device latency or test accuracy."""
import hashlib,json,os
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import numpy as np
import tensorflow as tf
from tensorflow.lite.python import schema_py_generated as schema
ROOT=Path(__file__).resolve().parents[1]
manifest=ROOT/'runs/nas-final-v1/selected-networks.json'
out=ROOT/'runs/strict6-esp32-v1'
out.mkdir(exist_ok=True)
rows=[]
for s in json.loads(manifest.read_text())['networks']:
    path=ROOT/'runs'/s['run']/s['quant_directory']/'model.tflite'
    blob=path.read_bytes()
    assert hashlib.sha256(blob).hexdigest()==s['production_sha256']
    flat=schema.Model.GetRootAsModel(blob,0)
    graph=flat.Subgraphs(0)
    const={i for i in range(graph.TensorsLength()) if flat.Buffers(graph.Tensors(i).Buffer()).DataLength()>0}
    it=tf.lite.Interpreter(model_content=blob,num_threads=1,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF)
    it.allocate_tensors()
    sizes={t['index']:int(np.prod(t['shape']))*np.dtype(t['dtype']).itemsize for t in it.get_tensor_details()}
    ops=it._get_ops_details()
    activation=sum(sizes[int(t)] for op in ops for key in ['inputs','outputs'] for t in op[key] if t>=0 and int(t) not in const)
    rows.append({'id':s['id'],'production_sha256':s['production_sha256'],
        'macs':s['macs'],'depthwise_mac_fraction':s['depthwise_mac_fraction'],
        'activation_io_bytes':activation,'operator_count':len(ops),
        'model_bytes':len(blob),'qemu_psram_peak_bytes':s['psram_peak_bytes'],
        'qemu_internal_heap_peak_bytes':s['internal_heap_peak_bytes'],
        'arena_bytes':s['tensor_arena_bytes'],'total_depth':sum(s['arch']['d']),
        'validation_top1':s['int8_top1']})
result={'selection_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
    'feature_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'scope':'static graph features plus previously available QEMU memory proxies; no physical labels or image-test results',
    'activation_io_definition':'sum of nonconstant input and output tensor bytes over all operators; counts repeated accesses; ignores fusion/cache/liveness; NOT measured memory traffic or peak RAM',
    'tensorflow':tf.__version__,'networks':rows}
path=out/'public-features.json'
if path.exists():assert json.loads(path.read_text())==result
else:path.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'models':len(rows),'file':str(path)}),flush=True)
