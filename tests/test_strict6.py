"""Safety of experimental separation and numerical correctness; synthetic data only."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import copy
import json
from pathlib import Path
import tempfile
import time
import unittest
import uuid
import numpy as np
import strict6 as s
from measurement_io import PROTOCOL_SHA, collect, design, parse_serial, read, sha, write_new


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(prefix='strict6-synthetic-test-')
        cls.base=Path(cls.temp.name); cls.protocol,cls.features=design()
        cls.support=cls.fixtures(cls.protocol['support_ids'],'support',time.time()-10)
        cls.lockdir=cls.base/'locked'
        s.calibrate(cls.support,cls.lockdir,simulation=True)
        cls.query=cls.fixtures(cls.protocol['query_ids'],'query',time.time())

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    @classmethod
    def fixtures(cls, ids, namespace, timestamp):
        paths=[]
        for ident in ids:
            row=cls.features[ident]; folder=cls.base/namespace/ident;folder.mkdir(parents=True)
            manifest={'id':ident,'production_sha256':row['production_sha256'], 'protocol_sha256':PROTOCOL_SHA,
                'golden_inputs_sha256':'synthetic-not-hardware','input_order':[list(range(102))]*3,
                'arena_used_bytes':row['arena_bytes'],'arena_reserved_bytes':((row['arena_bytes']+15)//16)*16+4096,
                'embedded_alignment_bytes':16,
                'embedded_symbol_addresses':{'_binary_model_tflite_start':0x3c000000,'_binary_golden_inputs_bin_start':0x3c400000,'_binary_golden_outputs_bin_start':0x3cb80000},
                'runtime_fingerprint':'SYNTHETIC_UNIT_TEST_ONLY'}
            write_new(folder/'manifest.json',manifest)
            for kind in ['timing','memory']:
                for session in range(3):
                    out=folder/f'{kind}-{session}';out.mkdir();nonce=uuid.uuid4().hex
                    begin={'event':'begin','id':ident,'model_sha256':row['production_sha256'],
                        'protocol_sha256':PROTOCOL_SHA,'input_sha256':manifest['golden_inputs_sha256'],
                        'kind':kind,'session':session,'nonce':nonce,'core':1,'cpu_hz':240000000,
                        'psram_bytes':8388608,'psram_mhz':80,'flash_mhz':80,'hardware_source':'unattested'}
                    count=204 if kind=='timing' else 102
                    done={'event':'done','count':count,'warmup':10 if kind=='timing' else 0,
                        'integer_mismatches':0,'class_agreement':count,'arena_used':row['arena_bytes'],
                        'arena_reserved':manifest['arena_reserved_bytes'],'stack_free_before_bytes':12000,'stack_free_after_bytes':10000}
                    records=[begin]
                    if kind=='timing':
                        records += [{'event':'invoke','n':i,'input':i%102,
                            'duration_us':int(row['macs']/1000)+session+i%7} for i in range(204)]
                    else:
                        peak=row['qemu_psram_peak_bytes']+32
                        records += [{'event':'memory','psram_peak_bytes':peak,'internal_heap_peak_bytes':0,
                            'psram_before':8000000,'psram_min_free':8000000-peak,'internal_before':300000,'internal_min_free':300000}]
                    records.append(done)
                    text='\n'.join('FLOWERS_JSON '+json.dumps(r) for r in records)+'\nFLOWERS_MEASUREMENT_COMPLETE\n'
                    (out/'serial.log').write_text(text)
                    receipt={'source':'qemu','transport':'qemu_subprocess','silicon_latency_valid':False,
                        'kind':kind,'session':session,'nonce':nonce,'started_unix':timestamp,'ended_unix':timestamp+1,
                        'log_file':'serial.log','log_sha256':sha(out/'serial.log'),'manifest_file':'../manifest.json',
                        'manifest_sha256':sha(folder/'manifest.json'),'parsed':parse_serial(text,manifest,kind,session,nonce)}
                    write_new(out/'receipt.json',receipt);paths.append(str(out/'receipt.json'))
        file=cls.base/(namespace+'.json');write_new(file,paths);return file

    def test_end_to_end_and_source_marking(self):
        result=s.evaluate(self.lockdir/'lock.json',self.query,self.base/'evaluation.json',True)
        self.assertEqual(result['physical_measured_networks'],0)
        self.assertTrue(result['simulation_only']);self.assertEqual(result['heldout_architectures'],9)
        self.assertEqual(set(result['metrics']),set(s.METHODS))
        self.assertNotIn('mape_percent',result['metrics']['raw_macs'])

    def test_simulation_rejected_by_default(self):
        with self.assertRaises(AssertionError):collect(s.file_list(self.support),self.protocol['support_ids'])

    def test_only_fixed_six_allowed(self):
        with self.assertRaises(AssertionError):collect(s.file_list(self.query)[:36],self.protocol['support_ids'],True)

    def test_seventh_network_rejected(self):
        with self.assertRaises(AssertionError):collect(s.file_list(self.support)+s.file_list(self.query)[:6],self.protocol['support_ids'],True)

    def test_missing_session_rejected(self):
        with self.assertRaises(AssertionError):collect(s.file_list(self.support)[:-1],self.protocol['support_ids'],True)

    def test_duplicate_boot_rejected(self):
        p=s.file_list(self.support)
        with self.assertRaises(AssertionError):collect(p+[p[0]],self.protocol['support_ids'],True)

    def test_query_before_lock_rejected(self):
        with self.assertRaises(AssertionError):collect(s.file_list(self.query),self.protocol['query_ids'],True,after=time.time()+100)

    def test_lock_cannot_be_overwritten(self):
        with self.assertRaises(AssertionError):s.calibrate(self.support,self.lockdir,True)

    def test_tampered_lock_rejected(self):
        out=self.base/'tampered';out.mkdir();lock=read(self.lockdir/'lock.json');lock['created_unix']-=100
        write_new(out/'lock.json',lock);write_new(out/'lock-receipt.json',read(self.lockdir/'lock-receipt.json'))
        with self.assertRaises(AssertionError):s.verify_lock(out/'lock.json',True)

    def test_serial_nonce_and_truncation(self):
        r=read(s.file_list(self.support)[0]);folder=s.file_list(self.support)[0].parent
        text=(folder/'serial.log').read_text();manifest=read(folder/r['manifest_file'])
        with self.assertRaises(AssertionError):parse_serial(text,manifest,r['kind'],r['session'],'0'*32)
        with self.assertRaises(AssertionError):parse_serial(text.replace('FLOWERS_MEASUREMENT_COMPLETE',''),manifest,r['kind'],r['session'],r['nonce'])

    def test_ridge_against_augmented_least_squares(self):
        rows=[self.features[i] for i in self.protocol['support_ids']];y=np.asarray([91,184,135,129,149,98],float)
        model=s.fit_ridge(rows,y,3);x=np.asarray([s.vector(r,3) for r in rows]);z=(x-x.mean(0))/x.std(0)
        design=np.column_stack([np.ones(6),z]);penalty=np.column_stack([np.zeros(3),np.eye(3)])
        beta=np.linalg.lstsq(np.vstack([design,penalty]),np.r_[np.log(y),[0,0,0]],rcond=None)[0]
        np.testing.assert_allclose(np.r_[model['intercept'],model['coefficients']],beta,rtol=1e-10,atol=1e-10)

    def test_ranks_handle_ties(self):
        metrics=s.rank_metrics([1,1,3],[1,2,3])
        self.assertAlmostEqual(metrics['spearman'],np.sqrt(3)/2)
        self.assertAlmostEqual(metrics['kendall_tau_b'],2/np.sqrt(6))
        self.assertAlmostEqual(metrics['pairwise_ordering_agreement'],2.5/3)
        self.assertIsNone(s.rank_metrics([1,1],[2,3])['spearman'])

    def test_memory_zero_and_underprediction(self):
        r=s.memory_metrics([0,0],[0,10],5)
        self.assertEqual(r['maximum_underprediction_bytes'],10)
        self.assertEqual(r['feasibility_counts']['false_feasible'],1)
        self.assertIsNone(r['percentage_error'])

    def test_infeasible_selection_never_gets_good_regret(self):
        a,b=self.protocol['query_ids'][:2];actual={a:{'latency_ms':1,'psram_peak_bytes':7*2**20,'internal_heap_peak_bytes':0},
            b:{'latency_ms':2,'psram_peak_bytes':100,'internal_heap_peak_bytes':0}}
        score=s.decision_score(a,actual,self.features)
        self.assertEqual(score['status'],'budget_violation');self.assertIsNone(score['regret'])

    def test_loocv_refits_support_scaler(self):
        rows=[self.features[i] for i in self.protocol['support_ids']];model=s.fit_ridge(rows[:5],[1,2,3,4,5],3)
        np.testing.assert_allclose(model['mean'],np.asarray([s.vector(r,3) for r in rows[:5]]).mean(0))
        self.assertEqual(model['training_count'],5)


if __name__=='__main__':unittest.main(verbosity=2)
