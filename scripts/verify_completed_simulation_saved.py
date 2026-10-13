#!/usr/bin/env python3
"""Independent stdlib-only audit of saved simulation records. No ML imports/fits."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import tarfile


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--bridge-root',type=Path,required=True);ap.add_argument('--v6-root',type=Path,required=True);ap.add_argument('--report-root',type=Path,required=True);ap.add_argument('--output',type=Path);a=ap.parse_args()
    repo=Path(__file__).resolve().parents[1];groups=defaultdict(list)
    frozen=json.loads((a.bridge_root/'BRIDGE_FROZEN.json').read_text());p=frozen['identity']['protocol']
    assert p['seed']==2026101371 and p['anchor_n']==500 and p['replications']==300 and p['workers']==12
    assert p['learners']==['lasso','rf','xgb'] and p['levels']==['baseline'] and not p['artificial_pressure']
    assert json.loads((a.bridge_root/'BRIDGE_EXIT.json').read_text())['returncode']==0
    source_archive=a.bridge_root.parent/'server_source_v7.tgz'
    with tarfile.open(source_archive) as archive:
        for name,h in frozen['identity']['source_hashes'].items():
            b=archive.extractfile(name).read() if Path(name).name.startswith('._') else (repo/name).read_bytes()
            assert hashlib.sha256(b).hexdigest()==h
    bridge_ledger=json.loads((a.bridge_root/'output/legacy_bridge_v7/RESULT_LEDGER.json').read_text());assert len(bridge_ledger)==600
    v6_ledger=json.loads((a.v6_root/'output/formal/RESULT_LEDGER.json').read_text())
    selected=[]
    for name,h in bridge_ledger.items():
        path=a.bridge_root/'output/legacy_bridge_v7'/name;assert sha(path)==h
        receipt=json.loads(path.read_text());assert receipt['identity_sha256']==frozen['identity_sha256']
        assert len(receipt['rows'])==6
        selected.extend(receipt['rows'])
    for d in ('linear','nonlinear'):
        for r in range(200):
            name=f'figure2_{d}_n300_k2_base_rep{r:04d}.json';path=a.v6_root/'output/formal'/name
            assert sha(path)==v6_ledger[name]
            rows=[x for x in json.loads(path.read_text())['rows'] if x['level']=='baseline'];assert len(rows)==9
            selected.extend(rows)
    assert len(selected)==7200
    for r in selected:
        assert r['finite'] and r['se']>0 and r['p']==20 and r['theta0']==.5 and not r['artificial_pressure']
        err=r['theta']-.5
        assert abs(err-r['bias'])<1e-12
        assert abs(err*err-r['squared_error'])<1e-12
        assert r['covers']==(abs(err)<=1.96*r['se'])
        if r['method']!='naive_ml':assert abs(err-sum(r[k] for k in ('weighted_oracle','weighted_first_order','weighted_second_order')))<1e-10
        groups[(r['design'],r['dgp'],r['learner'],r['method'])].append(r)
    saved=json.loads((a.report_root/'summary.json').read_text());assert len(saved['primary_results'])==len(groups)==30
    for summary in saved['primary_results']:
        key=tuple(summary[k] for k in ('design','dgp','learner','method'));g=sorted(groups[key],key=lambda r:r['replication']);R=300 if key[0]=='figure1' else 200
        assert len(g)==R and [r['replication'] for r in g]==list(range(R))
        values=[r['theta'] for r in g];err=[x-.5 for x in values]
        recalculated={'mean_theta':statistics.mean(values),'bias':statistics.mean(err),'mc_sd':statistics.stdev(values),
                      'mean_se':statistics.mean(r['se'] for r in g),'mae':statistics.mean(abs(x) for x in err),
                      'rmse':math.sqrt(statistics.mean(x*x for x in err)),
                      'coverage':sum(abs(r['theta']-.5)<=1.96*r['se'] for r in g)/R}
        for k,v in recalculated.items():assert abs(v-summary[k])<1e-12,(key,k,v,summary[k])
        if key[-1]=='dml_crossfit':
            other=groups[(*key[:-1],'naive_ml' if key[0]=='figure1' else 'equal_train_reuse')]
            other=sorted(other,key=lambda r:r['replication'])
            for x,y in zip(g,other):
                assert x['bank_identity']==y['bank_identity'] and x['fold_hash']==y['fold_hash']
                assert x['prediction_overlap_n']==0 and y['prediction_overlap_n']==(0 if key[0]=='figure1' else 300)
                if key[0]=='figure1':assert x['g_prediction_hash']==y['g_prediction_hash'] and x['m_prediction_hash']==y['m_prediction_hash']
    verification=json.loads((a.report_root/'verification.json').read_text())
    for name,h in verification['generated_files'].items():assert sha(a.report_root/name)==h,name
    assert len(saved['figures'])==6 and len(saved['tables'])==8 and len(saved['extension_figures'])==32
    result={'status':'PASS_INDEPENDENT_STDLIB_RECOUNT','saved_estimates_recounted':7200,'summary_cells':30,
                      'bridge_jobs':600,'reused_v6_jobs':400,'model_fits':0,'ML_imports':0,
                      'bias_sd_se_mae_rmse_coverage':True,'pairing_and_overlap':True,'all_report_file_hashes':True}
    if a.output:a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
