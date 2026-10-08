#!/usr/bin/env python3
"""Independent raw-receipt statistics and exact paired-bank verification."""
import hashlib
import json
import math
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.stats import t,norm
ROOT=Path(__file__).resolve().parents[1]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def interval(x):
    x=np.array(x,dtype=float);a=float(np.mean(x));h=float(t.ppf(.975,len(x)-1)*np.std(x,ddof=1)/math.sqrt(len(x)))
    return dict(mean=a,low=a-h,high=a+h)


def close(a,b):
    if not math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-12):raise AssertionError((a,b))


def main():
    root=ROOT/'output/formal';manifest=json.loads((root/'FROZEN_MANIFEST.json').read_text());p=manifest['identity']['protocol']
    for name,digest in manifest['identity']['source_hashes'].items():assert sha(ROOT/name)==digest
    assert sha(ROOT/'DOUBLEML_CHECK.json')==manifest['doubleml_check_sha256']
    assert hashlib.sha256(json.dumps(manifest['identity'],sort_keys=True).encode()).hexdigest()==manifest['identity_sha256']
    ledger=json.loads((root/'RESULT_LEDGER.json').read_text());groups=defaultdict(list);seen=set();warning_count=0
    assert set(ledger)=={f.name for f in root.glob('figure*.json')}
    expected={(d,g,r) for d in ('figure1','figure2') for g in ('linear','nonlinear') for r in range(p['replications'][d])}
    for name,digest in ledger.items():
        assert sha(root/name)==digest;receipt=json.loads((root/name).read_text());key=tuple(receipt['key'])
        assert key in expected and key not in seen;seen.add(key);assert receipt['identity_sha256']==manifest['identity_sha256']
        expected_methods=('naive_ml','dml_crossfit') if key[0]=='figure1' else ('full_no_split','equal_train_reuse','dml_crossfit')
        assert {(r['learner'],r['method']) for r in receipt['rows']}=={(l,m) for l in p['learners'] for m in expected_methods}
        assert len(receipt['rows'])==len(p['learners'])*len(expected_methods)
        warning_count+=len(receipt['warnings'])
        for row in receipt['rows']:
            assert (row['design'],row['dgp'],row['replication'])==key
            assert row['n']==1000 and row['p']==20 and not row['artificial_pressure']
            assert row['learner']==row['treatment_learner']==row['outcome_learner']
            assert row['bank_identity']==hashlib.sha256(json.dumps(row['bank_records'],sort_keys=True).encode()).hexdigest()
            if row['finite']:
                for name in ('theta','se','bias','jacobian'):assert math.isfinite(row[name])
                assert row['se']>0 and abs(row['score_mean'])<1e-10
                close(row['bias'],row['theta']-.5);close(row['abs_error'],abs(row['bias']));close(row['squared_error'],row['bias']**2)
                assert row['covers']==(abs(row['bias'])<=1.96*row['se'])
                if row['method']!='naive_ml':close(row['bias'],sum(row[k] for k in ('weighted_oracle','weighted_first_order','weighted_second_order')))
            groups[(row['design'],row['dgp'],row['learner'],row['method'])].append(row)
        for family in p['learners']:
            pairs={r['method']:r for r in receipt['rows'] if r['learner']==family}
            cf=pairs['dml_crossfit'];baseline=pairs['naive_ml' if key[0]=='figure1' else 'equal_train_reuse']
            assert cf['bank_records']==baseline['bank_records'] and cf['train_n_per_model']==baseline['train_n_per_model']==500
            assert cf['prediction_overlap_n']==0
            if key[0]=='figure1':assert cf['g_prediction_hash']==baseline['g_prediction_hash']
            else:
                assert baseline['prediction_overlap_n']==1000
                assert pairs['full_no_split']['train_n_per_model']==1000 and pairs['full_no_split']['prediction_overlap_n']==1000
    assert seen==expected
    summary=json.loads((root/'analysis/SUMMARY.json').read_text());assert len(summary['summaries'])==len(groups)==30
    assert summary['total_estimates']==sum(len(g) for g in groups.values())==30000
    assert len(summary['warnings'])==warning_count
    for s in summary['summaries']:
        g=groups[(s['design'],s['dgp'],s['learner'],s['method'])];f=[r for r in g if r['finite']]
        assert len(g)==s['R']==1000 and len(f)==s['finite_R']
        calculated=interval([r['bias'] for r in f])
        for k in calculated:close(calculated[k],s['bias'][k])
        close(s['mean_theta'],np.mean([r['theta'] for r in f]));close(s['rmse'],math.sqrt(np.mean([r['squared_error'] for r in f])))
        close(s['mae'],np.mean([r['abs_error'] for r in f]));close(s['mc_sd'],np.std([r['theta'] for r in f],ddof=1))
        close(s['mean_se'],np.mean([r['se'] for r in f]))
        n=len(f);rate=sum(r['covers'] for r in f)/n;z=norm.ppf(.975)
        center=(rate+z*z/(2*n))/(1+z*z/n);half=z*math.sqrt(rate*(1-rate)/n+z*z/(4*n*n))/(1+z*z/n)
        for k,v in dict(rate=rate,low=center-half,high=center+half).items():close(s['coverage'][k],v)
        assert s['practical_zero_supported']==(len(g)==len(f) and calculated['low']>=-.01 and calculated['high']<=.01)
        assert s['nonzero_bias_detected']==(calculated['low']>0 or calculated['high']<0)
        for channel in ('g','m'):close(s['nuisance_rmse_'+channel],np.mean([r['nuisance_rmse_'+channel] for r in f]))
        if s['method']!='naive_ml':
            for component in s['decomposition']:
                for k,v in interval([r[component] for r in f]).items():close(s['decomposition'][component][k],v)
    assert len(summary['contrasts'])==18
    for c in summary['contrasts']:
        key=(c['design'],c['dgp'],c['learner']);a=sorted(groups[(*key,c['baseline'])],key=lambda r:r['replication'])
        b=sorted(groups[(*key,'dml_crossfit')],key=lambda r:r['replication'])
        assert [r['replication'] for r in a]==[r['replication'] for r in b]
        pair=[(x,y) for x,y in zip(a,b) if x['finite'] and y['finite']]
        for source,target in (('theta','theta_difference'),('abs_error','absolute_error_difference'),('squared_error','squared_error_difference')):
            for k,v in interval([y[source]-x[source] for x,y in pair]).items():close(c[target][k],v)
    figures=json.loads((root/'analysis/FIGURE_LEDGER.json').read_text())
    assert len(figures)==8
    for name,digest in figures.items():assert sha(root/'analysis'/name)==digest
    print(json.dumps(dict(status='PASS',replications=len(seen),estimates=30000,
        summary_cells=30,paired_contrasts=18,figures=8,warnings=warning_count,identity_sha256=manifest['identity_sha256']),indent=2))


if __name__=='__main__':main()
