#!/usr/bin/env python3
"""Independent complete-universe, paired-data, statistics and Holm verification."""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from scipy.stats import t,norm
ROOT=Path(__file__).resolve().parents[1]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def close(a,b):assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-12),(a,b)


def interval(values):
    x=np.array(values);mu=float(np.mean(x));h=float(t.ppf(.975,len(x)-1)*np.std(x,ddof=1)/math.sqrt(len(x)))
    return dict(mean=mu,low=mu-h,high=mu+h)


def check_interval(actual,values):
    for key,value in interval(values).items():close(actual[key],value)


def main():
    root=ROOT/'output/exploratory';manifest=json.loads((root/'FROZEN_MANIFEST.json').read_text());p=manifest['identity']['protocol']
    assert hashlib.sha256(json.dumps(manifest['identity'],sort_keys=True).encode()).hexdigest()==manifest['identity_sha256']
    for name,digest in manifest['identity']['source_hashes'].items():assert sha(ROOT/name)==digest
    assert sha(ROOT/'DOUBLEML_CHECK.json')==manifest['doubleml_check_sha256']
    package=json.loads((ROOT/'DOUBLEML_CHECK.json').read_text());assert package['status']=='PASS' and len(package['checks'])==24
    assert json.loads((root/'STATUS.json').read_text())['status']=='complete'
    ledger=json.loads((root/'RESULT_LEDGER.json').read_text());assert set(ledger)=={f.name for f in root.glob('figure*.json')}
    expected={(d,g,r) for d in ('figure1','figure2') for g in ('linear','nonlinear') for r in range(p['replications'][d])}
    seen=set();groups=defaultdict(list);warning_count=0;all_rows=0
    for name,digest in ledger.items():
        assert sha(root/name)==digest;receipt=json.loads((root/name).read_text());key=tuple(receipt['key'])
        assert key in expected and key not in seen and receipt['identity_sha256']==manifest['identity_sha256'];seen.add(key)
        methods=('naive_ml','dml_crossfit') if key[0]=='figure1' else ('full_no_split','equal_train_reuse','dml_crossfit')
        assert {(r['level'],r['learner'],r['method']) for r in receipt['rows']}=={(l,f,m) for l in p['levels'] for f in p['learners'] for m in methods}
        assert len(receipt['rows'])==2*3*len(methods)
        for field in ('x_hash','u_hash','v_hash','fold_hash','data_seed'):assert len({r[field] for r in receipt['rows']})==1
        warning_count+=len(receipt['warnings']);all_rows+=len(receipt['rows'])
        for r in receipt['rows']:
            assert (r['design'],r['dgp'],r['replication'])==key and r['protocol_id']==p['protocol_id']
            assert r['n']==1000 and r['p']==20 and r['stage']=='exploratory_extension' and not r['artificial_pressure']
            assert r['controlled_known_treatment_support'] and r['learner']==r['treatment_learner']==r['outcome_learner']
            assert r['bank_identity']==hashlib.sha256(json.dumps(r['bank_records'],sort_keys=True).encode()).hexdigest()
            for record in r['bank_records']:assert record['treatment_columns']==[0,2,4,6]
            assert r['finite'] and r['se']>0 and abs(r['score_mean'])<1e-10
            close(r['bias'],r['theta']-.5);close(r['abs_error'],abs(r['bias']));close(r['squared_error'],r['bias']**2)
            assert r['covers']==(abs(r['bias'])<=1.96*r['se'])
            if r['method']!='naive_ml':close(r['bias'],sum(r[c] for c in ('weighted_oracle','weighted_first_order','weighted_second_order')))
            groups[(r['level'],r['design'],r['dgp'],r['learner'],r['method'])].append(r)
        for level in p['levels']:
            for family in p['learners']:
                method_rows={r['method']:r for r in receipt['rows'] if r['level']==level and r['learner']==family}
                cf=method_rows['dml_crossfit'];reuse=method_rows['naive_ml' if key[0]=='figure1' else 'equal_train_reuse']
                assert cf['bank_records']==reuse['bank_records'] and cf['train_n_per_model']==reuse['train_n_per_model']==500
                assert cf['prediction_overlap_n']==0
                if key[0]=='figure1':assert cf['g_prediction_hash']==reuse['g_prediction_hash']
                else:
                    assert reuse['prediction_overlap_n']==1000
                    assert method_rows['full_no_split']['train_n_per_model']==1000 and method_rows['full_no_split']['prediction_overlap_n']==1000
    assert seen==expected and len(seen)==800 and all_rows==12000
    s=json.loads((root/'analysis/SUMMARY.json').read_text())
    assert s['total_estimates']==all_rows and len(s['summaries'])==len(groups)==60
    assert len(s['warnings'])==warning_count and s['failures']==0
    for group in groups.values():group.sort(key=lambda r:r['replication'])
    for row in s['summaries']:
        g=groups[(row['level'],row['design'],row['dgp'],row['learner'],row['method'])];assert len(g)==row['R']==200
        check_interval(row['bias'],[r['bias'] for r in g]);close(row['mean_theta'],np.mean([r['theta'] for r in g]))
        close(row['mae'],np.mean([r['abs_error'] for r in g]));close(row['rmse'],math.sqrt(np.mean([r['squared_error'] for r in g])))
        close(row['mc_sd'],np.std([r['theta'] for r in g],ddof=1));close(row['mean_se'],np.mean([r['se'] for r in g]))
        for field in ('nuisance_rmse_g','nuisance_rmse_m','true_g_sd','true_m_sd'):close(row[field],np.mean([r[field] for r in g]))
        for component in row.get('decomposition',{}):check_interval(row['decomposition'][component],[r[component] for r in g])
        assert row['practical_zero_supported']==(row['bias']['low']>=-.01 and row['bias']['high']<=.01)
        assert row['nonzero_bias_detected']==(row['bias']['low']>0 or row['bias']['high']<0)
        n=len(g);rate=sum(r['covers'] for r in g)/n;z=float(norm.ppf(.975))
        center=(rate+z*z/(2*n))/(1+z*z/n);half=z*math.sqrt(rate*(1-rate)/n+z*z/(4*n*n))/(1+z*z/n)
        for k,v in dict(rate=rate,low=center-half,high=center+half).items():close(row['coverage'][k],v)
    assert len(s['contrasts'])==36 and s['holm_tests_per_metric']==36
    for c in s['contrasts']:
        key=(c['level'],c['design'],c['dgp'],c['learner']);a=groups[(*key,c['baseline'])];b=groups[(*key,'dml_crossfit')]
        assert c['R']==len(a)==len(b) and [r['replication'] for r in a]==[r['replication'] for r in b]
        for field,label in (('theta','theta_difference'),('abs_error','absolute_error_difference'),('squared_error','squared_error_difference')):
            check_interval(c[label],[y[field]-x[field] for x,y in zip(a,b)])
        for metric,field in (('mae','abs_error'),('mse','squared_error')):
            delta=np.array([y[field]-x[field] for x,y in zip(a,b)]);sd=delta.std(ddof=1)
            value=float(t.cdf(delta.mean()/sd*math.sqrt(len(delta)),len(delta)-1)) if sd else (0. if delta.mean()<0 else 1.)
            close(c[metric+'_improvement_p'],value)
        close(c['baseline_absolute_mean_bias'],abs(np.mean([r['bias'] for r in a])))
        close(c['crossfit_absolute_mean_bias'],abs(np.mean([r['bias'] for r in b])))
        labels=(c['level'],c['design'],c['dgp'],c['learner'],c['baseline'],'bootstrap-absolute-mean-bias')
        payload='\0'.join(map(str,('fair-iv-v1',p['seed'],*labels))).encode();seed=int.from_bytes(hashlib.sha256(payload).digest()[:4],'little')
        rng=np.random.default_rng(seed);indices=rng.integers(0,len(a),size=(p['bootstrap_draws'],len(a)))
        ax=np.array([r['bias'] for r in a]);bx=np.array([r['bias'] for r in b]);bootstrap=np.abs(bx[indices].mean(axis=1))-np.abs(ax[indices].mean(axis=1))
        actual=c['absolute_mean_bias_difference_bootstrap'];low,high=np.quantile(bootstrap,[.025,.975])
        close(actual['difference'],abs(bx.mean())-abs(ax.mean()));close(actual['low'],low);close(actual['high'],high)
    for metric in ('mae','mse'):
        ordered=sorted(enumerate(s['contrasts']),key=lambda item:item[1][metric+'_improvement_p']);running=0.
        for rank,(index,c) in enumerate(ordered):
            running=max(running,min(1.,(36-rank)*c[metric+'_improvement_p']));close(c[metric+'_holm_p'],running)
    assert len(s['difficulty_contrasts'])==30
    for c in s['difficulty_contrasts']:
        key=(c['design'],c['dgp'],c['learner'],c['method']);a=groups[('baseline',*key)];b=groups[('moderate',*key)]
        assert [r['replication'] for r in a]==[r['replication'] for r in b]
        for field,label in (('theta','theta_difference'),('abs_error','absolute_error_difference'),('squared_error','squared_error_difference'),
            ('nuisance_rmse_m','nuisance_rmse_m_difference'),('nuisance_rmse_g','nuisance_rmse_g_difference')):
            check_interval(c[label],[y[field]-x[field] for x,y in zip(a,b)])
    figures=json.loads((root/'analysis/FIGURE_LEDGER.json').read_text());assert len(figures)==16
    for name,digest in figures.items():assert sha(root/'analysis'/name)==digest
    print(json.dumps(dict(status='PASS',stage=p['stage'],paired_draws=len(seen),level_replications=1600,estimates=all_rows,
        summary_cells=60,method_contrasts=36,difficulty_contrasts=30,figures=16,warnings=warning_count,identity_sha256=manifest['identity_sha256']),indent=2))


if __name__=='__main__':main()
