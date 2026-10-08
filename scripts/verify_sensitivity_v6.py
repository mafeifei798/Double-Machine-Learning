#!/usr/bin/env python3
"""Independent source, fold membership, alias, raw-statistic and multiplicity audit."""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from scipy.stats import t,norm,binomtest
ROOT=Path(__file__).resolve().parents[1]
FIELDS=('level','design','dgp','n','k_folds','preset','learner','method')
COMPONENTS=('weighted_oracle','weighted_first_order','weighted_second_order')
FACTORS={'bias':'bias_change','abs_error':'mae_change','squared_error':'mse_change','covers':'coverage_change',
    'nuisance_rmse_g':'g_rmse_change','nuisance_rmse_m':'m_rmse_change'}


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def close(a,b):assert math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-12),(a,b)
def interval(x):
    x=np.asarray(x,dtype=float);mu=x.mean();half=t.ppf(.975,len(x)-1)*x.std(ddof=1)/math.sqrt(len(x))
    return dict(mean=mu,low=mu-half,high=mu+half)
def check_interval(actual,x):
    for k,v in interval(x).items():close(actual[k],v)
def seed(base,*labels):return int.from_bytes(hashlib.sha256('\0'.join(map(str,('fair-iv-v1',base,*labels))).encode()).digest()[:4],'little')
def vector_hash(x):return hashlib.sha256(np.asarray(x,dtype='<f8').tobytes()).hexdigest()


def main():
    root=ROOT/'output/formal';manifest=json.loads((root/'FROZEN_MANIFEST.json').read_text());identity=manifest['identity'];p=identity['protocol']
    assert hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()==manifest['identity_sha256']
    for name,digest in identity['source_hashes'].items():assert sha(ROOT/name)==digest
    assert sha(ROOT/'DOUBLEML_CHECK.json')==manifest['doubleml_check_sha256']
    parity=json.loads((ROOT/'DOUBLEML_CHECK.json').read_text());assert parity['status']=='PASS' and len(parity['checks'])==48
    assert json.loads((root/'STATUS.json').read_text())['status']=='complete'
    scenarios=[(n,k,'base') for n in p['sample_sizes'] for k in p['fold_counts']]+[(1000,2,s) for s in p['presets'] if s!='base']
    expected={(d,g,n,k,s,r) for d in ('figure1','figure2') for g in ('linear','nonlinear') for n,k,s in scenarios for r in range(p['replications'])}
    ledger=json.loads((root/'RESULT_LEDGER.json').read_text());assert set(ledger)=={f.name for f in root.glob('figure*.json')}
    seen=set();groups=defaultdict(list);warnings=[];all_rows=0
    for name,digest in sorted(ledger.items()):
        assert sha(root/name)==digest;receipt=json.loads((root/name).read_text());key=tuple(receipt['key'])
        assert key in expected and key not in seen and receipt['identity_sha256']==manifest['identity_sha256'];seen.add(key)
        design,dgp,n,k,preset,rep=key;methods=('naive_ml','dml_crossfit') if design=='figure1' else ('full_no_split','equal_train_reuse','dml_crossfit')
        order=np.random.default_rng(seed(p['seed'],design,dgp,n,rep,'sensitivity-v6-common-permutation')).permutation(n)
        folds=np.empty(n,dtype=np.int16)
        for j,idx in enumerate(np.array_split(order,k)):folds[idx]=j
        payload=b'thesis-iv-repair-fold-v1\0'+n.to_bytes(8,'little')+np.asarray(folds,dtype='<i2').tobytes()
        fold_hash=hashlib.sha256(payload).hexdigest()[:16]
        assert {(r['level'],r['learner'],r['method']) for r in receipt['rows']}=={(l,f,m) for l in p['levels'] for f in p['learners'] for m in methods}
        assert len(receipt['rows'])==len(p['levels'])*len(p['learners'])*len(methods)
        for field in ('x_hash','u_hash','v_hash','fold_hash','data_seed'):assert len({r[field] for r in receipt['rows']})==1
        warnings.extend(receipt['warnings']);all_rows+=len(receipt['rows'])
        for row in receipt['rows']:
            assert (row['design'],row['dgp'],row['n'],row['k_folds'],row['preset'],row['replication'])==key
            assert row['fold_hash']==fold_hash and row['protocol_id']==p['protocol_id'] and row['stage']==p['stage']
            assert row['controlled_known_treatment_support'] and not row['artificial_pressure']
            assert row['learner']==row['treatment_learner']==row['outcome_learner']
            assert row['finite'] and row['se']>0 and abs(row['score_mean'])<1e-10
            close(row['bias'],row['theta']-.5);close(row['abs_error'],abs(row['bias']));close(row['squared_error'],row['bias']**2)
            assert row['covers']==(abs(row['bias'])<=1.96*row['se'])
            assert row['bank_identity']==hashlib.sha256(json.dumps(row['bank_records'],sort_keys=True).encode()).hexdigest()
            for j,record in enumerate(row['bank_records']):
                train=np.arange(n) if row['method']=='full_no_split' else np.flatnonzero(folds!=j)
                assert record['training_indices_sha256']==vector_hash(train) and record['train_n']==len(train)
                assert record['treatment_columns']==[0,2,4,6] and record['pilot_family']=='lasso'
            if row['method']!='naive_ml':close(row['bias'],sum(row[c] for c in COMPONENTS))
            groups[tuple(row[f] for f in FIELDS)].append(row)
        for level in p['levels']:
            for family in p['learners']:
                group={r['method']:r for r in receipt['rows'] if r['level']==level and r['learner']==family}
                cf=group['dml_crossfit'];reuse=group['naive_ml' if design=='figure1' else 'equal_train_reuse']
                assert cf['bank_records']==reuse['bank_records'] and cf['train_n_per_model']==reuse['train_n_per_model']==n*(k-1)//k
                assert cf['prediction_overlap_n']==0
                if design=='figure1':assert cf['g_prediction_hash']==reuse['g_prediction_hash']
                else:assert reuse['prediction_overlap_n']==n and group['full_no_split']['train_n_per_model']==n
    assert seen==expected and len(seen)==8000 and all_rows==120000 and len(groups)==600
    for group in groups.values():group.sort(key=lambda r:r['replication']);assert [r['replication'] for r in group]==list(range(200))
    # Check same-data pairing across presets/K and exact aliases, without fitting.
    alias_checks=0
    for key,g in groups.items():
        level,design,dgp,n,k,preset,family,method=key
        ref=groups[(level,design,dgp,n,2,'base',family,method)]
        for a,b in zip(ref,g):
            for field in ('x_hash','u_hash','v_hash','data_seed'):assert a[field]==b[field]
        if method=='full_no_split' and preset=='base' and k==5:
            for a,b in zip(ref,g):assert a['bank_records']==b['bank_records'] and a['theta']==b['theta'];alias_checks+=1
        if design=='figure2' and family=='rf' and preset=='outcome_weak':
            for a,b in zip(ref,g):assert a['bank_records']==b['bank_records'] and a['theta']==b['theta'];alias_checks+=1
    report=json.loads((root/'analysis/SUMMARY.json').read_text());assert report['identity_sha256']==manifest['identity_sha256']
    assert report['total_estimates']==all_rows and report['warnings']==warnings and report['failures']==0
    assert len(report['summaries'])==600 and {tuple(r[f] for f in FIELDS) for r in report['summaries']}==set(groups)
    for row in report['summaries']:
        g=groups[tuple(row[f] for f in FIELDS)];assert row['R']==200;check_interval(row['bias'],[r['bias'] for r in g])
        close(row['mean_theta'],np.mean([r['theta'] for r in g]));close(row['mc_sd'],np.std([r['theta'] for r in g],ddof=1))
        for field,name in [('se','mean_se'),('abs_error','mae'),('nuisance_rmse_g','nuisance_rmse_g'),('nuisance_rmse_m','nuisance_rmse_m')]:close(row[name],np.mean([r[field] for r in g]))
        close(row['rmse'],np.sqrt(np.mean([r['squared_error'] for r in g])))
        assert row['practical_zero_supported']==(row['bias']['low']>=-.01 and row['bias']['high']<=.01)
        assert row['nonzero_bias_detected']==(row['bias']['low']>0 or row['bias']['high']<0)
        rate=sum(r['covers'] for r in g)/200;z=norm.ppf(.975);den=1+z*z/200
        center=(rate+z*z/400)/den;half=z*np.sqrt(rate*(1-rate)/200+z*z/(4*200*200))/den
        for name,value in dict(rate=rate,low=center-half,high=center+half).items():close(row['coverage'][name],value)
        for component in row.get('decomposition',{}):check_interval(row['decomposition'][component],[r[component] for r in g])
    contrasts=report['method_contrasts'];assert len(contrasts)==360 and report['method_holm_family_size']==360
    assert len({(*tuple(c[f] for f in FIELDS),c['baseline']) for c in contrasts})==360
    for c in contrasts:
        key=tuple(c[f] for f in FIELDS);a=groups[(*key[:-1],c['baseline'])];b=groups[key]
        close(c['absolute_mean_bias_before'],abs(np.mean([r['bias'] for r in a])));close(c['absolute_mean_bias_after'],abs(np.mean([r['bias'] for r in b])))
        check_interval(c['bias_difference'],[y['bias']-x['bias'] for x,y in zip(a,b)])
        for metric,field in [('mae','abs_error'),('mse','squared_error')]:
            delta=np.array([y[field]-x[field] for x,y in zip(a,b)]);check_interval(c[metric+'_difference'],delta);sd=delta.std(ddof=1)
            value=t.cdf(delta.mean()/sd*np.sqrt(200),199) if sd else (0. if delta.mean()<0 else 1.);close(c[metric+'_p'],value)
    factors=report['factor_contrasts'];assert len(factors)==960
    expected_factors=set()
    for key in groups:
        l,d,g,n,k,preset,f,m=key
        if preset!='base':expected_factors.add(('preset',preset+'-base',key))
        elif k==5:expected_factors.add(('fold','K5-K2',key))
        if preset=='base' and n in (1000,3000):expected_factors.add(('sample_size',f'n{n}-n{300 if n==1000 else 1000}',key))
        if l=='moderate':expected_factors.add(('difficulty','moderate-baseline',key))
    assert {(c['axis'],c['label'],tuple(c[f] for f in FIELDS)) for c in factors}==expected_factors
    for c in factors:
        key=tuple(c[f] for f in FIELDS);l,d,g,n,k,preset,f,m=key;b=groups[key]
        if c['axis']=='preset':ref=(l,d,g,n,k,'base',f,m)
        elif c['axis']=='fold':ref=(l,d,g,n,2,preset,f,m)
        elif c['axis']=='sample_size':ref=(l,d,g,300 if n==1000 else 1000,k,preset,f,m)
        else:ref=('baseline',*key[1:])
        a=groups[ref]
        for field,name in FACTORS.items():
            delta=np.array([float(y[field])-float(x[field]) for x,y in zip(a,b)]);check_interval(c[name],delta);sd=delta.std(ddof=1)
            if field=='covers':
                left=sum(not x['covers'] and y['covers'] for x,y in zip(a,b));right=sum(x['covers'] and not y['covers'] for x,y in zip(a,b))
                value=binomtest(left,left+right,.5).pvalue if left+right else 1.
            else:value=2*t.sf(abs(delta.mean())/sd*np.sqrt(200),199) if sd else (1. if delta.mean()==0 else 0.)
            close(c[name+'_p'],value)
    def check_holm(group,names):
        for name in names:
            ordered=sorted(group,key=lambda r:r[name+'_p']);running=0.
            for rank,c in enumerate(ordered):running=max(running,min(1.,(len(group)-rank)*c[name+'_p']));close(c[name+'_holm_p'],running)
    check_holm(contrasts,['mae','mse'])
    for axis in ('fold','sample_size','preset','difficulty'):check_holm([c for c in factors if c['axis']==axis],list(FACTORS.values()))
    figures=json.loads((root/'analysis/FIGURE_LEDGER.json').read_text());assert len(figures)==30
    for name,digest in figures.items():assert sha(root/'analysis'/name)==digest
    print(json.dumps(dict(status='PASS',jobs=len(seen),estimates=all_rows,summary_cells=600,method_contrasts=360,factor_contrasts=960,
        aliases_checked=alias_checks,figures=30,warnings=len(warnings),identity_sha256=manifest['identity_sha256']),indent=2))


if __name__=='__main__':main()
