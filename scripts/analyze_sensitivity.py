#!/usr/bin/env python3
"""Unfiltered sensitivity summaries and paired Monte Carlo contrasts."""
from collections import defaultdict
import json
from pathlib import Path
import sys
import numpy as np
from scipy import stats
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from dml_studies import orthogonalization_crossfitting as base
from study_utils import mean_interval,wilson,improvement_p,holm,atomic_json,digest
from run_sensitivity import universe,filename
FIELDS=('level','design','dgp','n','k_folds','preset','learner','method')
FACTOR_FIELDS={'bias':'bias_change','abs_error':'mae_change','squared_error':'mse_change','covers':'coverage_change',
    'nuisance_rmse_g':'g_rmse_change','nuisance_rmse_m':'m_rmse_change'}


def load():
    root=ROOT/'output/formal';manifest=json.loads((root/'FROZEN_MANIFEST.json').read_text());p=manifest['identity']['protocol']
    assert json.loads((root/'STATUS.json').read_text())['status']=='complete'
    ledger=json.loads((root/'RESULT_LEDGER.json').read_text());assert set(ledger)=={filename(k) for k in universe(p)}
    rows=[];warnings=[]
    for name,sha in sorted(ledger.items()):
        assert digest(root/name)==sha;receipt=json.loads((root/name).read_text())
        assert filename(receipt['key'])==name and receipt['identity_sha256']==manifest['identity_sha256']
        rows.extend(receipt['rows']);warnings.extend(receipt['warnings'])
    return root,manifest,rows,warnings


def two_sided(delta):
    x=np.asarray(delta,dtype=float);sd=x.std(ddof=1)
    return float(2*stats.t.sf(abs(x.mean())/sd*np.sqrt(len(x)),len(x)-1)) if sd else (1. if x.mean()==0 else 0.)


def factor_row(axis,key,a,b,label):
    assert [r['replication'] for r in a]==[r['replication'] for r in b]
    c=dict(axis=axis,label=label,**dict(zip(FIELDS,key)),R=len(a))
    for field,name in FACTOR_FIELDS.items():
        delta=[float(y[field])-float(x[field]) for x,y in zip(a,b)];c[name]=mean_interval(delta)
        if field=='covers':
            left=sum(not x['covers'] and y['covers'] for x,y in zip(a,b));right=sum(x['covers'] and not y['covers'] for x,y in zip(a,b))
            c[name+'_p']=float(stats.binomtest(left,left+right,.5).pvalue) if left+right else 1.
        else:c[name+'_p']=two_sided(delta)
    return c


def main():
    root,manifest,rows,warnings=load();p=manifest['identity']['protocol'];out=root/'analysis';out.mkdir(exist_ok=True)
    failures=[r for r in rows if not r['finite']]
    if failures:atomic_json(out/'FAILURES.json',failures);raise RuntimeError('All failed rows preserved; final comparison blocked')
    groups=defaultdict(list)
    for r in rows:groups[tuple(r[f] for f in FIELDS)].append(r)
    for g in groups.values():g.sort(key=lambda r:r['replication'])
    summaries=[]
    for key,g in sorted(groups.items()):
        b=mean_interval([r['bias'] for r in g]);tol=p['bias_tolerance']
        row=dict(zip(FIELDS,key));row.update(R=len(g),bias=b,mean_theta=float(np.mean([r['theta'] for r in g])),
            mc_sd=float(np.std([r['theta'] for r in g],ddof=1)),mean_se=float(np.mean([r['se'] for r in g])),
            mae=float(np.mean([r['abs_error'] for r in g])),rmse=float(np.sqrt(np.mean([r['squared_error'] for r in g]))),
            coverage=wilson(sum(r['covers'] for r in g),len(g)),practical_zero_supported=b['low']>=-tol and b['high']<=tol,
            nonzero_bias_detected=b['low']>0 or b['high']<0)
        for field in ('nuisance_rmse_g','nuisance_rmse_m'):row[field]=float(np.mean([r[field] for r in g]))
        if key[-1]!='naive_ml':row['decomposition']={name:mean_interval([r[name] for r in g]) for name in ('weighted_oracle','weighted_first_order','weighted_second_order')}
        summaries.append(row)
    contrasts=[];factors=[]
    for key,b in groups.items():
        if key[-1]=='dml_crossfit':
            for method in base.methods(key[1])[:-1]:
                a=groups[(*key[:-1],method)];c=dict(zip(FIELDS,key));c.update(baseline=method,R=len(a),
                    absolute_mean_bias_before=abs(np.mean([r['bias'] for r in a])),absolute_mean_bias_after=abs(np.mean([r['bias'] for r in b])))
                for field,name,metric in [('abs_error','mae_difference','mae'),('squared_error','mse_difference','mse')]:
                    delta=[y[field]-x[field] for x,y in zip(a,b)];c[name]=mean_interval(delta);c[metric+'_p']=improvement_p(delta)
                c['bias_difference']=mean_interval([y['bias']-x['bias'] for x,y in zip(a,b)]);contrasts.append(c)
        level,design,dgp,n,k,preset,family,method=key
        if preset!='base':
            a=groups[(level,design,dgp,n,k,'base',family,method)];factors.append(factor_row('preset',key,a,b,preset+'-base'))
        elif k==5:
            a=groups[(level,design,dgp,n,2,'base',family,method)];factors.append(factor_row('fold',key,a,b,'K5-K2'))
        if preset=='base' and n in (1000,3000):
            prev=300 if n==1000 else 1000;a=groups[(level,design,dgp,prev,k,preset,family,method)]
            factors.append(factor_row('sample_size',key,a,b,f'n{n}-n{prev}'))
        if level=='moderate':
            a=groups[('baseline',*key[1:])];factors.append(factor_row('difficulty',key,a,b,'moderate-baseline'))
    for metric in ('mae','mse'):
        for c,q in zip(contrasts,holm([r[metric+'_p'] for r in contrasts])):c[metric+'_holm_p']=q
    for axis in ('fold','sample_size','preset','difficulty'):
        group=[c for c in factors if c['axis']==axis]
        for name in FACTOR_FIELDS.values():
            for c,q in zip(group,holm([r[name+'_p'] for r in group])):c[name+'_holm_p']=q
    atomic_json(out/'SUMMARY.json',dict(identity_sha256=manifest['identity_sha256'],stage=p['stage'],total_estimates=len(rows),
        summaries=summaries,method_contrasts=contrasts,factor_contrasts=factors,warnings=warnings,failures=0,
        individual_cell_intervals=True,method_holm_family_size=len(contrasts),parameter_grid_not_full_factorial=True,
        full_no_split_k_aliases=True,rf_figure2_weak_outcome_alias=True))
    print(json.dumps(dict(status='ANALYZED',estimates=len(rows),summary_cells=len(summaries),method_contrasts=len(contrasts),factor_contrasts=len(factors))))


if __name__=='__main__':main()
