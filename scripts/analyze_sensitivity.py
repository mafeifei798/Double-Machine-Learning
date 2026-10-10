#!/usr/bin/env python3
"""Full unfiltered sensitivity summaries, paired contrasts and plots."""
from collections import defaultdict
import json
from pathlib import Path
import sys
import numpy as np
from scipy import stats
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from dml_studies import orthogonalization_crossfitting as base
from analyze_reference import mean_interval,wilson
from analyze_moderate_dgp import improvement_p,holm
from run_reference import atomic_json,digest
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


def plot_grid(summaries,p,out,level,design,axis,metric):
    import matplotlib.pyplot as plt
    methods=base.methods(design);colors={'naive_ml':'#dc8b20','full_no_split':'#dc8b20','equal_train_reuse':'#8860b0','dml_crossfit':'#2677b5'}
    fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for i,dgp in enumerate(('linear','nonlinear')):
        for j,family in enumerate(p['learners']):
            ax=axes[i,j]
            for method in methods:
                if axis=='sample_fold':
                    for k in p['fold_counts']:
                        selected=[next(s for s in summaries if (s['level'],s['design'],s['dgp'],s['learner'],s['method'],s['preset'],s['n'],s['k_folds'])==(level,design,dgp,family,method,'base',n,k)) for n in p['sample_sizes']]
                        x=np.arange(len(selected));label=f'{method}, K={k}';style='-' if k==2 else '--'
                        y=[s['bias']['mean'] if metric=='bias' else s['coverage']['rate'] if metric=='coverage' else s['rmse'] for s in selected]
                        ax.plot(x,y,style,marker='o',markersize=4,color=colors[method],label=label)
                        if metric in ('bias','coverage'):
                            low=[s[metric]['low'] for s in selected];high=[s[metric]['high'] for s in selected]
                            ax.fill_between(x,low,high,color=colors[method],alpha=.08)
                    ax.set_xticks(x,p['sample_sizes']);ax.set_xlabel('n (K=5 also changes training fraction)')
                else:
                    selected=[next(s for s in summaries if (s['level'],s['design'],s['dgp'],s['learner'],s['method'],s['preset'],s['n'],s['k_folds'])==(level,design,dgp,family,method,preset,p['anchor_n'],p['anchor_k'])) for preset in p['presets']]
                    x=np.arange(len(selected));y=[s['bias']['mean'] if metric=='bias' else s['coverage']['rate'] if metric=='coverage' else s['rmse'] for s in selected]
                    ax.plot(x,y,'-o',markersize=4,color=colors[method],label=method)
                    if metric in ('bias','coverage'):ax.fill_between(x,[s[metric]['low'] for s in selected],[s[metric]['high'] for s in selected],color=colors[method],alpha=.08)
                    ax.set_xticks(x,['base','out weak','out strong','m weak','m strong'],rotation=25,ha='right');ax.set_xlabel('one-role-at-a-time presets')
            if metric=='bias':ax.axhline(0,color='black',lw=.7);ax.axhspan(-.01,.01,color='green',alpha=.1)
            if metric=='coverage':ax.axhline(.95,color='black',lw=.7,ls=':');ax.set_ylim(-.03,1.03)
            ax.set_title(f'{dgp} | {family.upper()}');ax.set_ylabel(metric);ax.grid(alpha=.15)
    axes[0,0].legend(fontsize=6)
    fig.suptitle(f'{level} | {design} | R={p["replications"]} | {axis} | individual intervals; controlled treatment inputs')
    name=f'{level}_{design}_{axis}_{metric}.png';fig.savefig(out/name,dpi=170);plt.close(fig);return name


def plot_hist(groups,p,out,design,dgp,left):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,2,figsize=(12,11),layout='constrained')
    for i,family in enumerate(p['learners']):
        pair=[groups[('moderate',design,dgp,1000,2,'base',family,m)] for m in (left,'dml_crossfit')]
        errors=[np.array([r['bias']/r['se'] for r in g]) for g in pair]
        lo=min(-4.,*(float(z.min()) for z in errors));hi=max(4.,*(float(z.max()) for z in errors));edges=np.linspace(lo,hi,41);grid=np.linspace(lo,hi,1200)
        for j,(g,z,method) in enumerate(zip(pair,errors,(left,'dml_crossfit'))):
            ax=axes[i,j];ax.hist(z,bins=edges,density=True,color='#3675b5',alpha=.78,edgecolor='white',linewidth=.3)
            ax.plot(grid,stats.norm.pdf(grid),color='#c63838',lw=1.6,label='N(0,1)');ax.axvline(0,color='black',lw=.7,alpha=.5)
            b=mean_interval([r['bias'] for r in g]);ax.set_title(f'{family.upper()} | {method}\nBias {b["mean"]:+.4f}; MC95 [{b["low"]:+.4f}, {b["high"]:+.4f}]',fontsize=10)
            ax.set_xlim(lo,hi);ax.set_xlabel('(estimate - .5) / estimated SE');ax.set_ylabel('Density');ax.legend(fontsize=8)
    fig.suptitle(f'Independent sensitivity anchor | moderate {design}/{dgp} | n=1000, K=2, R=200')
    name=f'moderate_{design}_{dgp}_{left}_hist.png';fig.savefig(out/name,dpi=170);plt.close(fig);return name


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
    import matplotlib
    matplotlib.use('Agg')
    figures=[]
    for level in p['levels']:
        for design in ('figure1','figure2'):
            for axis in ('sample_fold','parameters'):
                for metric in ('bias','rmse','coverage'):figures.append(plot_grid(summaries,p,out,level,design,axis,metric))
    for design in ('figure1','figure2'):
        for dgp in ('linear','nonlinear'):
            for left in base.methods(design)[:-1]:figures.append(plot_hist(groups,p,out,design,dgp,left))
    atomic_json(out/'FIGURE_LEDGER.json',{name:digest(out/name) for name in figures})
    print(json.dumps(dict(status='ANALYZED',estimates=len(rows),summary_cells=len(summaries),method_contrasts=len(contrasts),factor_contrasts=len(factors),figures=len(figures))))


if __name__=='__main__':main()
