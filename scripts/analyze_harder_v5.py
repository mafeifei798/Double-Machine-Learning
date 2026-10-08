#!/usr/bin/env python3
"""Complete paired difficulty report, multiplicity-aware errors and all figures."""
from collections import defaultdict
import json
from pathlib import Path
import sys
import numpy as np
from scipy import stats
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from dml_original import two_claims_v4 as base,fair_iv as fair
from analyze_fair_iv import mean_interval,wilson
from run_fair_iv import atomic_json,digest


def load(root):
    manifest=json.loads((root/'FROZEN_MANIFEST.json').read_text());p=manifest['identity']['protocol']
    assert json.loads((root/'STATUS.json').read_text())['status']=='complete'
    ledger=json.loads((root/'RESULT_LEDGER.json').read_text())
    assert set(ledger)=={f.name for f in root.glob('figure*.json')}
    expected={(d,g,r) for d in ('figure1','figure2') for g in ('linear','nonlinear') for r in range(p['replications'][d])}
    rows=[];warnings=[];seen=set()
    for name,sha in sorted(ledger.items()):
        assert digest(root/name)==sha;receipt=json.loads((root/name).read_text());key=tuple(receipt['key'])
        assert key in expected and key not in seen and receipt['identity_sha256']==manifest['identity_sha256'];seen.add(key)
        target={(level,family,method) for level in p['levels'] for family in p['learners'] for method in base.methods(key[0])}
        assert {(r['level'],r['learner'],r['method']) for r in receipt['rows']}==target and len(receipt['rows'])==len(target)
        rows.extend(receipt['rows']);warnings.extend(receipt['warnings'])
    assert seen==expected
    return manifest,rows,warnings


def improvement_p(values):
    x=np.array(values);sd=x.std(ddof=1)
    if sd==0:return 0. if x.mean()<0 else 1.
    return float(stats.t.cdf(x.mean()/(sd/np.sqrt(len(x))),len(x)-1))


def holm(values):
    order=np.argsort(values);adjusted=np.empty(len(values));running=0.
    for rank,index in enumerate(order):
        running=max(running,min(1.,(len(values)-rank)*values[index]));adjusted[index]=running
    return adjusted.tolist()


def bias_bootstrap(a,b,p,labels):
    # Joint resampling preserves replication pairing. This is exploratory,
    # not a regular unbiasedness test for the nonsmooth absolute-mean metric.
    rng=np.random.default_rng(fair.seed_for(p['seed'],*labels,'bootstrap-absolute-mean-bias'))
    a=np.array(a);b=np.array(b);indices=rng.integers(0,len(a),size=(p['bootstrap_draws'],len(a)))
    delta=np.abs(b[indices].mean(axis=1))-np.abs(a[indices].mean(axis=1))
    low,high=np.quantile(delta,[.025,.975])
    return dict(difference=float(abs(b.mean())-abs(a.mean())),low=float(low),high=float(high),draws=p['bootstrap_draws'])


def plot_pair(groups,level,design,dgp,left,out,p):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels={'naive_ml':'Naive ML','full_no_split':'Full No-split','equal_train_reuse':'Same-bank reuse','dml_crossfit':'DML / Cross-fitting'}
    fig,axes=plt.subplots(3,2,figsize=(12,11),layout='constrained')
    for i,family in enumerate(p['learners']):
        pair=[groups[(level,design,dgp,family,m)] for m in (left,'dml_crossfit')]
        errors=[np.array([r['bias']/r['se'] for r in g]) for g in pair]
        lo=min(-4.,*(float(x.min()) for x in errors));hi=max(4.,*(float(x.max()) for x in errors))
        edges=np.linspace(lo,hi,51);normal=np.linspace(lo,hi,1200)
        for j,(g,z,m) in enumerate(zip(pair,errors,(left,'dml_crossfit'))):
            ax=axes[i,j];ax.hist(z,bins=edges,density=True,color='#3675b5',alpha=.78,edgecolor='white',linewidth=.3)
            ax.plot(normal,stats.norm.pdf(normal),color='#c63838',lw=1.6,label='N(0,1)');ax.axvline(0,color='black',lw=.8,alpha=.5)
            bias=mean_interval([r['bias'] for r in g])
            ax.set_title(f"{family.upper()} | {labels[m]}\nBias {bias['mean']:+.4f}; MC95 [{bias['low']:+.4f}, {bias['high']:+.4f}]",fontsize=10)
            ax.set_xlim(lo,hi);ax.set_xlabel('(estimate - 0.5) / estimated SE');ax.set_ylabel('Density');ax.legend(fontsize=8)
    fig.suptitle(f"{level.upper()} | {design} / {dgp} | n={p['n']}, R={p['replications'][design]} | controlled candidate inputs",fontsize=12)
    name=f'{level}_{design}_{dgp}_{left}_hist.png';fig.savefig(out/name,dpi=170);plt.close(fig);return name


def main():
    root=ROOT/'output/exploratory';manifest,rows,warnings=load(root);p=manifest['identity']['protocol']
    out=root/'analysis';out.mkdir(exist_ok=True)
    failures=[r for r in rows if not r['finite']]
    if failures:
        atomic_json(out/'FAILURES.json',failures);raise RuntimeError('numerical failures retained; final comparison blocked')
    groups=defaultdict(list)
    for r in rows:groups[(r['level'],r['design'],r['dgp'],r['learner'],r['method'])].append(r)
    summaries=[];contrasts=[];difficulty=[]
    for key,g in sorted(groups.items()):
        g.sort(key=lambda r:r['replication']);bias=mean_interval([r['bias'] for r in g]);tol=p['bias_tolerance']
        s=dict(level=key[0],design=key[1],dgp=key[2],learner=key[3],method=key[4],R=len(g),mean_theta=float(np.mean([r['theta'] for r in g])),
            bias=bias,mc_sd=float(np.std([r['theta'] for r in g],ddof=1)),mae=float(np.mean([r['abs_error'] for r in g])),
            rmse=float(np.sqrt(np.mean([r['squared_error'] for r in g]))),mean_se=float(np.mean([r['se'] for r in g])),
            coverage=wilson(sum(r['covers'] for r in g),len(g)),practical_zero_supported=bias['low']>=-tol and bias['high']<=tol,
            nonzero_bias_detected=bias['low']>0 or bias['high']<0)
        for name in ('nuisance_rmse_g','nuisance_rmse_m','true_g_sd','true_m_sd'):s[name]=float(np.mean([r[name] for r in g]))
        if key[4]!='naive_ml':s['decomposition']={k:mean_interval([r[k] for r in g]) for k in ('weighted_oracle','weighted_first_order','weighted_second_order')}
        summaries.append(s)
    for level in p['levels']:
        for design in ('figure1','figure2'):
            for dgp in ('linear','nonlinear'):
                for family in p['learners']:
                    b=groups[(level,design,dgp,family,'dml_crossfit')]
                    for method in base.methods(design)[:-1]:
                        a=groups[(level,design,dgp,family,method)]
                        mae=[y['abs_error']-x['abs_error'] for x,y in zip(a,b)];mse=[y['squared_error']-x['squared_error'] for x,y in zip(a,b)]
                        contrasts.append(dict(level=level,design=design,dgp=dgp,learner=family,baseline=method,R=len(a),
                            theta_difference=mean_interval([y['theta']-x['theta'] for x,y in zip(a,b)]),
                            absolute_error_difference=mean_interval(mae),squared_error_difference=mean_interval(mse),
                            mae_improvement_p=improvement_p(mae),mse_improvement_p=improvement_p(mse),
                            baseline_absolute_mean_bias=abs(float(np.mean([x['bias'] for x in a]))),
                            crossfit_absolute_mean_bias=abs(float(np.mean([y['bias'] for y in b]))),
                            absolute_mean_bias_difference_bootstrap=bias_bootstrap([x['bias'] for x in a],[y['bias'] for y in b],p,(level,design,dgp,family,method))))
    for metric in ('mae','mse'):
        adjusted=holm([c[metric+'_improvement_p'] for c in contrasts])
        for c,q in zip(contrasts,adjusted):c[metric+'_holm_p']=q
    for design in ('figure1','figure2'):
        for dgp in ('linear','nonlinear'):
            for family in p['learners']:
                for method in base.methods(design):
                    a=groups[('baseline',design,dgp,family,method)];b=groups[('moderate',design,dgp,family,method)]
                    difficulty.append(dict(design=design,dgp=dgp,learner=family,method=method,R=len(a),
                        theta_difference=mean_interval([y['theta']-x['theta'] for x,y in zip(a,b)]),
                        absolute_error_difference=mean_interval([y['abs_error']-x['abs_error'] for x,y in zip(a,b)]),
                        squared_error_difference=mean_interval([y['squared_error']-x['squared_error'] for x,y in zip(a,b)]),
                        nuisance_rmse_m_difference=mean_interval([y['nuisance_rmse_m']-x['nuisance_rmse_m'] for x,y in zip(a,b)]),
                        nuisance_rmse_g_difference=mean_interval([y['nuisance_rmse_g']-x['nuisance_rmse_g'] for x,y in zip(a,b)])))
    atomic_json(out/'SUMMARY.json',dict(identity_sha256=manifest['identity_sha256'],stage=p['stage'],summaries=summaries,
        contrasts=contrasts,difficulty_contrasts=difficulty,warnings=warnings,failures=0,total_estimates=len(rows),
        individual_cell_intervals=True,holm_tests_per_metric=len(contrasts),bootstrap_bias_intervals_exploratory=True,
        controlled_known_treatment_support=True))
    figures=[]
    for level in p['levels']:
        for design in ('figure1','figure2'):
            for dgp in ('linear','nonlinear'):
                figures.append(plot_pair(groups,level,design,dgp,'naive_ml' if design=='figure1' else 'full_no_split',out,p))
                if design=='figure2':figures.append(plot_pair(groups,level,design,dgp,'equal_train_reuse',out,p))
    import matplotlib.pyplot as plt
    for level in p['levels']:
        for design in ('figure1','figure2'):
            fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
            for ax,dgp in zip(axes,('linear','nonlinear')):
                cell=[s for s in summaries if s['level']==level and s['design']==design and s['dgp']==dgp]
                for j,method in enumerate(base.methods(design)):
                    sel=[next(s for s in cell if s['learner']==f and s['method']==method) for f in p['learners']]
                    x=np.arange(3)+(j-(len(base.methods(design))-1)/2)*.16;y=[s['bias']['mean'] for s in sel]
                    errors=np.array([[s['bias']['mean']-s['bias']['low'] for s in sel],[s['bias']['high']-s['bias']['mean'] for s in sel]])
                    ax.errorbar(x,y,yerr=errors,fmt='o',capsize=4,label=method)
                ax.axhline(0,color='black',lw=.8);ax.axhspan(-.01,.01,color='green',alpha=.1)
                ax.set_xticks(np.arange(3),[f.upper() for f in p['learners']]);ax.set_title(dgp);ax.set_ylabel('Mean bias with individual MC95');ax.legend(fontsize=8)
            fig.suptitle(f'{level.upper()} | {design} | n=1000, exploratory R=200')
            name=f'{level}_{design}_bias.png';fig.savefig(out/name,dpi=170);plt.close(fig);figures.append(name)
    atomic_json(out/'FIGURE_LEDGER.json',{name:digest(out/name) for name in figures})
    print(json.dumps(dict(status='ANALYZED_EXPLORATORY',estimates=len(rows),summaries=len(summaries),contrasts=len(contrasts),figures=len(figures))))


if __name__=='__main__':main()
