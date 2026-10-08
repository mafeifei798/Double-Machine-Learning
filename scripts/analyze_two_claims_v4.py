#!/usr/bin/env python3
"""All cells, honest equivalence decisions, paired contrasts, original-style figures."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy import stats
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from dml_original.two_claims_v4 import methods
from analyze_fair_iv import mean_interval,wilson
from run_fair_iv import atomic_json,digest


def load(root):
    manifest=json.loads((root/'FROZEN_MANIFEST.json').read_text())
    status=json.loads((root/'STATUS.json').read_text());ledger=json.loads((root/'RESULT_LEDGER.json').read_text())
    if status['status']!='complete':raise RuntimeError('formal incomplete')
    protocol=manifest['identity']['protocol']
    expected={(d,g,r) for d in ('figure1','figure2') for g in ('linear','nonlinear')
        for r in range(protocol['replications'][d])}
    if set(ledger)!={p.name for p in root.glob('figure*.json')}:raise RuntimeError('disk/ledger exact cover')
    rows=[];warnings=[];seen=set()
    for name,sha in sorted(ledger.items()):
        if digest(root/name)!=sha:raise RuntimeError('receipt hash mismatch')
        receipt=json.loads((root/name).read_text());key=tuple(receipt['key'])
        if key not in expected or key in seen or receipt['identity_sha256']!=manifest['identity_sha256']:
            raise RuntimeError('universe/identity mismatch')
        seen.add(key)
        universe={(l,m) for l in protocol['learners'] for m in methods(key[0])}
        if {(r['learner'],r['method']) for r in receipt['rows']}!=universe or len(receipt['rows'])!=len(universe):
            raise RuntimeError('learner/method exact cover')
        for row in receipt['rows']:
            if (row['design'],row['dgp'],row['replication'])!=key:raise RuntimeError('row identity')
            if row['artificial_pressure'] or row['treatment_learner']!=row['learner'] or row['outcome_learner']!=row['learner']:
                raise RuntimeError('model/pressure contract')
        rows.extend(receipt['rows']);warnings.extend(receipt['warnings'])
    if seen!=expected:raise RuntimeError('missing replications')
    return manifest,rows,warnings


def plot_pair(groups,design,dgp,left,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels={'naive_ml':'Naive ML (same held-out g)','full_no_split':'No-split (1000 training rows)',
        'equal_train_reuse':'Reuse (same 500-row model bank)','dml_crossfit':'DML / 2-fold Cross-fitting'}
    fig,axes=plt.subplots(3,2,figsize=(12,11),layout='constrained')
    for i,family in enumerate(('lasso','rf','xgb')):
        pair=[groups[(design,dgp,family,m)] for m in (left,'dml_crossfit')]
        errors=[np.array([(r['theta']-r['theta0'])/r['se'] for r in group if r['finite']]) for group in pair]
        lo=min(-4.,*(float(z.min()) for z in errors));hi=max(4.,*(float(z.max()) for z in errors))
        edges=np.linspace(lo,hi,61);normal_x=np.linspace(lo,hi,1000)
        for j,(group,z,method) in enumerate(zip(pair,errors,(left,'dml_crossfit'))):
            ax=axes[i,j];ax.hist(z,bins=edges,density=True,color='#3675b5',alpha=.78,edgecolor='white',linewidth=.25)
            ax.plot(normal_x,stats.norm.pdf(normal_x),color='#c63838',lw=1.6,label='N(0,1)')
            ax.axvline(0,color='black',lw=.8,alpha=.55)
            bias=mean_interval([r['bias'] for r in group if r['finite']])
            ax.set_title(f"{family.upper()} | {labels[method]}",fontsize=11)
            ax.text(.97,.95,f"mean bias = {bias['mean']:+.4f}\nMC95 [{bias['low']:+.4f}, {bias['high']:+.4f}]",
                transform=ax.transAxes,ha='right',va='top',fontsize=9)
            ax.set_xlim(lo,hi);ax.set_xlabel('(estimate - 0.5) / estimated SE');ax.set_ylabel('Density')
            ax.legend(loc='upper left',fontsize=8)
    fig.suptitle(f"{design.upper()} / {dgp} | n=1000, R=1000 | controlled known-support treatment channel",fontsize=13)
    path=out/f'{design}_{dgp}_{left}_hist.png';fig.savefig(path,dpi=170);plt.close(fig)
    return path.name


def main():
    root=ROOT/'output/formal';manifest,rows,warnings=load(root);protocol=manifest['identity']['protocol']
    out=root/'analysis';out.mkdir(exist_ok=True);groups=defaultdict(list)
    for row in rows:groups[(row['design'],row['dgp'],row['learner'],row['method'])].append(row)
    summaries=[];contrasts=[]
    for key,group in sorted(groups.items()):
        group.sort(key=lambda r:r['replication']);finite=[r for r in group if r['finite']]
        bias=mean_interval([r['bias'] for r in finite]);t=protocol['bias_tolerance']
        summary=dict(design=key[0],dgp=key[1],learner=key[2],method=key[3],R=len(group),finite_R=len(finite),
            failures=len(group)-len(finite),mean_theta=float(np.mean([r['theta'] for r in finite])),bias=bias,
            practical_zero_supported=len(finite)==len(group) and bias['low']>=-t and bias['high']<=t,
            nonzero_bias_detected=bias['low']>0 or bias['high']<0,
            mc_sd=float(np.std([r['theta'] for r in finite],ddof=1)),
            rmse=float(np.sqrt(np.mean([r['squared_error'] for r in finite]))),
            mae=float(np.mean([r['abs_error'] for r in finite])),mean_se=float(np.mean([r['se'] for r in finite])),
            coverage=wilson(sum(r['covers'] for r in finite),len(finite)),
            nuisance_rmse_g=float(np.mean([r['nuisance_rmse_g'] for r in finite])),
            nuisance_rmse_m=float(np.mean([r['nuisance_rmse_m'] for r in finite])))
        if key[3]!='naive_ml':
            summary['decomposition']={k:mean_interval([r[k] for r in finite]) for k in
                ('weighted_oracle','weighted_first_order','weighted_second_order')}
        summaries.append(summary)
    for design in ('figure1','figure2'):
        for dgp in ('linear','nonlinear'):
            for family in protocol['learners']:
                b=groups[(design,dgp,family,'dml_crossfit')]
                for method in methods(design)[:-1]:
                    a=groups[(design,dgp,family,method)];paired=[(x,y) for x,y in zip(a,b) if x['finite'] and y['finite']]
                    contrasts.append(dict(design=design,dgp=dgp,learner=family,baseline=method,
                        R=len(paired),theta_difference=mean_interval([y['theta']-x['theta'] for x,y in paired]),
                        absolute_error_difference=mean_interval([y['abs_error']-x['abs_error'] for x,y in paired]),
                        squared_error_difference=mean_interval([y['squared_error']-x['squared_error'] for x,y in paired]),
                        baseline_absolute_mean_bias=abs(float(np.mean([x['bias'] for x,y in paired]))),
                        crossfit_absolute_mean_bias=abs(float(np.mean([y['bias'] for x,y in paired])))))
    atomic_json(out/'SUMMARY.json',dict(identity_sha256=manifest['identity_sha256'],summaries=summaries,
        individual_cell_intervals=True,joint_95_percent_claim=False,contrasts=contrasts,warnings=warnings,
        total_estimates=len(rows),failures=sum(not r['finite'] for r in rows),
        cf_practical_zero_count=sum(s['practical_zero_supported'] for s in summaries if s['method']=='dml_crossfit'),
        cf_total=12,controlled_known_treatment_support=True))
    figures=[]
    for design in ('figure1','figure2'):
        for dgp in ('linear','nonlinear'):
            figures.append(plot_pair(groups,design,dgp,'naive_ml' if design=='figure1' else 'full_no_split',out))
            if design=='figure2':figures.append(plot_pair(groups,design,dgp,'equal_train_reuse',out))
    import matplotlib.pyplot as plt
    for design in ('figure1','figure2'):
        fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
        for ax,dgp in zip(axes,('linear','nonlinear')):
            cell=[s for s in summaries if s['design']==design and s['dgp']==dgp]
            for j,method in enumerate(methods(design)):
                sel=[next(s for s in cell if s['learner']==l and s['method']==method) for l in protocol['learners']]
                x=np.arange(3)+(j-(len(methods(design))-1)/2)*.16
                y=[s['bias']['mean'] for s in sel];err=np.array([[s['bias']['mean']-s['bias']['low'] for s in sel],
                    [s['bias']['high']-s['bias']['mean'] for s in sel]])
                ax.errorbar(x,y,yerr=err,fmt='o',capsize=4,label=method)
            ax.axhline(0,color='black',lw=1);ax.axhspan(-.01,.01,color='green',alpha=.1)
            ax.set_xticks(np.arange(3),[l.upper() for l in protocol['learners']]);ax.set_title(dgp)
            ax.set_ylabel('Mean bias with individual MC95 interval');ax.legend(fontsize=8)
        fig.suptitle(design+' | all three families | controlled treatment structure')
        name=design+'_bias.png';fig.savefig(out/name,dpi=170);plt.close(fig);figures.append(name)
    atomic_json(out/'FIGURE_LEDGER.json',{name:digest(out/name) for name in figures})
    print(json.dumps(dict(status='ANALYZED',estimates=len(rows),summaries=len(summaries),figures=len(figures))))


if __name__=='__main__':main()
