#!/usr/bin/env python3
"""Audit SAVED estimates, then render the completed simulation chapter. Never fit ML."""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import html
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tarfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether

REPO = Path(__file__).resolve().parents[1]
NAMES = {'rf':'Random Forest', 'xgb':'XGBoost', 'lasso':'Lasso'}
ORDER = ('rf','xgb','lasso')
THETA = .5


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def verify_row(r):
    assert r['finite'] and r['se'] > 0 and r['p'] == 20 and r['theta0'] == THETA
    assert r['artificial_pressure'] is False
    error = r['theta'] - THETA
    assert abs(error-r['bias']) < 1e-12
    assert abs(abs(error)-r['abs_error']) < 1e-12
    assert abs(error*error-r['squared_error']) < 1e-12
    assert r['covers'] == (abs(error) <= 1.96*r['se'])
    if r['method'] not in ('naive_ml','nonorthogonal_crossfit'):
        assert abs(sum(r[k] for k in ('weighted_oracle','weighted_first_order','weighted_second_order'))-error) < 1e-10


def load_bridge(root, archive):
    frozen = json.loads((root/'BRIDGE_FROZEN.json').read_text())
    ident = frozen['identity_sha256']
    assert hashlib.sha256(json.dumps(frozen['identity'],sort_keys=True).encode()).hexdigest() == ident
    source_archive = archive.parent/'server_source_v7.tgz'
    metadata_count = 0
    with tarfile.open(source_archive) as source:
        for name, expected in frozen['identity']['source_hashes'].items():
            # macOS AppleDouble entries are metadata, not imported Python modules.
            if Path(name).name.startswith('._'):
                data = source.extractfile(name).read()
                metadata_count += 1
            else:
                data = (REPO/name).read_bytes()
            assert hashlib.sha256(data).hexdigest() == expected, name
    status = json.loads((root/'BRIDGE_STATUS.json').read_text())
    exit_receipt = json.loads((root/'BRIDGE_EXIT.json').read_text())
    assert status['status']=='complete' and status['completed_jobs']==status['expected_jobs']==600
    assert exit_receipt['returncode']==0
    ledger = json.loads((root/'output/legacy_bridge_v7/RESULT_LEDGER.json').read_text())
    assert len(ledger)==600
    rows=[]
    warnings=[]
    expected={(d,r) for d in ('linear','nonlinear') for r in range(300)}
    seen=set()
    for name, expected_sha in sorted(ledger.items()):
        path=root/'output/legacy_bridge_v7'/name
        assert digest(path)==expected_sha
        receipt=json.loads(path.read_text())
        assert receipt['identity_sha256']==ident and len(receipt['rows'])==6
        key=(receipt['key'][1],receipt['key'][-1]);assert key in expected and key not in seen;seen.add(key)
        for r in receipt['rows']:
            verify_row(r)
            assert (r['level'],r['n'],r['k_folds'],r['preset'],r['R'] if 'R' in r else 300)==('baseline',500,2,'base',300)
            assert r['design']=='figure1' and r['prediction_overlap_n']==0 and r['train_n_per_model']==250
        for learner in ORDER:
            pair={r['method']:r for r in receipt['rows'] if r['learner']==learner}
            a,b=pair['naive_ml'],pair['dml_crossfit']
            assert all(a[k]==b[k] for k in ('bank_identity','g_prediction_hash','m_prediction_hash','fold_hash'))
        rows.extend(receipt['rows']);warnings.extend(receipt['warnings'])
    assert seen==expected and len(rows)==3600
    return rows, {'identity_sha256':ident,'server_host':json.loads((root/'BRIDGE_START.json').read_text())['host'],
                  'jobs':600,'estimates':3600,'actual_exit':exit_receipt,'warnings':len(warnings),
                  'source_hashes_verified':len(frozen['identity']['source_hashes']),
                  'appledouble_metadata_entries_verified_not_executable_sources':metadata_count,
                  'result_archive_sha256':digest(archive),'source_archive_sha256':digest(source_archive)}


def load_v6(root):
    formal=root/'output/formal'
    frozen=json.loads((formal/'FROZEN_MANIFEST.json').read_text());ident=frozen['identity_sha256']
    assert hashlib.sha256(json.dumps(frozen['identity'],sort_keys=True).encode()).hexdigest()==ident
    scientific=0
    for name, expected in frozen['identity']['source_hashes'].items():
        assert digest(root/name)==expected,name
        scientific+=1
    ledger=json.loads((formal/'RESULT_LEDGER.json').read_text())
    rows=[];inputs={}
    for dgp in ('linear','nonlinear'):
        for rep in range(200):
            name=f'figure2_{dgp}_n300_k2_base_rep{rep:04d}.json'
            path=formal/name;assert digest(path)==ledger[name];inputs[name]=ledger[name]
            receipt=json.loads(path.read_text());assert receipt['identity_sha256']==ident
            selected=[r for r in receipt['rows'] if r['level']=='baseline']
            assert len(selected)==9
            for r in selected:
                verify_row(r)
                assert (r['design'],r['dgp'],r['n'],r['k_folds'],r['preset'],r['replication'])==('figure2',dgp,300,2,'base',rep)
                assert r['prediction_overlap_n']==(0 if r['method']=='dml_crossfit' else 300)
                assert r['train_n_per_model']==(300 if r['method']=='full_no_split' else 150)
            for learner in ORDER:
                pair={r['method']:r for r in selected if r['learner']==learner}
                assert pair['equal_train_reuse']['bank_identity']==pair['dml_crossfit']['bank_identity']
                assert pair['equal_train_reuse']['fold_hash']==pair['dml_crossfit']['fold_hash']
            rows.extend(selected)
    assert len(rows)==3600
    return rows,{'identity_sha256':ident,'reused_jobs':400,'selected_baseline_estimates':3600,
                'source_hashes_verified':scientific,'input_receipt_hashes':inputs,'new_model_fits':0}


def summarize(rows):
    groups=defaultdict(list)
    for r in rows:groups[(r['design'],r['dgp'],r['learner'],r['method'])].append(r)
    summaries=[]
    for key,g in sorted(groups.items()):
        g.sort(key=lambda r:r['replication']);R=len(g)
        assert [r['replication'] for r in g]==list(range(R))
        values=np.array([r['theta'] for r in g]);se=np.array([r['se'] for r in g]);error=values-THETA
        sd=float(values.std(ddof=1));bias=float(error.mean());critical=float(t.ppf(.975,R-1))
        summaries.append(dict(design=key[0],dgp=key[1],learner=key[2],method=key[3],n=g[0]['n'],R=R,
            mean_theta=float(values.mean()),bias=bias,mc_sd=sd,mean_se=float(se.mean()),
            mc_mean_bias_ci=[bias-critical*sd/math.sqrt(R),bias+critical*sd/math.sqrt(R)],
            mc_sampling_spread=[float(values.mean()-1.96*sd),float(values.mean()+1.96*sd)],
            coverage=float(np.mean(abs(error)<=1.96*se)),mae=float(abs(error).mean()),
            rmse=float(np.sqrt(np.mean(error*error)))))
    return groups,summaries


def assert_v6_summary_match(rows,source):
    for r in rows:
        matches=[x for x in source['summaries'] if all(x[k]==r[k] for k in ('design','dgp','learner','method','n'))
                 and x['k_folds']==2 and x['level']=='baseline' and x['preset']=='base']
        assert len(matches)==1
        x=matches[0]
        for field in ('mean_theta','mc_sd','mean_se','mae','rmse'):
            assert abs(r[field]-x[field])<1e-12,(field,r,x)
        assert abs(r['bias']-x['bias']['mean'])<1e-12 and r['coverage']==x['coverage']['rate']


def normal_pdf(x,sigma=1):
    return np.exp(-.5*(x/sigma)**2)/(sigma*math.sqrt(2*math.pi))


def histogram(groups,design,dgp,path,standardized):
    left='naive_ml' if design=='figure1' else 'full_no_split'
    methods=(left,'dml_crossfit');labels=('Non-Orthogonal ML','IV-type DML') if design=='figure1' else ('No Split','2-fold Cross-Fitting')
    n=500 if design=='figure1' else 300;R=300 if design=='figure1' else 200
    arrays={}
    for learner in ORDER:
        for method in methods:
            g=groups[(design,dgp,learner,method)]
            arrays[(learner,method)]=np.array([(r['theta']-THETA)/(r['se'] if standardized else 1) for r in g])
    fig,axes=plt.subplots(3,2,figsize=(11.2,12),dpi=180)
    for i,learner in enumerate(ORDER):
        bound=max(4.5 if standardized else .35,*(abs(arrays[(learner,m)]).max()*1.06 for m in methods))
        grid=np.linspace(-bound,bound,1000)
        sigma=1 if standardized else arrays[(learner,'dml_crossfit')].std(ddof=1)
        reference=normal_pdf(grid,sigma)
        for j,method in enumerate(methods):
            ax=axes[i,j];values=arrays[(learner,method)]
            counts,_,_=ax.hist(values,bins=30,density=True,color='#8ecae6',edgecolor='#457b9d',alpha=.9,label='Simulation')
            ax.plot(grid,reference,'r-',lw=1.1,label='N(0, 1)' if standardized else 'N(0, DML MC variance)')
            ax.axvline(0,color='black',ls=':',lw=1,alpha=.8)
            ax.set(xlim=(-bound,bound),ylim=(0,1.17*max(counts.max(),reference.max())),
                   xlabel=r'$(\hat{\theta}-\theta_0)/\widehat{SE}$' if standardized else r'$\hat{\theta}-\theta_0$')
            if j==0:ax.set_ylabel('Density')
            ax.set_title(NAMES[learner]+' - '+labels[j],fontsize=11)
            ax.grid(alpha=.24)
            g=groups[(design,dgp,learner,method)];bias=np.mean([r['bias'] for r in g])
            ax.text(.98,.94,f'Bias = {bias:+.4f}',ha='right',va='top',transform=ax.transAxes,fontsize=9)
            ax.legend(loc='upper left',fontsize=7,frameon=True)
    fig.suptitle(f'{dgp.capitalize()} DGP: {labels[0]} vs. {labels[1]}\nn={n}, p=20, R={R}; '+('studentized errors' if standardized else 'raw estimation errors'),fontsize=13,y=.995)
    fig.tight_layout(rect=[0,0,1,.96]);fig.savefig(path,bbox_inches='tight');plt.close(fig)


def learner_plot(summaries,dgp,path):
    rows=[next(r for r in summaries if r['design']=='figure1' and r['dgp']==dgp and r['learner']==l and r['method']=='dml_crossfit') for l in ORDER]
    means=[r['mean_theta'] for r in rows];spread=[1.96*r['mc_sd'] for r in rows]
    fig,ax=plt.subplots(figsize=(10.2,4.2),dpi=180)
    ax.barh(range(3),means,color=['#6a9bcc','#2a9d8f','#d97757'],edgecolor='white',height=.55,label=r'Mean $\hat{\theta}$ (MC)')
    ax.errorbar(means,range(3),xerr=spread,fmt='none',ecolor='#141413',capsize=4,lw=1.8,label=r'$\pm 1.96\times$ MC SD')
    ax.axvline(THETA,color='black',ls='--',lw=1,alpha=.85,label=r'True $\theta_0=0.5$')
    ax.set(yticks=range(3),yticklabels=['IV-type DML ('+NAMES[l]+')' for l in ORDER],
           xlabel=r'$\hat{\theta}$',xlim=(0,max(.7,max(np.array(means)+spread)+.03)),
           title=f'Sensitivity: learner choice ({dgp.capitalize()} DGP, n=500, p=20, R=300)')
    ax.grid(axis='x',alpha=.25);ax.legend(loc='center left',bbox_to_anchor=(1.01,.5),fontsize=8,frameon=True)
    fig.tight_layout();fig.savefig(path,bbox_inches='tight');plt.close(fig)


def comparison_table(summaries,design,dgp,baseline):
    rows=[['Learner','Comparator','','','IV-type DML','',''],['','Bias','MC SD','Mean SE','Bias','MC SD','Mean SE']]
    for learner in ORDER:
        a=next(r for r in summaries if (r['design'],r['dgp'],r['learner'],r['method'])==(design,dgp,learner,baseline))
        b=next(r for r in summaries if (r['design'],r['dgp'],r['learner'],r['method'])==(design,dgp,learner,'dml_crossfit'))
        rows.append([NAMES[learner]]+[f'{r[k]:.4f}' for r in (a,b) for k in ('bias','mc_sd','mean_se')])
    rows[0][1]='Non-Orthogonal ML' if design=='figure1' else 'Full-Sample No Split' if baseline=='full_no_split' else 'Equal-Training Reuse'
    return rows


def sensitivity_table(summaries,dgp):
    rows=[['Learner','Mean estimate','Bias','MC SD','95% MC spread','Coverage']]
    for learner in ORDER:
        r=next(r for r in summaries if (r['design'],r['dgp'],r['learner'],r['method'])==('figure1',dgp,learner,'dml_crossfit'))
        rows.append([NAMES[learner],f'{r["mean_theta"]:.4f}',f'{r["bias"]:.4f}',f'{r["mc_sd"]:.4f}',
                     f'[{r["mc_sampling_spread"][0]:.4f}, {r["mc_sampling_spread"][1]:.4f}]',f'{100*r["coverage"]:.1f}%'])
    return rows


def table_image(rows,caption,path,grouped=False):
    fig=plt.figure(figsize=(11.2,2.5 if grouped else 2.2),dpi=180);ax=fig.add_axes([.02,.03,.96,.76]);ax.axis('off')
    fig.text(.5,.93,caption,ha='center',va='top',fontsize=12)
    table=ax.table(cellText=rows,cellLoc='center',loc='center',bbox=[0,.05,1,.9])
    table.auto_set_font_size(False);table.set_fontsize(10)
    for (i,j),cell in table.get_celld().items():
        cell.set_facecolor('white');cell.set_edgecolor('black');cell.set_linewidth(.6);cell.visible_edges=''
        if i==0:cell.visible_edges='T'
        if i==(1 if grouped else 0):cell.visible_edges+='B'
        if i==len(rows)-1:cell.visible_edges+='B'
    if grouped:
        # Merge group headings visually without changing their numerical columns.
        for j in (1,3,4,6):table[(0,j)].get_text().set_text('')
        table[(0,2)].get_text().set_text(rows[0][1]);table[(0,5)].get_text().set_text(rows[0][4])
    fig.savefig(path,bbox_inches='tight');plt.close(fig)


BODY=ParagraphStyle('Body',fontName='Times-Roman',fontSize=11,leading=14.8,spaceAfter=8,alignment=TA_JUSTIFY)
TITLE=ParagraphStyle('Title',parent=BODY,fontName='Times-Bold',fontSize=17,leading=21,spaceAfter=14)
HEAD=ParagraphStyle('Head',parent=BODY,fontName='Times-Bold',fontSize=14,leading=18,spaceBefore=8,spaceAfter=10)
SUB=ParagraphStyle('Sub',parent=HEAD,fontSize=12,leading=16)
CAPTION=ParagraphStyle('Caption',parent=BODY,fontSize=10.5,leading=13,alignment=TA_CENTER,spaceBefore=7)
SMALL=ParagraphStyle('Small',parent=BODY,fontSize=9,leading=11.5)


def p(text,style=BODY):
    return Paragraph(text,style)


def pdf_table(rows,grouped=False):
    widths=[72]+[62]*6 if grouped else [73,65,57,61,138,50]
    data=[[p(html.escape(str(c)),ParagraphStyle('Cell',parent=SMALL,alignment=TA_CENTER)) for c in row] for row in rows]
    table=Table(data,colWidths=widths,hAlign='CENTER',repeatRows=2 if grouped else 1)
    commands=[('LINEABOVE',(0,0),(-1,0),.85,colors.black),('LINEBELOW',(0,1 if grouped else 0),(-1,1 if grouped else 0),.5,colors.black),
              ('LINEBELOW',(0,-1),(-1,-1),.85,colors.black),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),('VALIGN',(0,0),(-1,-1),'MIDDLE')]
    if grouped:commands += [('SPAN',(1,0),(3,0)),('SPAN',(4,0),(6,0)),('SPAN',(0,0),(0,1))]
    table.setStyle(TableStyle(commands));return table


def scaled_image(path,maxw=480,maxh=600):
    im=Image(str(path));ratio=min(maxw/im.imageWidth,maxh/im.imageHeight)
    im.drawWidth=im.imageWidth*ratio;im.drawHeight=im.imageHeight*ratio;im.hAlign='CENTER';return im


def equation(text,name,out):
    path=out/'equations'/f'{name}.png';path.parent.mkdir(exist_ok=True)
    fig=plt.figure(figsize=(10,.6),dpi=220);fig.text(.5,.5,'$'+text+'$',ha='center',va='center',fontsize=17)
    plt.axis('off');fig.savefig(path,bbox_inches='tight',pad_inches=.06);plt.close(fig)
    return scaled_image(path,450,48)


def make_pdf(out,tables,figures,summaries,v6,nu,audit,extra_figures):
    story=[]
    def add(text,style=BODY):story.append(p(text,style))
    def eq(tex,name):story.append(equation(tex,name,out));story.append(Spacer(1,7))
    def page():story.append(PageBreak())
    def tbl(i):
        spec=tables[i];story.extend([p(spec['caption'],CAPTION),pdf_table(spec['rows'],spec['grouped']),Spacer(1,12)])
    add('3 Simulation: Repaired and Completed',TITLE)
    add('Scope: the simulation contribution only. The empirical examples in the original group report are outside this chapter. The table and figure numbering 1-6 follows the original simulation presentation; the estimates are not copied from the old report.')
    add('3.1 Setup and Objective',HEAD)
    add('We examine two questions: whether a Neyman-orthogonal IV-type score reduces nuisance-estimation bias relative to a non-orthogonal ML ratio, and whether out-of-sample nuisance predictions reduce training-data reuse bias. Random Forest, XGBoost, and Lasso are evaluated under both linear and nonlinear DGPs.')
    eq(r'Y_i=\theta_0D_i+g_0(X_i)+U_i,\qquad D_i=m_0(X_i)+V_i','dgp_model')
    add('The true effect is 0.5, p=20, and X is Gaussian with Toeplitz covariance Cov(X<sub>j</sub>, X<sub>k</sub>)=0.7<super>|j-k|</super>. U and V are mutually independent standard Gaussian disturbances, independent of X. The repaired orthogonalization bridge uses n=500, K=2, R=300. The cross-fitting comparison reuses the completed v6 baseline n=300, K=2, R=200 receipts without new fitting.')
    add('3.2 Data Generating Processes',HEAD)
    add('The two experiments use different baseline coefficients. This distinction was missing from the original report: its unified DGP description matched the old Figure 2 code, but not the archived Figure 1 code. Here X<sub>1</sub> and X<sub>3</sub> correspond to Python columns 0 and 2; s(t)=1/(1+exp(-t)).')
    dgp_rows=[['Experiment / DGP','m0(X)','g0(X)'],['Orthogonalization / linear','0.9 X1 + 0.20 X3','0.25 X1 + 0.25 X3'],['Orthogonalization / nonlinear','0.9 X1 + 0.20 s(X3)','s(X1) + 0.25 X3'],['Cross-fitting / linear','1.2 X1 + 0.35 X3','0.40 X1 + 0.40 X3'],['Cross-fitting / nonlinear','1.2 X1 + 0.35 s(X3)','1.2 s(X1) + 0.40 X3']]
    dt=Table([[p(html.escape(c),SMALL) for c in row] for row in dgp_rows],colWidths=[160,157,157]);dt.setStyle(TableStyle([('LINEABOVE',(0,0),(-1,0),.8,colors.black),('LINEBELOW',(0,0),(-1,0),.5,colors.black),('LINEBELOW',(0,-1),(-1,-1),.8,colors.black),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]));story.append(dt)
    page()
    add('3.3 Estimation and Reporting',HEAD)
    add('Correct nuisance targets',SUB)
    add('The outcome regression and structural nuisance are different objects. In particular, learning E[Y|X] directly and then calling it structural g does not yield the claimed Neyman-orthogonal IV score. The repaired definitions are:')
    eq(r'm_0(X)=E[D\mid X],\quad g_0(X)=E[Y-\theta_0D\mid X]','targets')
    eq(r'\ell_0(X)=E[Y\mid X]=g_0(X)+\theta_0m_0(X)','ell_target')
    add('For each training fold, a joint Lasso pilot is fitted using D, all raw X, and sigmoid(X), with alpha=0.001. In the orthogonalization experiment, g is learned from the training response Y-pilot*D. In the reuse experiment, ell is learned from Y and g_hat=ell_hat-pilot*m_hat. The pilot, scalers, and nuisance models never use the held-out outcomes. True functions are used only for simulation generation and post-fit diagnostics.')
    eq(r'\widehat V_i=D_i-\widehat m_i,\quad\widehat\theta_{IV}=\frac{\sum_i\widehat V_i(Y_i-\widehat g_i)}{\sum_i\widehat V_iD_i}','iv_score')
    eq(r'\widehat\theta_{ML}=\frac{\sum_iD_i(Y_i-\widehat g_i)}{\sum_iD_i^2}','naive_score')
    add('The naive and IV estimators in Tables 1-2 share exactly the same held-out structural-g predictions. Their difference is the score, not different nuisance training or added pressure. The IV-type score is used throughout; no partialling-out estimator is restored.')
    add('Nuisance learners and controlled prior information',SUB)
    add('Treatment models use fixed candidate columns [0,2,4,6], including two inactive baseline candidates. RF and XGBoost treatment models use positive monotonicity constraints; treatment Lasso uses positive coefficients. Outcome RF/XGBoost use all raw covariates; nonlinear outcome Lasso uses a fixed raw/sigmoid dictionary. These are disclosed controlled priors, not unrestricted variable discovery.')
    add('RF: 300 trees; treatment leaf size 10, unrestricted depth, all candidate features; structural-g depth 8/leaf 2/sqrt features; ell unrestricted depth/leaf 1. XGBoost: treatment 300 trees/depth 1/rate .05/lambda 5; g 300/depth 3/rate .01; ell 600/depth 6/rate .1. Lasso alpha: treatment .001; g .02 linear/.06 nonlinear; ell .0001. All settings are inherited unchanged from the current mainline.')
    page()
    add('3.3 Estimation and Reporting (continued)',HEAD)
    add('Uncertainty and histogram definitions',SUB)
    eq(r'\psi_i=\widehat V_i(Y_i-\widehat g_i-\widehat\theta_{IV}D_i),\quad\widehat{SE}=\sqrt{\frac{\overline{\psi_i^2}}{n(\overline{\widehat V_iD_i})^2}}','se')
    add('Bias is the Monte Carlo average of theta_hat-0.5. MC SD is the sample standard deviation of the R estimates. Mean SE averages the per-replication sandwich SE. Coverage is the fraction of intervals theta_hat +/- 1.96*SE containing 0.5. MAE and RMSE are retained in the machine-readable summaries but are not conflated with absolute mean bias.')
    add('Figures 1-2 use raw errors theta_hat-0.5. Both panels in a learner row share a zero-centered normal reference whose SD equals that learner\'s DML MC SD, matching the original visual convention. Figures 5-6 use studentized errors (theta_hat-0.5)/SE and a standard-normal reference. Every observed error is included; paired panel axes have identical bounds.')
    add('Figures 3-4 and Tables 3-4 reproduce the original mean +/- 1.96*MC SD display. This is an approximate sampling spread of individual estimates, not a confidence interval for the Monte Carlo mean and not the per-replication causal confidence interval. Monte Carlo uncertainty about mean bias is reported separately as mean bias +/- t<sub>R-1,.975</sub>*MC SD/sqrt(R).')
    add('What this bridge does and does not replicate',SUB)
    add('The n=500 experiment restores the old sample size and repetition count, and follows the archived Figure 1 DGP code. It changes the historical mismatched nuisance target to the repaired structural IV target and retains the current controlled learner settings. Therefore it is a repaired legacy-size bridge, not a literal reproduction of the old biased procedure or its printed numerical values.')
    add('A separate prespecified seed was fixed before observing any bridge result. All six learner-by-DGP cells and both methods are reported. No seed, learner, repetition count, or hyperparameter was selected using causal-effect accuracy. The server software preflight checked finiteness, identical model-bank inputs, absence of prediction overlap, and the algebraic error decomposition; low bias or a method win was not a gate.')
    page()
    add('3.3.1 Orthogonalization and Bias Reduction',HEAD)
    add('The following tables preserve the original layout and show the repaired n=500, R=300 results. Interpret the size and uncertainty of the remaining bias separately from its reduction relative to non-orthogonal ML.')
    tbl(1);tbl(2)
    add('Bias is signed. Neither small displayed bias nor failure to detect a nonzero bias proves exact finite-sample unbiasedness. Learner-specific residual bias is retained, rather than removed by post-result tuning.')
    add('A small bias-variance tradeoff is present even in this bridge: for linear Lasso, absolute mean bias decreases from 0.00652 to 0.00436, but MAE changes from 0.034659 to 0.034771 and RMSE from 0.043406 to 0.043454. These small observed changes are not evidence of a statistically established deterioration, but they rule out describing every error metric as uniformly improved.')
    for i in (1,2):
        page();story.append(scaled_image(out/figures[i],480,615));add(f'Figure {i}: '+('Linear' if i==1 else 'Nonlinear')+' DGP. Raw estimation errors: non-orthogonal ML versus repaired IV-type DML. n=500, p=20, R=300.',CAPTION)
    page();add('3.3.2 Sensitivity to Learner Choice',HEAD);tbl(3);tbl(4)
    add('The interval column is labelled MC spread to correct the ambiguity of the original 95% CI heading. Coverage still uses the individual sandwich-SE intervals. Differences in MC SD describe variance and cannot by themselves establish which learner is uniformly more accurate.')
    page()
    for i in (3,4):
        story.append(scaled_image(out/figures[i],480,245));add(f'Figure {i}: Sensitivity of repaired IV-type estimates to learner choice under the '+('linear' if i==3 else 'nonlinear')+' DGP. Error bars show +/- 1.96*MC SD, not the MC-mean confidence interval.',CAPTION);story.append(Spacer(1,14))
    page();add('3.3.3 Cross-Fitting and Training-Data Reuse',HEAD);tbl(5);tbl(6)
    add('These are the existing v6 baseline n=300/K=2/base results, recomputed directly from 400 hash-verified receipts. No n=300 model was refitted. Full no-split trains on n observations, whereas each cross-fit model trains on n/2. This conventional contrast changes both prediction membership and training size; Tables 7-8 provide the separate same-bank control.')
    add('Cross-fitting substantially reduces the displayed bias but does not make every setting unbiased: linear RF retains +0.01069, linear XGBoost -0.01164, and nonlinear XGBoost -0.01349 mean bias. These cells are reported without retuning or replacing the completed experiment.')
    for i in (5,6):
        page();story.append(scaled_image(out/figures[i],480,615));add(f'Figure {i}: '+('Linear' if i==5 else 'Nonlinear')+' DGP. Full-sample no-split versus 2-fold cross-fitting; studentized estimation errors. n=300, p=20, R=200.',CAPTION)
    page();add('3.4 Controlled Extensions',HEAD)
    add('3.4.1 Equal-training-size, shared-model-bank control',SUB)
    add('The same two fitted nuisance model banks predict every observation. Cross-fitting selects a model trained on the other half; reuse selects the model trained on that observation\'s own half. Training sizes, fitted models, targets, pilot settings, and seeds are identical. No artificial own-response pressure is introduced only into no-split.')
    tbl(7);tbl(8)
    add('This controlled comparison isolates membership in the nuisance training sample more directly than full-sample no-split. Its results remain separate from the conventional full-sample comparison.')
    page();add('3.4.2 Correctly specified treatment channel',SUB)
    add('A separate completed n=1,000 study uses an unregularized Series OLS treatment model on [1, X1, X3, sigmoid(X1), sigmoid(X3)] and decision-tree outcome learners with minimum leaf sizes 2, 5, 10, and 20. It isolates the reuse mechanism in a correctly specified nuisance channel and is not substituted for the RF/XGBoost/Lasso experiments.')
    controlled=[r for r in nu if r['design']=='figure2' and r['track']=='controlled' and r['method']=='crossfit' and r['learner'].startswith('tree_leaf_')]
    rows=[['DGP / leaf','Mean estimate','Bias','MC SD','Mean SE','Coverage']]
    for r in sorted(controlled,key=lambda x:(x['dgp'],int(x['learner'].split('_')[-1]))):
        rows.append([r['dgp']+' / '+r['learner'].split('_')[-1],f'{r["mean_theta"]:.4f}',f'{r["bias"]["mean"]:.4f}',f'{r["mc_sd"]:.4f}',f'{r["mean_se"]:.4f}',f'{r["coverage"]["rate"]*100:.1f}%'])
    tab=Table([[p(html.escape(c),SMALL) for c in row] for row in rows],colWidths=[104,80,67,68,75,65]);tab.setStyle(TableStyle([('LINEABOVE',(0,0),(-1,0),.8,colors.black),('LINEBELOW',(0,0),(-1,0),.5,colors.black),('LINEBELOW',(0,-1),(-1,-1),.8,colors.black),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]));story.append(tab)
    add('Maximum absolute MC mean bias across these eight cross-fit settings is 0.00457467. This is a point-bias summary, not proof that every setting\'s entire bias-uncertainty interval lies inside +/-0.01. In particular, the nonlinear leaf-5 setting does not pass that stronger interval criterion.')
    for name in ('controlled_figure2_linear.png','controlled_figure2_nonlinear.png'):
        page();story.append(scaled_image(out/'extensions'/name,480,615));add('Completed controlled Series OLS / decision-tree study: '+('linear' if 'linear.png' in name and 'nonlinear' not in name else 'nonlinear')+' DGP. Existing source figure, unchanged.',CAPTION)
    page();add('3.4.3 Sample Size, Folds, and Hyperparameters',SUB)
    add('The completed sensitivity study contains 600 reported summary cells: n=300/1,000/3,000, K=2/5, baseline and moderate DGP levels, and four one-nuisance-role-at-a-time parameter alternatives at n=1,000/K=2. It is not a full factorial hyperparameter grid. The complete JSON retains all adverse and inconclusive cells, paired MAE/MSE contrasts with the original Holm corrections, and nuisance-error decompositions.')
    add('Some reported settings are aliases rather than independent experiments: full-sample no-split does not depend on K, and the RF Figure 2 weak-outcome preset equals the baseline ell leaf-size setting. The count 600 is a count of reported summary cells, not 600 independent experimental configurations.')
    add('The exact moderate additions below apply to each experiment\'s own baseline nuisance functions (superscripts M and B denote moderate and baseline). Baseline and moderate comparisons share simulation draws. The Lasso raw/sigmoid dictionary does not contain the nonlinear interaction, providing a disclosed limitation rather than an arbitrary-misspecification guarantee.')
    eq(r'm_0^M=m_0^B+0.20X_5+0.15X_7,\quad g_0^M=g_0^B+0.15X_5+0.15X_7','moderate_linear')
    eq(r'm_0^M=m_0^B+0.20X_5+0.20s(X_7)','moderate_nonlinear_m')
    eq(r'g_0^M=g_0^B+0.25s(X_5)+0.20X_7+0.30\tanh(X_1)\tanh(X_5)','moderate_nonlinear_g')
    add('At the moderate n=1,000/K=2/base anchor, the equal-weight average absolute MC mean bias over six learner-by-DGP cells falls from 0.0365543 to 0.00387464 for non-orthogonal ML versus DML (89.4%), and from 0.255777 to 0.00648017 for full no-split versus cross-fitting (97.5%). These percentages are specific to that anchor, not averages over the whole sensitivity grid or MAE reductions.')
    add('Important adverse findings are retained: moderate linear RF Figure 1 DML bias is +0.01275; moderate linear Figure 2 RF bias is +0.01151; nonlinear Figure 2 XGBoost bias is -0.01209. Cross-fit coverage in the sensitivity grid can fall to 87.5%. In the moderate linear Lasso reuse example, bias decreases but MAE and RMSE increase. Five folds are not uniformly better than two.')
    add('The accompanying HTML gallery includes every existing selected-stage sensitivity figure, all 600 tabulated cells, all six restored primary figures, and all eight restored comparison/sensitivity tables. Complete numerical summaries are included without changing the source values.')
    page()
    for name in ('moderate_figure2_sample_fold_bias.png','moderate_figure2_parameters_coverage.png'):
        story.append(scaled_image(out/'extensions'/name,480,290));add('Existing sensitivity evidence: '+name.replace('_',' ').replace('.png','')+'. Layout-only source figure.',CAPTION);story.append(Spacer(1,14))
    page();add('3.5 Interpretation, Limitations, and Provenance',HEAD)
    add('The experiments evaluate bias reduction, training-data reuse, variance, and interval calibration. They do not establish universal finite-sample unbiasedness, uniformly lower total error, or valid coverage for every learner. Histograms provide visual distribution diagnostics, not a formal proof of asymptotic normality. The known candidate treatment features, monotonicity restrictions, and correctly specified Series OLS control are explicit scope restrictions.')
    add('The original report\'s empirical bonus-policy and smoking examples are not rerun or altered. No IRM experiment or partialling-out estimator is added to the simulation contribution. Historical adverse ordinary-learner results remain in the existing nuisance-comparison summary, alongside the controlled results.')
    add('Reproduction boundary',SUB)
    add('New model fitting occurred only on the authorized Linux server, in one launch with 12 CPU workers and no GPU use. The new protocol fixed n=500, R=300, K=2, two DGPs, three learner families, and one seed before outcomes. Every one of the 600 jobs and 3,600 estimates is retained. The actual server exit was zero. Local work reads saved estimates and produces diagnostics, figures, tables, and this PDF; it imports no ML learner libraries.')
    add('The n=300 tables were independently recomputed from the original v6 receipts and match its saved summaries to numerical precision. Figure 1 bridge nuisance bank hashes match between naive and DML. Figure 2 equal-training reuse and cross-fit share fitted-bank identities. Receipt hashes, frozen source hashes, the IV algebraic error decomposition, coverage indicators, and complete replication indices are checked before rendering.')
    add('Evidence identifiers',SUB)
    add('Base repository commit: b01dd19f22d44c3ca5dd6771772d1cb8f31fb755. Bridge identity: '+audit['bridge']['identity_sha256']+'. Existing v6 identity: '+audit['v6']['identity_sha256']+'. Detailed physical hashes and machine-readable results are in verification.json, summary.json, and input_receipt_hashes.json.',ParagraphStyle('Evidence',parent=SMALL,alignment=0))
    add('Mac packaging metadata entries are hash-checked against the transmitted source archive and excluded from scientific-source counts. They are not executable model modules. Original experiment chronology, archived source, raw records, and the original group PDF are preserved; no experiment is backdated.')
    add('Completion checklist',SUB)
    for text in ('Restored Tables 1-6 and Figures 1-6 in the original report\'s visual style.',
                 'Restored learner-choice comparison with correctly labelled MC sampling spread.',
                 'Corrected structural g versus outcome ell, IV estimator, SE, and histogram definitions.',
                 'Separated the two baseline DGPs according to archived code; disclosed controlled learner priors.',
                 'Completed the n=500/R=300 server-only bridge; reused n=300/R=200 raw evidence.',
                 'Retained shared-bank controls, Series OLS / tree mechanisms, sensitivity extensions, and negative findings.'):
        add('- '+text)
    def footer(canvas,doc):
        canvas.setFont('Times-Roman',10);canvas.drawCentredString(A4[0]/2,30,str(doc.page))
    doc=SimpleDocTemplate(str(out/'Simulation_Completed.pdf'),pagesize=A4,leftMargin=54,rightMargin=54,topMargin=48,bottomMargin=48,title='Double Machine Learning - Repaired Simulation Chapter',author='Hongyi Ding - simulation contribution')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    canonical=REPO/'output/pdf/Simulation_Completed.pdf'
    canonical.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(out/'Simulation_Completed.pdf',canonical)


def make_html(out,tables,figures,v6,nu,extras):
    parts=['<!doctype html><html><head><meta charset="utf-8"><title>Completed DML Simulation</title><style>body{font-family:Georgia,serif;max-width:1120px;margin:35px auto;padding:0 20px;color:#222}h1,h2{font-weight:600}img{max-width:100%;height:auto}table{border-collapse:collapse;width:100%;margin:16px 0 28px;font-variant-numeric:tabular-nums}th,td{padding:9px 7px;text-align:center}thead{border-top:2px solid;border-bottom:1px solid}tbody{border-bottom:2px solid}section{margin-bottom:48px}summary{cursor:pointer;padding:12px;background:#f4f4f4}.note{background:#f7f7f7;padding:16px;line-height:1.6}a{color:#075b9c}</style></head><body>']
    parts+=['<h1>Completed DML Simulation / Simulation 补齐结果</h1><p><a href="Simulation_Completed.pdf">完整修正章节 PDF</a> · <a href="summary.json">完整基础数值</a> · <a href="verification.json">核验记录</a></p>',
            '<div class="note">原报告样式的表1-6、图1-6均在下方。n=500/R=300 是服务器固定补跑；n=300/R=200 使用已有原始记录。未在本地拟合模型。Tables 3-4 / Figures 3-4 的误差条为均值 ±1.96 MC SD（抽样散布范围），不是均值置信区间。全部结果保留，包括残余偏差、误差增加和覆盖率不足。</div>']
    parts.append('<section><h2>修正后的公式与 DGP</h2><p>只使用结构 IV-type score。结构函数 g 与 outcome regression ell 分开定义；所有 pilot、scaler 和 nuisance learner 都只用训练折。处理模型使用已知候选列 [0,2,4,6] 及正向/单调先验，不能据此声称任意模型错设下均无偏。</p>')
    for name in ('dgp_model','targets','ell_target','iv_score','naive_score','se'):
        parts.append(f'<img src="equations/{name}.png" alt="{name}">')
    parts.append('<table><thead><tr><th>Experiment / DGP</th><th>m0(X)</th><th>g0(X)</th></tr></thead><tbody>')
    for row in [('Orthogonalization / linear','0.9 X1 + 0.20 X3','0.25 X1 + 0.25 X3'),('Orthogonalization / nonlinear','0.9 X1 + 0.20 s(X3)','s(X1) + 0.25 X3'),('Cross-fitting / linear','1.2 X1 + 0.35 X3','0.40 X1 + 0.40 X3'),('Cross-fitting / nonlinear','1.2 X1 + 0.35 s(X3)','1.2 s(X1) + 0.40 X3')]:
        parts.append('<tr>'+''.join('<td>'+html.escape(x)+'</td>' for x in row)+'</tr>')
    parts.append('</tbody></table><p>X1/X3 对应 Python 列 0/2；s 为 sigmoid。两组实验的系数分别遵循旧代码，不再把旧 Figure 2 的设定写成全部 simulation 的共同 DGP。Moderate 扩展的精确增加项：</p>')
    for name in ('moderate_linear','moderate_nonlinear_m','moderate_nonlinear_g'):
        parts.append(f'<img src="equations/{name}.png" alt="{name}">')
    parts.append('<p>n=500 修复版 DML bias 的绝对值均不超过 0.00446；n=300 cross-fit 仍保留 RF/XGB 的部分残余偏差。线性 Lasso 的 bias 降低不代表 MAE/RMSE 也降低。完整结果均列在下方。</p></section>')
    for i,spec in tables.items():
        parts.append(f'<section><h2>{html.escape(spec["caption"])}</h2><table><thead>')
        if spec['grouped']:
            parts.append('<tr><th rowspan="2">Learner</th><th colspan="3">'+html.escape(spec['rows'][0][1])+'</th><th colspan="3">IV-type DML</th></tr>')
            parts.append('<tr>'+''.join('<th>'+html.escape(x)+'</th>' for x in spec['rows'][1][1:])+'</tr>')
        else:
            parts.append('<tr>'+''.join('<th>'+html.escape(x)+'</th>' for x in spec['rows'][0])+'</tr>')
        parts.append('</thead><tbody>')
        for row in spec['rows'][2 if spec['grouped'] else 1:]:parts.append('<tr>'+''.join('<td>'+html.escape(x)+'</td>' for x in row)+'</tr>')
        parts.append(f'</tbody></table><a href="tables/table_{i}.png">高分辨率表格图片</a></section>')
    for i,path in figures.items():parts.append(f'<section><h2>Figure {i}</h2><img loading="lazy" src="{path}" alt="Figure {i}"></section>')
    parts.append('<h2>全部扩展图：共享模型控制、受控 nuisance 与敏感性分析</h2>')
    for name in extras:parts.append(f'<details><summary>{html.escape(name)}</summary><img loading="lazy" src="extensions/{name}"></details>')
    parts.append('<h2>全部600个敏感性单元</h2><p>保留所有原始统计值；不是完整超参数交叉网格，包含 no-split 的折数别名和 RF weak-outcome 别名，并非600个独立配置。<a href="sensitivity_summary.json">原始完整 JSON 与配对检验</a></p><details><summary>展开完整结果表</summary><table><thead><tr>'+''.join('<th>'+x+'</th>' for x in ('Level','Design','DGP','n','K','Preset','Learner','Method','Bias','MC SD','Mean SE','MAE','RMSE','Coverage'))+'</tr></thead><tbody>')
    for r in v6['summaries']:
        vals=[r[k] for k in ('level','design','dgp','n','k_folds','preset','learner','method')]+[f'{r["bias"]["mean"]:+.6f}']+[f'{r[k]:.6f}' for k in ('mc_sd','mean_se','mae','rmse')]+[f'{100*r["coverage"]["rate"]:.1f}%']
        parts.append('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in vals)+'</tr>')
    parts.append('</tbody></table></details><h2>全部106个 nuisance 比较单元（含 ordinary 负结果）</h2><p><a href="nuisance_summary.json">完整结果 JSON</a></p><details><summary>展开 nuisance 结果</summary><table><thead><tr>'+''.join('<th>'+x+'</th>' for x in ('Design','DGP','Track','Learner','Method','R','Bias','MC SD','Mean SE','Coverage'))+'</tr></thead><tbody>')
    for r in nu:
        vals=[r[k] for k in ('design','dgp','track','learner','method','R')]+[f'{r["bias"]["mean"]:+.6f}',f'{r["mc_sd"]:.6f}',f'{r["mean_se"]:.6f}',f'{100*r["coverage"]["rate"]:.1f}%']
        parts.append('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in vals)+'</tr>')
    parts.append('</tbody></table></details></body></html>');(out/'index.html').write_text('\n'.join(parts))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--bridge-root',type=Path,required=True);parser.add_argument('--v6-root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    for name in ('figures','tables','extensions'):out.joinpath(name).mkdir(exist_ok=True)
    bridge_root=args.bridge_root;archive=bridge_root.parent/'completion_results_v7.tgz'
    bridge,bridge_audit=load_bridge(bridge_root,archive);old,v6_audit=load_v6(args.v6_root)
    groups,summaries=summarize(bridge+old)
    v6_path=REPO/'results/sensitivity/summary.json';v6=json.loads(v6_path.read_text());assert len(v6['summaries'])==600
    assert_v6_summary_match([r for r in summaries if r['design']=='figure2'],v6)
    nu_path=REPO/'results/nuisance_comparison/summary.json';nu=json.loads(nu_path.read_text());assert len(nu)==106
    plt.style.use('classic');plt.rcParams.update({'font.family':'DejaVu Serif','mathtext.fontset':'cm','savefig.facecolor':'white','figure.facecolor':'white'})
    figures={}
    for i,dgp in ((1,'linear'),(2,'nonlinear')):
        path=f'figures/figure_{i}_orthogonalization_{dgp}.png';histogram(groups,'figure1',dgp,out/path,False);figures[i]=path
    for i,dgp in ((3,'linear'),(4,'nonlinear')):
        path=f'figures/figure_{i}_learner_sensitivity_{dgp}.png';learner_plot(summaries,dgp,out/path);figures[i]=path
    for i,dgp in ((5,'linear'),(6,'nonlinear')):
        path=f'figures/figure_{i}_crossfitting_{dgp}.png';histogram(groups,'figure2',dgp,out/path,True);figures[i]=path
    tables={}
    for i,design,dgp,baseline in ((1,'figure1','linear','naive_ml'),(2,'figure1','nonlinear','naive_ml'),(5,'figure2','linear','full_no_split'),(6,'figure2','nonlinear','full_no_split'),(7,'figure2','linear','equal_train_reuse'),(8,'figure2','nonlinear','equal_train_reuse')):
        n,R=(500,300) if design=='figure1' else (300,200)
        title='Non-Orthogonal vs. IV-Type Estimation' if design=='figure1' else 'Full-Sample vs. Cross-Fitting' if baseline=='full_no_split' else 'Equal-Training Reuse vs. Cross-Fitting'
        tables[i]={'caption':f'Table {i}: {title} ({dgp.capitalize()} DGP, n={n}, R={R})','rows':comparison_table(summaries,design,dgp,baseline),'grouped':True}
    for i,dgp in ((3,'linear'),(4,'nonlinear')):tables[i]={'caption':f'Table {i}: Learner Sensitivity ({dgp.capitalize()} DGP, n=500, R=300)','rows':sensitivity_table(summaries,dgp),'grouped':False}
    tables=dict(sorted(tables.items()))
    for i,spec in tables.items():table_image(spec['rows'],spec['caption'],out/'tables'/f'table_{i}.png',spec['grouped'])
    extras=[]
    presentation=args.v6_root/'presentation'
    for path in sorted(presentation.glob('*.png')):
        shutil.copy2(path,out/'extensions'/path.name);extras.append(path.name)
    for path in sorted((args.v6_root/'output/formal/analysis').glob('*hist.png')):
        shutil.copy2(path,out/'extensions'/path.name);extras.append(path.name)
    for path in sorted((REPO/'results/nuisance_comparison').glob('*.png')):
        shutil.copy2(path,out/'extensions'/path.name);extras.append(path.name)
    assert len(extras)==32,(len(extras),extras)
    save_json(out/'summary.json',{'theta0':THETA,'primary_results':summaries,'tables':tables,'figures':figures,'extension_figures':extras,'no_universal_unbiasedness_claim':True})
    save_json(out/'input_receipt_hashes.json',{'bridge':json.loads((bridge_root/'output/legacy_bridge_v7/RESULT_LEDGER.json').read_text()),'v6_reused':v6_audit.pop('input_receipt_hashes')})
    shutil.copy2(v6_path,out/'sensitivity_summary.json');shutil.copy2(nu_path,out/'nuisance_summary.json')
    audit={'status':'PASS','local_model_fits':0,'local_ML_imports':0,'bridge':bridge_audit,'v6':v6_audit,
           'primary_summary_cells':len(summaries),'all_raw_estimates_retained':True,'all_bias_coverage_decomposition_and_pairing_checks_passed':True,
           'old_n300_statistics_match_saved_summary_to_1e_12':True,'original_group_pdf_modified':False,'github_modified':False,
           'sensitivity_summary_sha256':digest(v6_path),'nuisance_summary_sha256':digest(nu_path)}
    make_pdf(out,tables,figures,summaries,v6,nu,audit,extras);make_html(out,tables,figures,v6,nu,extras)
    assert not any(x.split('.')[0] in ('sklearn','xgboost','doubleml') for x in sys.modules),'Renderer must not import learner libraries'
    audit['generated_files']={str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file() and p.name!='verification.json'}
    save_json(out/'verification.json',audit)
    print(json.dumps({'status':'BUILT_AND_NUMERICALLY_VERIFIED','out':str(out),'primary_figures':6,'tables':8,'extension_figures':len(extras),'local_model_fits':0}))


if __name__=='__main__':main()
