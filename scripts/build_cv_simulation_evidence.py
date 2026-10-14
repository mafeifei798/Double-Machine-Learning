#!/usr/bin/env python3
"""Render CV-matched DML evidence from saved server records. Never fit learners."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import shutil
import statistics
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

from build_legacy_simulation_report import (
    comparison_table, sensitivity_table, table_image, pdf_table,
    scaled_image, summarize, normal_pdf, NAMES, ORDER, THETA,
    BODY, TITLE, HEAD, CAPTION, SMALL,
)

REPO = Path(__file__).resolve().parents[1]
ANCHOR = dict(level='moderate', n=1000, k_folds=2, preset='base')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def load_saved(root, complete):
    formal = root / 'output/formal'
    frozen = json.loads((formal / 'FROZEN_MANIFEST.json').read_text())
    identity = frozen['identity_sha256']
    assert hashlib.sha256(json.dumps(frozen['identity'], sort_keys=True).encode()).hexdigest() == identity
    for name, digest in frozen['identity']['source_hashes'].items():
        assert sha(root / name) == digest, name
    assert sha(formal / 'analysis/SUMMARY.json') == sha(REPO / 'results/sensitivity/summary.json')
    ledger = json.loads((formal / 'RESULT_LEDGER.json').read_text())
    rows, inputs = [], {}
    for design in ('figure1', 'figure2'):
        for dgp in ('linear', 'nonlinear'):
            for rep in range(200):
                name = f'{design}_{dgp}_n1000_k2_base_rep{rep:04d}.json'
                path = formal / name
                assert sha(path) == ledger[name], name
                inputs[name] = ledger[name]
                receipt = json.loads(path.read_text())
                assert receipt['identity_sha256'] == identity
                selected = [r for r in receipt['rows'] if r['level'] == 'moderate']
                assert len(selected) == (6 if design == 'figure1' else 9)
                for r in selected:
                    assert all(r[k] == v for k, v in ANCHOR.items())
                    assert (r['design'], r['dgp'], r['replication'], r['p'], r['theta0']) == (design, dgp, rep, 20, THETA)
                    assert r['finite'] and r['se'] > 0 and not r['artificial_pressure']
                    error = r['theta'] - THETA
                    assert abs(error-r['bias']) < 1e-12
                    assert abs(abs(error)-r['abs_error']) < 1e-12
                    assert abs(error*error-r['squared_error']) < 1e-12
                    assert r['covers'] == (abs(error) <= 1.96*r['se'])
                    assert r['train_n_per_model'] == (1000 if r['method'] == 'full_no_split' else 500)
                    assert r['prediction_overlap_n'] == (1000 if r['method'] in ('full_no_split', 'equal_train_reuse') else 0)
                    if r['method'] != 'naive_ml':
                        assert abs(error-sum(r[k] for k in ('weighted_oracle', 'weighted_first_order', 'weighted_second_order'))) < 1e-10
                for learner in ORDER:
                    pair = {r['method']: r for r in selected if r['learner'] == learner}
                    a = pair['naive_ml' if design == 'figure1' else 'equal_train_reuse']
                    b = pair['dml_crossfit']
                    assert a['bank_identity'] == b['bank_identity'] and a['fold_hash'] == b['fold_hash']
                    assert a['x_hash'] == b['x_hash'] and a['u_hash'] == b['u_hash'] and a['v_hash'] == b['v_hash']
                    if design == 'figure1':
                        assert a['g_prediction_hash'] == b['g_prediction_hash'] and a['m_prediction_hash'] == b['m_prediction_hash']
                rows.extend(selected)
    assert len(inputs) == 800 and len(rows) == 6000
    groups, summaries = summarize(rows)
    assert len(summaries) == 30
    for s in summaries:
        matched = [r for r in complete['summaries'] if all(r[k] == s[k] for k in ('design', 'dgp', 'learner', 'method'))
                   and all(r[k] == v for k, v in ANCHOR.items())]
        assert len(matched) == 1 and matched[0]['R'] == s['R'] == 200
        old = matched[0]
        for k in ('mean_theta', 'mc_sd', 'mean_se', 'mae', 'rmse'):
            assert abs(s[k]-old[k]) < 1e-12, (k, s)
        assert abs(s['bias']-old['bias']['mean']) < 1e-12 and s['coverage'] == old['coverage']['rate']
        # A separate standard-library calculation checks the rendering summary.
        g = groups[(s['design'], s['dgp'], s['learner'], s['method'])]
        errors = [r['theta']-THETA for r in g]
        for k, value in dict(bias=statistics.mean(errors), mc_sd=statistics.stdev(errors),
                             mae=statistics.mean(abs(e) for e in errors),
                             rmse=math.sqrt(statistics.mean(e*e for e in errors))).items():
            assert abs(s[k]-value) < 1e-12
    return rows, groups, summaries, {'identity_sha256': identity, 'saved_jobs_read': len(inputs),
        'selected_moderate_estimates': len(rows), 'original_source_hashes_verified': len(frozen['identity']['source_hashes']),
        'source_summary_sha256': sha(formal / 'analysis/SUMMARY.json'), 'input_receipt_hashes': inputs}


def histogram(groups, design, dgp, path, standardized):
    baseline = 'naive_ml' if design == 'figure1' else 'full_no_split'
    methods = (baseline, 'dml_crossfit')
    labels = ('Non-Orthogonal ML', 'IV-Type DML') if design == 'figure1' else ('Full-Sample No Split', '2-Fold Cross-Fitting')
    fig, axes = plt.subplots(3, 2, figsize=(11.2, 11.4), dpi=180)
    for i, learner in enumerate(ORDER):
        values = {m: np.array([(r['theta']-THETA)/(r['se'] if standardized else 1)
                               for r in groups[(design, dgp, learner, m)]]) for m in methods}
        bound = max(4.5 if standardized else .16, *(float(abs(v).max())*1.06 for v in values.values()))
        # Keep the N(0,1) peak visible even when no-split has extreme t-errors.
        grid = np.unique(np.concatenate((np.linspace(-bound, bound, 1800),
                  np.linspace(-min(bound, 6), min(bound, 6), 1800))))
        sigma = 1 if standardized else float(values['dml_crossfit'].std(ddof=1))
        reference = normal_pdf(grid, sigma)
        ymax = max(float(reference.max()), *(float(np.histogram(v, bins=30, density=True)[0].max()) for v in values.values()))*1.18
        for j, method in enumerate(methods):
            ax = axes[i, j]
            ax.hist(values[method], bins=30, density=True, color='#8ecae6', edgecolor='#457b9d', alpha=.9, label='Simulation')
            ax.plot(grid, reference, 'r-', lw=1.2, label='N(0, 1)' if standardized else 'Centered normal, DML MC variance')
            ax.axvline(0, color='black', ls=':', lw=1)
            ax.set(xlim=(-bound, bound), ylim=(0, ymax),
                   xlabel=r'$(\hat{\theta}-\theta_0)/\widehat{SE}$' if standardized else r'$\hat{\theta}-\theta_0$')
            if j == 0:
                ax.set_ylabel('Density')
            ax.set_title(NAMES[learner]+' - '+labels[j], fontsize=11)
            bias = statistics.mean(r['bias'] for r in groups[(design, dgp, learner, method)])
            ax.text(.98, .94, f'Bias = {bias:+.4f}', transform=ax.transAxes, ha='right', va='top', fontsize=9)
            ax.grid(alpha=.22)
            ax.legend(loc='upper left', fontsize=7)
            assert np.all(abs(values[method]) < bound)
    fig.suptitle(f'{dgp.capitalize()} Moderate DGP: {labels[0]} vs. {labels[1]}\n'
                 'n=1,000, p=20, R=200, K=2; '+('studentized errors' if standardized else 'raw estimation errors'), fontsize=13, y=.995)
    fig.tight_layout(rect=[0, 0, 1, .96])
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def learner_plot(summaries, dgp, path):
    g = [next(r for r in summaries if (r['design'], r['dgp'], r['learner'], r['method']) ==
              ('figure1', dgp, learner, 'dml_crossfit')) for learner in ORDER]
    means = [r['mean_theta'] for r in g]
    spread = [1.96*r['mc_sd'] for r in g]
    fig, ax = plt.subplots(figsize=(11.2, 4), dpi=180)
    ax.barh(range(3), means, height=.55, color=['#6a9bcc', '#2a9d8f', '#d97757'], label='Mean estimate (MC)')
    ax.errorbar(means, range(3), xerr=spread, fmt='none', ecolor='black', capsize=4, label=r'$\pm 1.96\times$ MC SD')
    ax.axvline(THETA, color='black', ls='--', lw=1.1, label=r'True $\theta_0=0.5$')
    ax.set(yticks=range(3), yticklabels=[NAMES[l] for l in ORDER], xlabel=r'$\hat{\theta}$',
           xlim=(0, .7), title=f'Learner Sensitivity ({dgp.capitalize()} Moderate DGP, n=1,000, R=200)')
    ax.grid(axis='x', alpha=.22)
    ax.legend(loc='center left', bbox_to_anchor=(1.01, .5), fontsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def headline_metrics(summaries):
    metrics = []
    for design, baseline in (('figure1', 'naive_ml'), ('figure2', 'full_no_split')):
        a = [r for r in summaries if r['design'] == design and r['method'] == baseline]
        b = [r for r in summaries if r['design'] == design and r['method'] == 'dml_crossfit']
        assert len(a) == len(b) == 6
        before = statistics.mean(abs(r['bias']) for r in a)
        after = statistics.mean(abs(r['bias']) for r in b)
        metrics.append(dict(design=design, comparator=baseline, before=before, after=after,
                            bias_reduction_percent=100*(1-after/before), cells=6))
    assert [round(m['bias_reduction_percent'], 1) for m in metrics] == [89.4, 97.5]
    return metrics


def headline_plot(metrics, path):
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), dpi=200)
    for ax, metric, label in zip(axes, metrics, ('Non-Orthogonal ML', 'Full-Sample No Split')):
        values = [metric['before'], metric['after']]
        ax.barh([1, 0], values, color=['#8ecae6', '#2a9d8f'], edgecolor='#457b9d', height=.48)
        ax.set(yticks=[1, 0], yticklabels=[label, 'IV-Type DML / Cross-Fit'],
               xlim=(0, max(values)*1.42), xlabel='Average absolute MC mean bias')
        for y, value in zip([1, 0], values):
            ax.text(value+max(values)*.025, y, f'{value:.6f}', va='center', fontsize=10)
        ax.set_title(f"{metric['bias_reduction_percent']:.1f}% lower bias", fontsize=15, pad=13)
        ax.grid(axis='x', alpha=.22)
    fig.suptitle('CV Headline Evidence: Three Learners x Two DGPs\nModerate anchor: n=1,000, p=20, K=2, R=200', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, .85], w_pad=2)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def headline_table(metrics, path):
    rows = [['Comparison', 'Before', 'After', 'Bias reduction']]
    for m, label in zip(metrics, ('Non-Orthogonal ML', 'Full-Sample No Split')):
        rows.append([label, f"{m['before']:.7f}", f"{m['after']:.7f}", f"{m['bias_reduction_percent']:.1f}%"])
    with plt.style.context('classic'), plt.rc_context({'font.family':'DejaVu Serif', 'mathtext.fontset':'cm'}):
        table_image(rows, 'Table 0: CV Headline Comparison (Moderate DGP, n=1,000, K=2, R=200)', path)


def controlled_rows(nuisance):
    selected = [r for r in nuisance if r['track'] == 'controlled' and r['design'] == 'figure2'
                and r['learner'].startswith('tree_leaf_') and r['method'] == 'crossfit']
    assert len(selected) == 8
    rows = [['DGP', 'Tree min. leaf', 'Mean estimate', 'Bias', 'Coverage']]
    for r in sorted(selected, key=lambda x: (x['dgp'], int(x['learner'].split('_')[-1]))):
        rows.append([r['dgp'].capitalize(), r['learner'].split('_')[-1], f"{r['mean_theta']:.4f}",
                     f"{r['bias']['mean']:+.4f}", f"{100*r['coverage']['rate']:.1f}%"])
    assert max(abs(r['bias']['mean']) for r in selected) < .0046
    return rows


def grid_rows(complete):
    rows = [['Design', 'Learner', 'Bias range', 'Coverage range']]
    for design in ('figure1', 'figure2'):
        for learner in ORDER:
            selected = [r for r in complete['summaries'] if r['design'] == design and r['learner'] == learner
                        and r['method'] == 'dml_crossfit']
            b = [r['bias']['mean'] for r in selected]
            c = [r['coverage']['rate'] for r in selected]
            rows.append(['Orthogonalization' if design == 'figure1' else 'Cross-fitting', NAMES[learner],
                         f'[{min(b):+.4f}, {max(b):+.4f}]', f'[{100*min(c):.1f}%, {100*max(c):.1f}%]'])
    return rows


def simple_pdf_table(rows, widths):
    cell = ParagraphStyle('CVCell', parent=SMALL, alignment=TA_CENTER)
    table = Table([[Paragraph(html.escape(str(c)), cell) for c in row] for row in rows], colWidths=widths, repeatRows=1)
    table.setStyle(TableStyle([('LINEABOVE', (0, 0), (-1, 0), .85, colors.black),
        ('LINEBELOW', (0, 0), (-1, 0), .5, colors.black), ('LINEBELOW', (0, -1), (-1, -1), .85, colors.black),
        ('TOPPADDING', (0, 0), (-1, -1), 6), ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')]))
    return table


def report_pdf(out, pdf_path, tables, figures, metrics, controls, grid, audit):
    story = []
    def add(text, style=BODY):
        story.append(Paragraph(text, style))
    def page():
        story.append(PageBreak())
    def tbl(i):
        spec = tables[i]
        add(spec['caption'], CAPTION)
        story.append(pdf_table(spec['rows'], spec['grouped']))
        story.append(Spacer(1, 9))
    add('Double Machine Learning: CV Evidence', TITLE)
    add('A focused companion to the research-CV project description. This document maps each claim to saved Monte Carlo evidence; it is not a replacement for the CV and does not add new model fits.')
    add('1. Main comparison and metric', HEAD)
    add('Three nuisance learner families (Lasso, Random Forest and XGBoost) are evaluated in linear and nonlinear moderate DGPs. Both headline comparisons use n=1,000, p=20, true effect 0.5, K=2, the base learner preset and 200 replications per setting.')
    story.append(scaled_image(out/'figures/headline_bias.png', 480, 215))
    add('Headline metric', CAPTION)
    head = [['Comparison', 'Before', 'After', 'Reduction']]
    for m, label in zip(metrics, ('Non-orthogonal ML vs. DML', 'Full no-split vs. cross-fitting')):
        head.append([label, f"{m['before']:.7f}", f"{m['after']:.7f}", f"{m['bias_reduction_percent']:.1f}%"])
    story.append(simple_pdf_table(head, [215, 80, 80, 95]))
    add('For each learner-DGP setting, bias is the mean of theta_hat minus 0.5 over 200 replications. The headline averages the absolute values of those six signed means. It is NOT MAE, RMSE, the average of six percentage reductions, or a result pooled over the sensitivity grid.', SMALL)
    add('These n=1,000 moderate results are separate from the newly completed n=500 baseline bridge and the reused n=300 baseline report. Raw-error and studentized-error plots below are explicitly labelled.', SMALL)
    for i, dgp in ((1, 'linear'), (2, 'nonlinear')):
        page()
        add('2. Orthogonalization: '+dgp.capitalize()+' DGP', HEAD)
        tbl(i)
        story.append(scaled_image(out/figures[i], 470, 490))
        add(f'Figure {i}. Non-orthogonal ML and IV-type DML use the same out-of-sample structural-nuisance predictions. The estimator score is the changed component. Raw errors are shown; axes match within each learner pair and every draw is retained.', SMALL)
    page()
    add('3. Learner choice and finite-sample spread', HEAD)
    for i, dgp in ((3, 'linear'), (4, 'nonlinear')):
        tbl(i)
        story.append(scaled_image(out/figures[i], 475, 150))
    add('Error bars are mean estimate +/-1.96 Monte Carlo SD: an approximate sampling spread, NOT a confidence interval for the Monte Carlo mean. Learner-dependent residual bias remains; these results do not establish exact finite-sample unbiasedness.', SMALL)
    for i, dgp in ((5, 'linear'), (6, 'nonlinear')):
        page()
        add('4. Cross-fitting: '+dgp.capitalize()+' DGP', HEAD)
        tbl(i)
        story.append(scaled_image(out/figures[i], 470, 490))
        add(f'Figure {i}. Studentized errors (theta_hat - 0.5)/SE_hat. Full no-split trains on all 1,000 observations; each cross-fit nuisance model trains on 500. Same learner presets and data are used, with no comparator-only stress. Large no-split t-errors are not clipped. Raw-error companion plots are in the appendix.', SMALL)
    page()
    add('5. Equal-training-size reuse control', HEAD)
    add('To separate sample reuse from training size, equal-training reuse and cross-fitting share exactly the same fitted nuisance-model bank and train on 500 observations per model. Their predictions differ in whether the evaluated outcome was in the training set. The conventional full-sample comparison is reported separately.')
    tbl(7)
    tbl(8)
    add('The shared-bank control supports the reuse mechanism but is not relabelled as the full-sample no-split comparator. Its numerical reductions are not substituted into the CV headline.', SMALL)
    page()
    add('6. Controlled Series OLS / decision-tree evidence', HEAD)
    add('A separate controlled study fixes the treatment regression to Series OLS with features [1, X1, X3, sigmoid(X1), sigmoid(X3)] and varies decision-tree outcome leaf size (2, 5, 10, 20). Its n=1,000/R=200 Figure 2 outcomes are not pooled with the Lasso/RF/XGBoost headline.')
    add('Table 9: Controlled tree cross-fit settings', CAPTION)
    story.append(simple_pdf_table(controls, [70, 95, 105, 90, 110]))
    add('Across these eight tree-by-DGP settings, maximum absolute MC mean bias is below 0.0046. This descriptive bound is not a guarantee that every mean-bias interval lies inside +/-0.01. The broader controlled and ordinary-learner outcomes remain in the full 106-cell nuisance summary.', SMALL)
    for dgp in ('linear', 'nonlinear'):
        page()
        story.append(scaled_image(out/f'extensions/controlled_figure2_{dgp}.png', 480, 590))
        add('Controlled nuisance study ('+dgp+' DGP). Original saved figure, byte-for-byte unchanged.', CAPTION)
    page()
    add('7. Sensitivity analysis and limitations', HEAD)
    add('The fixed grid varies n=300/1,000/3,000, K=2/5, baseline/moderate DGP level and four one-nuisance-role-at-a-time regularization alternatives. All 600 reported summary cells are retained, with paired MAE/MSE tests, Holm corrections and error decompositions. This is not a full factorial hyperparameter grid; some cells are aliases, not independent runs.')
    add('Table 10: Cross-fit ranges over the complete sensitivity grid', CAPTION)
    story.append(simple_pdf_table(grid, [120, 90, 130, 130]))
    add('Ranges describe heterogeneous settings, not one pooled estimator. Coverage can fall to 87.5%; five folds are not uniformly superior to two; and lower absolute mean bias need not lower MAE or RMSE. The sensitivity study evaluates robustness boundaries instead of proving unconditional robustness.', SMALL)
    for level in ('baseline', 'moderate'):
        for topic in ('sample_fold', 'parameters'):
            for metric in ('bias', 'rmse', 'coverage'):
                page()
                add(level.capitalize()+' sensitivity: '+topic.replace('_',' ')+' / '+metric.upper(), HEAD)
                add('Each image contains BOTH DGPs: linear in the upper row and nonlinear in the lower row. These are separate validation settings, not additional observations pooled into the CV headline.', SMALL)
                for design in ('figure1', 'figure2'):
                    name = f'{level}_{design}_{topic}_{metric}.png'
                    story.append(scaled_image(out/'extensions'/name, 480, 275))
                    add(name.replace('_', ' ').replace('.png', '')+'. Original saved figure, unchanged.', SMALL)
    for dgp in ('linear', 'nonlinear'):
        page()
        story.append(scaled_image(out/f'extensions/raw_crossfitting_{dgp}.png', 480, 590))
        add('Raw-error companion: full no-split versus cross-fitting ('+dgp+' moderate DGP, n=1,000, R=200). Every draw is shown on common within-learner axes. This plot is not a studentized-error plot.', CAPTION)
    for design, comparator in (('figure1','naive_ml'),('figure2','full_no_split'),('figure2','equal_train_reuse')):
        for dgp in ('linear','nonlinear'):
            page()
            name = f'moderate_{design}_{dgp}_{comparator}_hist.png'
            story.append(scaled_image(out/'extensions'/name, 480, 590))
            add('Archived original histogram: '+design+' / '+dgp+' / '+comparator.replace('_',' ')+'. Studentized errors; n=1,000, K=2, R=200. This original presentation of the SAME saved experiment is retained for completeness, not counted as a new independent experiment.', CAPTION)
    page()
    add('8. Methods, provenance and claim boundaries', HEAD)
    add('Model: Y=theta0*D+g0(X)+U and D=m0(X)+V, theta0=0.5. X is Gaussian with p=20 and covariance 0.7 to the power |j-k|; U and V are independent standard Gaussian disturbances. Structural g0=E[Y-theta0*D|X] differs from ell0=E[Y|X]=g0+theta0*m0. Only the structural IV-type score (D-m_hat)*(Y-theta*D-g_hat) underlies the headline; partialling-out is not used.')
    add('All pilots, scalers and nuisance models use training-fold data only. In the orthogonalization experiment, g is learned from Y-pilot*D; in the reuse experiment, g_hat=ell_hat-pilot*m_hat. Known treatment candidate columns [0,2,4,6] and positive/monotone restrictions are explicit controlled priors, not unrestricted confounder discovery. Exact archived DGP coefficients and moderate additions are retained in the configuration files.')
    add('No local fitting, retuning, seed selection or new server run occurred for this attachment. All 800 selected job receipts and 19 frozen source files were hash-checked. The 6,000 selected estimates were recomputed and matched to the saved 600-cell summary to 1e-12. Pairing, prediction overlap, shared nuisance banks, error decomposition and coverage indicators were checked before plotting.')
    add('The original raw results, earlier completed simulation chapter and GitHub branch were not modified by this build. Source images are copied unchanged; new figures are deterministic visualizations of the saved records. The complete source summaries retain adverse and inconclusive outcomes, including ordinary-learner residual bias. Histograms are descriptive diagnostics, not proofs of asymptotic normality or universal finite-sample unbiasedness.')
    add('Evidence identity: '+audit['identity_sha256'], ParagraphStyle('EvidenceID', parent=SMALL, alignment=0))
    add('Machine-readable outputs: summary.json, primary_draws.csv, input_receipt_hashes.json and verification.json. The offline index includes every new table/figure and the complete sensitivity and nuisance summaries.', SMALL)
    def footer(canvas, doc):
        canvas.setFont('Times-Roman', 10)
        canvas.drawCentredString(A4[0]/2, 28, str(doc.page))
    SimpleDocTemplate(str(pdf_path), pagesize=A4, leftMargin=54, rightMargin=54,
        topMargin=46, bottomMargin=46, title='Double Machine Learning - CV Evidence',
        author='Hongyi Ding - simulation contribution').build(story, onFirstPage=footer, onLaterPages=footer)


def gallery(out, tables, figures, metrics, extensions):
    parts = ['<!doctype html><html><head><meta charset="utf-8"><title>DML CV Evidence</title>',
        '<style>body{font-family:Georgia,serif;max-width:1120px;margin:35px auto;padding:0 20px;color:#222}h1,h2{font-weight:600}img{max-width:100%;height:auto}table{border-collapse:collapse;width:100%;margin:16px 0 28px;font-variant-numeric:tabular-nums}th,td{padding:9px 7px;text-align:center}thead{border-top:2px solid;border-bottom:1px solid}tbody{border-bottom:2px solid}section{margin-bottom:45px}summary{cursor:pointer;padding:12px;background:#f4f4f4}.note{background:#f7f7f7;padding:16px;line-height:1.65}a{color:#075b9c}</style></head><body>',
        '<h1>DML 简历主实验：Linear + Nonlinear 完整版</h1><p><a href="../../output/pdf/DML_CV_Evidence.pdf">完整 PDF</a> · <a href="summary.json">主实验数值</a> · <a href="verification.json">复核记录</a></p>',
        '<div class="note">仅使用已保存的服务器结果，未拟合任何模型。简历两项指标均对应 moderate DGP、n=1,000、K=2、R=200，每项对三个 learner × 两个 DGP 的绝对 Monte Carlo 平均偏差等权平均。不是 MAE，不混入 n=500 / n=300 baseline，也不把全部敏感性实验合成一个去偏百分比。保留残余偏差、覆盖不足及负结果。</div>',
        '<img src="figures/headline_bias.png" alt="89.4% and 97.5% headline bias reductions">',
        '<section><h2>Table 0: 简历主指标</h2><img src="tables/table_0_headline.png" alt="Headline table"></section>']
    parts.append('<p><a href="#linear">Linear DGP</a> · <a href="#nonlinear">Nonlinear DGP</a> · <a href="#sensitivity">全部24张敏感性图</a> · <a href="#archived">原始直方图</a></p>')
    displayed = set()
    def table(i):
        spec = tables.get(i, tables.get(str(i)))
        parts.append('<section><h2>'+html.escape(spec['caption'])+'</h2><table><thead>')
        if spec['grouped']:
            parts.append('<tr><th rowspan="2">Learner</th><th colspan="3">'+html.escape(spec['rows'][0][1])+'</th><th colspan="3">IV-type DML</th></tr>')
            parts.append('<tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in spec['rows'][1][1:])+'</tr>')
        else:
            parts.append('<tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in spec['rows'][0])+'</tr>')
        parts.append('</thead><tbody>')
        for row in spec['rows'][2 if spec['grouped'] else 1:]:
            parts.append('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>')
        parts.append(f'</tbody></table><a href="tables/table_{i}.png">高分辨率表格</a></section>')
    def picture(path, title):
        parts.append(f'<section><h3>{html.escape(title)}</h3><img loading="lazy" src="{path}" alt="{html.escape(title)}"></section>')
        if path.startswith('extensions/'):
            displayed.add(path.split('/',1)[1])
    for dgp, indices in (('linear',(1,3,5,7)), ('nonlinear',(2,4,6,8))):
        parts.append(f'<h2 id="{dgp}">{dgp.capitalize()} DGP：全部主表、主图与对照</h2>')
        for i in indices[:3]:
            table(i)
            picture(figures.get(i, figures.get(str(i))), f'Figure {i}: {dgp.capitalize()} DGP')
        picture(f'extensions/raw_crossfitting_{dgp}.png', 'No-split vs Cross-fitting: raw errors / '+dgp)
        table(indices[3])
        picture(f'extensions/moderate_figure2_{dgp}_equal_train_reuse_hist.png', 'Equal-training shared-bank control / '+dgp)
        picture(f'extensions/controlled_figure2_{dgp}.png', 'Series OLS / Decision Tree control / '+dgp)
    for i, title in ((9, 'Series OLS / Decision Tree 受控实验'), (10, '完整敏感性网格中的 Cross-fit 范围')):
        parts.append(f'<section><h2>Table {i}: {title}</h2><img src="tables/table_{i}.png" alt="Table {i}"></section>')
    parts.append('<p>Tables 3-4 / Figures 3-4 中的误差条为 MC 抽样散布，不是均值置信区间。Figures 1-2 是原始误差，Figures 5-6 是 studentized errors；No-split 极端值未截掉。完整全部600个敏感性单元及106个 nuisance 单元见 <a href="sensitivity_summary.json">sensitivity JSON</a> 和 <a href="nuisance_summary.json">nuisance JSON</a>。</p>')
    parts.append('<h2 id="sensitivity">完整敏感性分析：24张，Linear 与 Nonlinear 均保留</h2><p>每张图的上排是 Linear、下排是 Nonlinear。Baseline / Moderate × 两组机制 × 两类敏感因素 × Bias / RMSE / Coverage 全部展示，不只选 Bias 或 Coverage。</p>')
    for level in ('baseline','moderate'):
        parts.append('<h3>'+level.capitalize()+' sensitivity</h3>')
        for topic in ('sample_fold','parameters'):
            for metric in ('bias','rmse','coverage'):
                for design in ('figure1','figure2'):
                    name = f'{level}_{design}_{topic}_{metric}.png'
                    parts.append(f'<details><summary>{html.escape(name)}</summary><img loading="lazy" src="extensions/{name}" alt="{html.escape(name)}"></details>')
                    displayed.add(name)
    parts.append('<h2 id="archived">其余原始 Studentized 直方图：两种 DGP 全部保留</h2><p>同一份已保存实验的原始展示版本，不是新实验，不重复计入简历指标。</p>')
    for name in extensions:
        if name in displayed:
            continue
        parts.append(f'<details><summary>{html.escape(name)}</summary><img loading="lazy" src="extensions/{name}" alt="{html.escape(name)}"></details>')
        displayed.add(name)
    assert set(extensions) == displayed
    parts.append('</body></html>')
    (out/'index.html').write_text('\n'.join(parts))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--v6-root', type=Path, required=True)
    ap.add_argument('--output', type=Path, default=REPO/'results/cv_evidence')
    ap.add_argument('--pdf-output', type=Path, default=REPO/'output/pdf/DML_CV_Evidence.pdf')
    a = ap.parse_args()
    out = a.output
    pdf_path = a.pdf_output
    assert not pdf_path.exists(), 'Do not overwrite an existing evidence PDF'
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=False)
    for name in ('figures', 'tables', 'extensions'):
        (out/name).mkdir()
    primary_path = REPO/'results/sensitivity/summary.json'
    nuisance_path = REPO/'results/nuisance_comparison/summary.json'
    complete = json.loads(primary_path.read_text())
    nuisance = json.loads(nuisance_path.read_text())
    assert len(complete['summaries']) == 600 and len(nuisance) == 106
    rows, groups, summaries, audit = load_saved(a.v6_root, complete)
    metrics = headline_metrics(summaries)
    plt.style.use('classic')
    plt.rcParams.update({'font.family':'DejaVu Serif', 'mathtext.fontset':'cm',
                         'savefig.facecolor':'white', 'figure.facecolor':'white'})
    figures = {}
    for i, design, dgp, standard in ((1,'figure1','linear',False),(2,'figure1','nonlinear',False),
                                    (5,'figure2','linear',True),(6,'figure2','nonlinear',True)):
        name = f'figures/figure_{i}_{design}_{dgp}.png'
        histogram(groups, design, dgp, out/name, standard)
        figures[i] = name
    for i, dgp in ((3,'linear'),(4,'nonlinear')):
        name = f'figures/figure_{i}_learner_sensitivity_{dgp}.png'
        learner_plot(summaries, dgp, out/name)
        figures[i] = name
    figures = dict(sorted(figures.items()))
    headline_plot(metrics, out/'figures/headline_bias.png')
    headline_table(metrics, out/'tables/table_0_headline.png')
    tables = {}
    for i, design, dgp, baseline in ((1,'figure1','linear','naive_ml'),(2,'figure1','nonlinear','naive_ml'),
                                    (5,'figure2','linear','full_no_split'),(6,'figure2','nonlinear','full_no_split'),
                                    (7,'figure2','linear','equal_train_reuse'),(8,'figure2','nonlinear','equal_train_reuse')):
        label = 'Non-Orthogonal ML vs. DML' if design == 'figure1' else 'Full-Sample No Split vs. Cross-Fitting' if baseline == 'full_no_split' else 'Equal-Training Reuse vs. Cross-Fitting'
        tables[i] = dict(caption=f'Table {i}: {label} ({dgp.capitalize()} Moderate DGP, n=1,000, R=200)',
                         rows=comparison_table(summaries, design, dgp, baseline), grouped=True)
    for i, dgp in ((3,'linear'),(4,'nonlinear')):
        tables[i] = dict(caption=f'Table {i}: Learner Sensitivity ({dgp.capitalize()} Moderate DGP, n=1,000, R=200)',
                         rows=sensitivity_table(summaries, dgp), grouped=False)
    tables = dict(sorted(tables.items()))
    for i, spec in tables.items():
        table_image(spec['rows'], spec['caption'], out/'tables'/f'table_{i}.png', spec['grouped'])
    controls, grid = controlled_rows(nuisance), grid_rows(complete)
    for i, tab, label in ((9, controls, 'Controlled Tree Cross-Fit Settings'),
                          (10, grid, 'Cross-Fit Ranges Over the Full Sensitivity Grid')):
        fig, ax = plt.subplots(figsize=(11.2, 4.0 if i == 9 else 3.6), dpi=180)
        ax.axis('off')
        ax.set_title(f'Table {i}: {label}', fontsize=12, pad=20)
        table = ax.table(cellText=tab, cellLoc='center', loc='center', bbox=[0,0,1,.92])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        for (r,c), cell in table.get_celld().items():
            cell.set_facecolor('white');cell.set_edgecolor('black');cell.set_linewidth(.6);cell.visible_edges=''
            if r == 0: cell.visible_edges='TB'
            if r == len(tab)-1: cell.visible_edges += 'B'
        fig.tight_layout();fig.savefig(out/'tables'/f'table_{i}.png', bbox_inches='tight');plt.close(fig)
    extension_hashes = {}
    names = sorted(p.name for p in (REPO/'results/legacy_report/extensions').glob('*.png'))
    assert len(names) == 32
    for name in names:
        source = REPO/('results/nuisance_comparison' if name.startswith('controlled') else 'results/legacy_report/extensions')/name
        shutil.copy2(source, out/'extensions'/name)
        assert sha(source) == sha(out/'extensions'/name)
        extension_hashes[name] = sha(source)
    for dgp in ('linear','nonlinear'):
        name = f'raw_crossfitting_{dgp}.png'
        histogram(groups, 'figure2', dgp, out/'extensions'/name, False)
        names.append(name)
    save(out/'summary.json', dict(protocol=ANCHOR, R=200, p=20, theta0=THETA,
        metric='equal-weight average of absolute signed Monte Carlo mean biases over six learner-DGP settings',
        headline_metrics=metrics, primary_results=summaries, tables=tables, figures=figures,
        controlled_tree_table=controls, full_grid_crossfit_ranges=grid,
        extensions=names, no_universal_unbiasedness_or_dominance_claim=True))
    inputs = audit.pop('input_receipt_hashes')
    save(out/'input_receipt_hashes.json', inputs)
    fields = ['design','dgp','learner','method','replication','n','k_folds','theta','se','bias','covers','bank_identity','fold_hash']
    with (out/'primary_draws.csv').open('w', newline='') as stream:
        writer=csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for r in rows: writer.writerow({k:r[k] for k in fields})
    shutil.copy2(primary_path, out/'sensitivity_summary.json')
    shutil.copy2(nuisance_path, out/'nuisance_summary.json')
    report_pdf(out, pdf_path, tables, figures, metrics, controls, grid, audit)
    gallery(out, tables, figures, metrics, names)
    assert not any(k.split('.')[0] in ('sklearn','xgboost','doubleml') for k in sys.modules)
    audit.update(status='PASS_SAVED_EVIDENCE_RECOUNT', local_model_fits=0, new_server_runs=0,
                 local_ML_imports=0, github_modified=False, historical_raw_records_modified=False,
                 all_primary_statistics_match_original_summary_to_1e_12=True,
                 all_replications_retained=True, no_histogram_clipping=True,
                 same_bank_and_overlap_checks_pass=True, unmodified_extension_hashes=extension_hashes,
                 source_nuisance_summary_sha256=sha(nuisance_path), pdf_sha256=sha(pdf_path),
                 builder_sha256=sha(Path(__file__)),
                 generated_files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()})
    save(out/'verification.json', audit)
    print(json.dumps(dict(status='BUILT_AND_VERIFIED', pdf=str(pdf_path), gallery=str(out/'index.html'),
                         main_figures=6, overview_figures=1, support_figures=34, table_images=11,
                         headline_percent=[round(m['bias_reduction_percent'],1) for m in metrics],
                         selected_estimates=6000, saved_jobs=800, local_model_fits=0), indent=2))


if __name__ == '__main__':
    main()
