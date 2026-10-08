#!/usr/bin/env python3
"""Presentation-only derivative of verified v6 summary; never fits a model."""
import hashlib
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
if platform.system() != 'Linux' or str(ROOT) != '/data2/lzx/dhy/dml_sensitivity_v6_20261002':
    raise RuntimeError('render on authorized server; no scientific rerun')
sys.path.insert(0, str(ROOT / 'src'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from analyze_sensitivity_v6 import plot_grid


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = ROOT / 'output/formal/analysis/SUMMARY.json'
    data = json.loads(source.read_text())
    manifest = json.loads((ROOT / 'output/formal/FROZEN_MANIFEST.json').read_text())
    assert data['identity_sha256'] == manifest['identity_sha256']
    assert json.loads((ROOT / 'SERVER_EXIT.json').read_text())['returncode'] == 0
    out = ROOT / 'presentation'
    out.mkdir(exist_ok=False)
    original_subplots, original_save = plt.subplots, Figure.savefig
    checked = []
    current = {}

    def subplots(*args, **kwargs):
        kwargs['layout'] = None
        kwargs['figsize'] = (14, 8.5)
        return original_subplots(*args, **kwargs)

    def save(fig, target, *args, **kwargs):
        level, design, axis, metric = (current[k] for k in ('level', 'design', 'axis', 'metric'))
        # Independently check every plotted point against the frozen summary.
        for i, dgp in enumerate(('linear', 'nonlinear')):
            for j, learner in enumerate(manifest['identity']['protocol']['learners']):
                ax = fig.axes[3*i+j]
                for line in ax.lines:
                    label = line.get_label()
                    if label.startswith('_'):
                        continue
                    if axis == 'sample_fold':
                        method, suffix = label.split(', K=')
                        k = int(suffix)
                        selected = [next(r for r in data['summaries'] if
                            (r['level'], r['design'], r['dgp'], r['learner'], r['method'], r['preset'], r['n'], r['k_folds']) ==
                            (level, design, dgp, learner, method, 'base', n, k)) for n in (300, 1000, 3000)]
                    else:
                        selected = [next(r for r in data['summaries'] if
                            (r['level'], r['design'], r['dgp'], r['learner'], r['method'], r['preset'], r['n'], r['k_folds']) ==
                            (level, design, dgp, learner, label, preset, 1000, 2)) for preset in
                            ('base', 'outcome_weak', 'outcome_strong', 'treatment_weak', 'treatment_strong')]
                    expected = [r['bias']['mean'] if metric == 'bias' else r['coverage']['rate'] if metric == 'coverage' else r['rmse'] for r in selected]
                    assert list(line.get_ydata()) == expected
                    checked.append(len(expected))
        fig.subplots_adjust(left=.07, right=.985, bottom=.12, top=.88, hspace=.46, wspace=.30)
        text = f'{level} | {design} | {axis} | {metric.upper()} | R=200 per cell'
        if axis == 'parameters':
            text += ' | n=1000, K=2'
        fig._suptitle.set_text(text + '\nControlled treatment inputs; individual MC/Wilson intervals where shown')
        fig._suptitle.set_position((.5, .98))
        fig._suptitle.set_fontsize(12)
        kwargs['bbox_inches'] = 'tight'
        return original_save(fig, target, *args, **kwargs)

    plt.subplots, Figure.savefig = subplots, save
    files = []
    protocol = manifest['identity']['protocol']
    for level in protocol['levels']:
        for design in ('figure1', 'figure2'):
            for axis in ('sample_fold', 'parameters'):
                for metric in ('bias', 'rmse', 'coverage'):
                    current.update(level=level, design=design, axis=axis, metric=metric)
                    files.append(plot_grid(data['summaries'], protocol, out, level, design, axis, metric))
    ledger = dict(status='PASS', purpose='layout-only derivative; original sources, estimates and figures unchanged',
        identity_sha256=data['identity_sha256'], summary_sha256=sha(source), renderer_sha256=sha(Path(__file__)),
        figures={name: sha(out/name) for name in files}, plotted_series_checked=len(checked), plotted_points_checked=sum(checked))
    with (out / 'PRESENTATION_LEDGER.json').open('x') as stream:
        json.dump(ledger, stream, indent=2)
    print(json.dumps({k:v for k,v in ledger.items() if k != 'figures'}))


if __name__ == '__main__':
    main()
