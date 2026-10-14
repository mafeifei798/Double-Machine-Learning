#!/usr/bin/env python3
"""Independently recount the saved CSV and hashes. Stdlib only; never fits ML."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1]/'results/cv_evidence')
    args = parser.parse_args()
    root = args.root.resolve()
    summary = json.loads((root/'summary.json').read_text())
    audit = json.loads((root/'verification.json').read_text())
    repo = root.parent.parent
    assert sha(repo/'output/pdf/DML_CV_Evidence.pdf') == audit['pdf_sha256']
    assert sha(repo/'scripts/build_cv_simulation_evidence.py') == audit['builder_sha256']
    for name, digest in audit['generated_files'].items():
        assert sha(root/name) == digest, name
    assert sha(root/'sensitivity_summary.json') == audit['source_summary_sha256']
    assert sha(root/'nuisance_summary.json') == audit['source_nuisance_summary_sha256']
    assert len(json.loads((root/'input_receipt_hashes.json').read_text())) == 800
    groups = defaultdict(list)
    with (root/'primary_draws.csv').open(newline='') as stream:
        for row in csv.DictReader(stream):
            assert int(row['n']) == 1000 and int(row['k_folds']) == 2
            key = tuple(row[k] for k in ('design','dgp','learner','method'))
            theta, se = float(row['theta']), float(row['se'])
            error = theta-.5
            assert se > 0 and abs(error-float(row['bias'])) < 1e-12
            assert (row['covers'] == 'True') == (abs(error) <= 1.96*se)
            groups[key].append(row)
    assert len(groups) == len(summary['primary_results']) == 30
    for cell in summary['primary_results']:
        key = tuple(cell[k] for k in ('design','dgp','learner','method'))
        rows = sorted(groups[key], key=lambda r:int(r['replication']))
        assert len(rows) == 200 and [int(r['replication']) for r in rows] == list(range(200))
        errors = [float(r['theta'])-.5 for r in rows]
        values = dict(mean_theta=statistics.mean(float(r['theta']) for r in rows),
            bias=statistics.mean(errors), mc_sd=statistics.stdev(errors),
            mean_se=statistics.mean(float(r['se']) for r in rows),
            mae=statistics.mean(abs(e) for e in errors),
            rmse=math.sqrt(statistics.mean(e*e for e in errors)),
            coverage=sum(r['covers'] == 'True' for r in rows)/len(rows))
        for name, value in values.items():
            assert abs(value-cell[name]) < 1e-12, (key, name)
        if key[-1] == 'dml_crossfit':
            other = sorted(groups[(*key[:-1], 'naive_ml' if key[0]=='figure1' else 'equal_train_reuse')],
                           key=lambda r:int(r['replication']))
            assert all(a['bank_identity'] == b['bank_identity'] and a['fold_hash'] == b['fold_hash']
                       for a, b in zip(rows, other))
    reductions = []
    for metric in summary['headline_metrics']:
        cells = summary['primary_results']
        before = [r for r in cells if r['design']==metric['design'] and r['method']==metric['comparator']]
        after = [r for r in cells if r['design']==metric['design'] and r['method']=='dml_crossfit']
        assert len(before) == len(after) == 6
        a = statistics.mean(abs(r['bias']) for r in before)
        b = statistics.mean(abs(r['bias']) for r in after)
        assert abs(a-metric['before']) < 1e-12 and abs(b-metric['after']) < 1e-12
        reductions.append(round(100*(1-b/a),1))
    assert reductions == [89.4,97.5]
    print(json.dumps({'status':'PASS_INDEPENDENT_STDLIB_CSV_RECOUNT',
        'saved_estimates':sum(len(v) for v in groups.values()),'summary_cells':len(groups),
        'headline_percent':reductions,'all_asset_hashes_match':True,'model_fits':0,
        'scope':'CSV statistics and cached hashes; full original job receipts are not bundled.'}, indent=2))


if __name__ == '__main__':
    main()
