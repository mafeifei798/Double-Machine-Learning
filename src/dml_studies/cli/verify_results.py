#!/usr/bin/env python3
"""Independently recount curated saved results. Stdlib only; no model fitting."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics

REPO = Path(__file__).resolve().parents[3]


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def statistics_for(rows):
    errors = [float(row["theta"]) - .5 for row in rows]
    return {"mean_theta": statistics.mean(float(row["theta"]) for row in rows),
        "bias": statistics.mean(errors), "mc_sd": statistics.stdev(errors),
        "mean_se": statistics.mean(float(row["se"]) for row in rows),
        "mae": statistics.mean(abs(x) for x in errors),
        "rmse": math.sqrt(statistics.mean(x * x for x in errors)),
        "coverage": sum(abs(float(row["theta"]) - .5) <= 1.96 * float(row["se"]) for row in rows) / len(rows)}


def compare(rows, summary, nested=False):
    for key, value in statistics_for(rows).items():
        expected = summary[key]
        if nested and key in ("bias", "coverage"):
            expected = expected["mean" if key == "bias" else "rate"]
        assert abs(value - expected) < 1e-12, (key, value, expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO / "results")
    parser.add_argument("--input-directory", type=Path,
                        help="Optionally compare every exported job with its original archived receipt.")
    args = parser.parse_args()
    root = args.root
    provenance = json.loads((root / "data_manifest.json").read_text())
    assert provenance["record_type"] == "derived_saved_result_export"
    for name, expected in provenance["result_files"].items():
        assert digest((root / name).read_bytes()) == expected, name
    assert len(provenance["primary_source_receipt_hashes"]) == 800
    raw = (root / "simulation_records.jsonl.gz").read_bytes()
    assert raw[4:8] == b"\0" * 4, "Derived gzip has no creation-clock value"
    with gzip.open(root / "simulation_records.jsonl.gz", "rt") as stream:
        jobs = [json.loads(line) for line in stream]
    assert len(jobs) == 1000
    if args.input_directory:
        try:
            from .export_results import source_jobs
        except ImportError:
            from export_results import source_jobs
        original, archives, sources, durations = source_jobs(args.input_directory)
        assert jobs == original
        assert archives == provenance["original_archives"]
        assert sources == provenance["sources"] and durations == provenance["control_durations"]
    rows, seen, counts, finite_sample = [], set(), defaultdict(int), []
    for job in jobs:
        key = (job["source"], job["source_member"])
        assert key not in seen
        seen.add(key)
        receipt = job["receipt"]
        assert digest(canonical(receipt)) == job["canonical_receipt_sha256"]
        assert receipt["identity_sha256"] == provenance["sources"][job["source"]]["original_identity_sha256"]
        assert math.isfinite(receipt["seconds"]) and receipt["seconds"] >= 0
        assert isinstance(receipt["warnings"], list)
        counts[job["source"]] += 1
        assert len(receipt["rows"]) == (6 if job["source"] == "finite_sample" else 18)
        for row in receipt["rows"]:
            assert row["finite"] and row["se"] > 0 and row["p"] == 20 and row["theta0"] == .5
            assert not row["artificial_pressure"]
            error = row["theta"] - .5
            assert abs(error - row["bias"]) < 1e-12
            assert abs(abs(error) - row["abs_error"]) < 1e-12
            assert abs(error * error - row["squared_error"]) < 1e-12
            assert row["covers"] == (abs(error) <= 1.96 * row["se"])
            if row["method"] != "naive_ml":
                assert abs(error - sum(row[k] for k in ("weighted_oracle", "weighted_first_order", "weighted_second_order"))) < 1e-10
            rows.append(row)
            if job["source"] == "finite_sample" or row["level"] == "baseline":
                finite_sample.append(row)
    assert dict(counts) == {"finite_sample": 600, "sensitivity": 400}
    assert len(rows) == 10800 and len(finite_sample) == 7200
    for row in provenance["control_durations"]:
        assert math.isfinite(row["seconds"]) and row["seconds"] >= 0
    groups = defaultdict(list)
    for row in rows:
        groups[(row["level"], row["design"], row["dgp"], row["n"], row["learner"], row["method"])].append(row)
    assert len(groups) == 48
    for key, group in groups.items():
        group.sort(key=lambda row: row["replication"])
        expected_R = 300 if key[3] == 500 else 200
        assert len(group) == expected_R
        assert [row["replication"] for row in group] == list(range(expected_R))
        if key[-1] == "dml_crossfit":
            other = groups[(*key[:-1], "naive_ml" if key[1] == "figure1" else "equal_train_reuse")]
            other = sorted(other, key=lambda row: row["replication"])
            for a, b in zip(group, other):
                assert a["bank_identity"] == b["bank_identity"] and a["fold_hash"] == b["fold_hash"]
                assert a["prediction_overlap_n"] == 0
                assert b["prediction_overlap_n"] == (0 if key[1] == "figure1" else key[3])
                if key[1] == "figure1":
                    assert a["g_prediction_hash"] == b["g_prediction_hash"] and a["m_prediction_hash"] == b["m_prediction_hash"]
    finite_sample_summary = json.loads((root / "finite_sample/summary.json").read_text())
    assert len(finite_sample_summary["primary_results"]) == 30
    for summary in finite_sample_summary["primary_results"]:
        n = 500 if summary["design"] == "figure1" else 300
        key = ("baseline", summary["design"], summary["dgp"], n, summary["learner"], summary["method"])
        compare(groups[key], summary)
    sensitivity = json.loads((root / "sensitivity/summary.json").read_text())
    nuisance = json.loads((root / "nuisance_comparison/summary.json").read_text())
    assert len(sensitivity["summaries"]) == 600 and len(nuisance) == 106
    checked = 0
    for summary in sensitivity["summaries"]:
        if summary["design"] == "figure2" and summary["n"] == 300 and summary["k_folds"] == 2 and summary["preset"] == "base":
            key = (summary["level"], summary["design"], summary["dgp"], 300, summary["learner"], summary["method"])
            compare(groups[key], summary, nested=True)
            checked += 1
    assert checked == 36
    main_summary = json.loads((root / "primary/summary.json").read_text())
    main_groups = defaultdict(list)
    with (root / "primary/estimates.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            assert int(row["n"]) == 1000 and int(row["k_folds"]) == 2
            theta, se = float(row["theta"]), float(row["se"])
            assert se > 0 and abs(theta - .5 - float(row["bias"])) < 1e-12
            assert (row["covers"] == "True") == (abs(theta - .5) <= 1.96 * se)
            main_groups[tuple(row[k] for k in ("design", "dgp", "learner", "method"))].append(row)
    assert len(main_groups) == len(main_summary["primary_results"]) == 30
    assert sum(map(len, main_groups.values())) == 6000
    for summary in main_summary["primary_results"]:
        key = tuple(summary[k] for k in ("design", "dgp", "learner", "method"))
        group = sorted(main_groups[key], key=lambda row: int(row["replication"]))
        assert len(group) == 200 and [int(row["replication"]) for row in group] == list(range(200))
        compare(group, summary)
        if key[-1] == "dml_crossfit":
            other = sorted(main_groups[(*key[:-1], "naive_ml" if key[0] == "figure1" else "equal_train_reuse")], key=lambda row: int(row["replication"]))
            assert all(a["bank_identity"] == b["bank_identity"] and a["fold_hash"] == b["fold_hash"] for a, b in zip(group, other))
    reductions = []
    for metric in main_summary["headline_metrics"]:
        cells = main_summary["primary_results"]
        before = [r for r in cells if r["design"] == metric["design"] and r["method"] == metric["comparator"]]
        after = [r for r in cells if r["design"] == metric["design"] and r["method"] == "dml_crossfit"]
        assert len(before) == len(after) == 6
        a, b = statistics.mean(abs(r["bias"]) for r in before), statistics.mean(abs(r["bias"]) for r in after)
        assert abs(a - metric["before"]) < 1e-12 and abs(b - metric["after"]) < 1e-12
        reductions.append(round(100 * (1 - b / a), 1))
    assert reductions == [89.4, 97.5]
    print(json.dumps({"status": "PASS_INDEPENDENT_SAVED_RESULT_RECOUNT", "exported_jobs": 1000,
        "exported_estimates": 10800, "finite_sample_comparison_estimates": 7200, "primary_csv_estimates": 6000,
        "sensitivity_summary_cells": 600, "nuisance_summary_cells": 106,
        "headline_percent": reductions, "pairing_and_error_decomposition": True,
        "all_result_hashes_match": True, "original_archives_compared": bool(args.input_directory),
        "model_fits": 0, "ML_imports": 0}, indent=2))


if __name__ == "__main__":
    main()
