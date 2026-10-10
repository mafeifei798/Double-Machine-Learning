#!/usr/bin/env python3
"""Read-only verification of selected DML results from frozen summaries.

This script uses only the Python standard library. It does not import learners,
fit models, launch jobs, or modify evidence.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V6_SUMMARY = "output/sensitivity_v6_server_20261002/evidence/output/formal/analysis/SUMMARY.json"
V3_SUMMARY = "output/iv_repair_v3_server_20261002/output/formal/analysis/summary.json"


def read_json(relative: str):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def sha256(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def build_evidence() -> dict:
    v6 = read_json(V6_SUMMARY)
    v3 = read_json(V3_SUMMARY)
    check(sha256(V6_SUMMARY) == "6b5e66c4ec9c8c53c0f5a0c2542c361deb8aad6a24afd17714bedbfd36c05117",
          "v6 evidence differs from the frozen result")
    check(v6["failures"] == 0, "v6 contains scientific failures")
    anchor = [s for s in v6["summaries"] if
              (s["level"], s["n"], s["k_folds"], s["preset"]) ==
              ("moderate", 1000, 2, "base")]
    expected_cells = {(dgp, learner) for dgp in ("linear", "nonlinear")
                      for learner in ("lasso", "rf", "xgb")}
    aggregates = []
    for design, before_method, claimed in (
            ("figure1", "naive_ml", 89.4),
            ("figure2", "full_no_split", 97.5)):
        before = {(s["dgp"], s["learner"]): s for s in anchor
                  if s["design"] == design and s["method"] == before_method}
        after = {(s["dgp"], s["learner"]): s for s in anchor
                 if s["design"] == design and s["method"] == "dml_crossfit"}
        check(set(before) == expected_cells == set(after), "incomplete six-cell comparison")
        check(all(s["R"] == 200 for s in list(before.values()) + list(after.values())),
              "unexpected anchor replication count")
        b = math.fsum(abs(s["bias"]["mean"]) for s in before.values()) / 6
        a = math.fsum(abs(s["bias"]["mean"]) for s in after.values()) / 6
        percent = 100 * (1 - a / b)
        check(round(percent, 1) == claimed, "headline percentage does not match saved evidence")
        aggregates.append({
            "design": design, "before_method": before_method,
            "after_method": "dml_crossfit", "cells_each": 6, "R_per_cell": 200,
            "absolute_mean_bias_before": b, "absolute_mean_bias_after": a,
            "reduction_percent": percent, "rounded_percent": claimed,
            "cells": [{"dgp": key[0], "learner": key[1],
                       "before_signed_mean_bias": before[key]["bias"]["mean"],
                       "after_signed_mean_bias": after[key]["bias"]["mean"]}
                      for key in sorted(expected_cells)],
        })

    trees = [s for s in v3 if s["track"] == "controlled" and s["design"] == "figure2"
             and s["learner"].startswith("tree_leaf_") and s["method"] == "crossfit"]
    check({(s["dgp"], int(s["learner"].rsplit("_", 1)[1])) for s in trees} ==
          {(dgp, leaf) for dgp in ("linear", "nonlinear") for leaf in (2, 5, 10, 20)}
          and len(trees) == 8, "incomplete eight-setting Series OLS / tree study")
    max_bias = max(abs(s["bias"]["mean"]) for s in trees)
    check(max_bias < 0.0046 and all(s["R"] == 200 for s in trees),
          "eight-setting bound or replication count does not match")

    check(sorted({s["n"] for s in v6["summaries"]}) == [300, 1000, 3000], "sample-size scope changed")
    check(sorted({s["k_folds"] for s in v6["summaries"]}) == [2, 5], "fold-count scope changed")
    check(len(v6["summaries"]) == 600 and len(v6["method_contrasts"]) == 360
          and len(v6["factor_contrasts"]) == 960, "sensitivity universe changed")
    check(all("mae_holm_p" in c and "mse_holm_p" in c for c in v6["method_contrasts"]),
          "missing global method-family Holm adjustments")
    check(all("decomposition" in s for s in v6["summaries"] if s["method"] == "dml_crossfit"),
          "missing error-decomposition evidence")

    def anchor_row(design: str, dgp: str, learner: str, method: str) -> dict:
        matches = [s for s in anchor if (s["design"], s["dgp"], s["learner"], s["method"]) ==
                   (design, dgp, learner, method)]
        check(len(matches) == 1, "ambiguous anchor cell")
        return matches[0]

    lasso_before = anchor_row("figure2", "linear", "lasso", "full_no_split")
    lasso_after = anchor_row("figure2", "linear", "lasso", "dml_crossfit")
    worst = min((s for s in v6["summaries"] if s["method"] == "dml_crossfit"),
                key=lambda s: s["coverage"]["rate"])
    return {
        "schema_version": 1,
        "status": "SAVED_RESULTS_VERIFIED",
        "source_evidence": [{"path": p, "sha256": sha256(p)} for p in (V6_SUMMARY, V3_SUMMARY)],
        "bias_reduction": {
            "setting": {"study": "sensitivity_v6", "level": "moderate", "n": 1000,
                        "p": 20, "theta0": 0.5, "k_folds": 2, "preset": "base"},
            "metric_definition": "Equal-weight average of abs(Monte Carlo mean(theta_hat - theta0)) across six learner-by-DGP cells. Not MAE, not mean percentage improvement, not a full-grid aggregate.",
            "reduction_formula": "100 * (1 - absolute_mean_bias_after / absolute_mean_bias_before)",
            "comparisons": aggregates,
            "boundary": "Controlled known treatment-candidate / monotone structure; learned nuisance functions, no oracle fitting. Do not substitute v4 or v5 percentages.",
        },
        "training_data_reuse": {
            "matched_model_evidence": ["src/dml_original/sensitivity_v6.py",
                                       "src/dml_original/two_claims_v4.py",
                                       "docs/SENSITIVITY_V6_PROTOCOL.md",
                                       "output/sensitivity_v6_server_20261002/evidence/INDEPENDENT_VERIFICATION.json"],
            "matched_model_definition": "Reuse and cross-fit share fitted model banks, training sizes, targets, pilots, and random states; prediction membership differs. Full-sample no-split remains a separate arm.",
            "tree_study": "iv_repair_v3", "tree_setting_count": 8,
            "R_per_tree_cell": 200, "maximum_absolute_mean_bias": max_bias,
            "tree_cells": [{"dgp": s["dgp"], "min_samples_leaf": int(s["learner"].rsplit("_", 1)[1]),
                            "signed_mean_bias": s["bias"]["mean"], "mc95": s["bias"],
                            "coverage": s["coverage"]["rate"]}
                           for s in sorted(trees, key=lambda s: (s["dgp"], int(s["learner"].rsplit("_", 1)[1])))],
            "boundary": "The 0.0046 bound concerns point estimates of absolute Monte Carlo mean bias, not MAE, RMSE, all bias-interval endpoints, or an exact-unbiasedness guarantee. The treatment learner is correctly specified Series OLS.",
        },
        "sensitivity": {
            "study": "sensitivity_v6", "sample_sizes": [300, 1000, 3000], "fold_counts": [2, 5],
            "presets": ["base", "outcome_weak", "outcome_strong", "treatment_weak", "treatment_strong"],
            "parameter_scope": "Base n-by-K grid plus one-role-at-a-time alternatives at n=1000,K=2; not a full hyperparameter factorial.",
            "method_contrast_count": 360, "factor_contrast_count": 960,
            "holm_definition": "Paired MAE/MSE improvement tests with separate global 360-test method families; factor tests have separate axis-by-metric families. Headline bias percentages are descriptive point summaries, not Holm-adjusted effect sizes.",
            "lasso_bias_variance_example": {
                "setting": "moderate/figure2/linear/n1000/K2/base/lasso",
                "no_split_bias": lasso_before["bias"]["mean"], "crossfit_bias": lasso_after["bias"]["mean"],
                "no_split_mae": lasso_before["mae"], "crossfit_mae": lasso_after["mae"],
                "no_split_rmse": lasso_before["rmse"], "crossfit_rmse": lasso_after["rmse"],
            },
            "worst_crossfit_coverage": {
                "setting": {k: worst[k] for k in ("level", "design", "dgp", "n", "k_folds", "preset", "learner")},
                "coverage": worst["coverage"], "signed_mean_bias": worst["bias"]["mean"],
            },
            "boundary": "Assessing robustness does not establish parameter insensitivity, universal unbiasedness, or uniformly valid inference. Five folds are not uniformly superior to two.",
        },
        "study_scope": {
            "primary": ["v6 moderate anchor for headline percentages", "v6 complete sensitivity and limitations", "v3 eight-setting Series OLS / tree mechanism"],
            "supporting": ["v4 formal R=1000 experiment", "v5 exploratory paired difficulty extension"],
            "appendix": ["Earlier ordinary and equal-training adverse findings; do not hide them"],
            "chronology": "The original undergraduate project ran from February through June 2026. Supporting repair, extension, and verification runs were completed in October 2026. Original timestamps and provenance are retained.",
        },
        "execution": {"read_only_summary_check": True, "model_fits": 0, "experiments_relaunched": 0,
                      "evidence_modified": False},
    }


if __name__ == "__main__":
    print(json.dumps(build_evidence(), indent=2, ensure_ascii=False, allow_nan=False))
