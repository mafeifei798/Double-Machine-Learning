from __future__ import annotations

import math

import numpy as np
import pytest

from dml_original.simulation import (
    SimConfig,
    aggregate_rows,
    fold_hash,
    make_fold_ids,
    partialling_out,
    run_replication,
    run_overfit_replication,
    wilson_interval,
)


def test_partialling_out_matches_closed_form_and_score_equation() -> None:
    y = np.asarray([2.0, 0.5, 4.0, 1.0, 3.0])
    d = np.asarray([1.5, -0.5, 2.0, 0.0, 1.0])
    ell = np.asarray([0.8, 0.2, 1.4, 0.4, 1.1])
    m = np.asarray([0.5, -0.2, 0.7, 0.1, 0.4])
    v = d - m
    r = y - ell
    expected = float(np.dot(v, r) / np.dot(v, v))

    result = partialling_out(y, d, ell, m)

    assert result.finite
    assert result.theta == pytest.approx(expected)
    assert result.score_mean == pytest.approx(0.0, abs=1e-14)
    assert result.jacobian == pytest.approx(-float(np.mean(v * v)))


def test_noiseless_oracle_recovers_theta_to_machine_precision() -> None:
    theta0 = 0.5
    m0 = np.asarray([-1.0, -0.2, 0.4, 1.1, 1.6, 2.0])
    v = np.asarray([0.5, -1.2, 0.8, -0.7, 1.3, -0.9])
    d = m0 + v
    ell0 = np.asarray([0.2, 0.1, -0.4, 0.7, 1.0, -0.5])
    y = ell0 + theta0 * v

    result = partialling_out(y, d, ell0, m0)

    assert result.finite
    assert result.theta == pytest.approx(theta0, abs=2e-15)


def test_degenerate_or_nonfinite_inputs_fail_explicitly() -> None:
    y = np.ones(5)
    d = np.ones(5)
    degenerate = partialling_out(y, d, np.zeros(5), d.copy())
    nonfinite = partialling_out(y, d, np.asarray([0.0, 0.0, math.nan, 0.0, 0.0]), np.zeros(5))

    assert not degenerate.finite
    assert degenerate.failure_reason == "degenerate_residual_treatment_signal"
    assert not nonfinite.finite
    assert nonfinite.failure_reason == "non_finite_or_shape_mismatch"


def test_folds_are_exact_deterministic_and_hash_encodes_k() -> None:
    fold2 = make_fold_ids(101, 2, 123)
    fold5 = make_fold_ids(101, 5, 123)

    assert np.array_equal(fold2, make_fold_ids(101, 2, 123))
    assert set(fold2) == {0, 1}
    assert set(fold5) == {0, 1, 2, 3, 4}
    assert all(np.sum(fold5 == fold) > 0 for fold in range(5))
    assert fold_hash(fold2, 2) != fold_hash(fold5, 5)


def test_one_replication_reuses_partition_and_writes_complete_method_universe() -> None:
    config = SimConfig(
        design="figure1",
        dgp="linear",
        n=60,
        p=5,
        reps=1,
        k_folds=2,
        learners=("lasso",),
        n_estimators=5,
    )

    rows = run_replication(config, 0)

    assert len(rows) == 4
    assert {row["method"] for row in rows} == {
        "oracle_po",
        "outcome_only_nonorthogonal",
        "po_no_split",
        "po_crossfit",
    }
    assert len({row["fold_hash"] for row in rows}) == 1
    assert all(row["finite"] for row in rows)
    assert max(abs(float(row["score_mean"])) for row in rows) < 1e-12


def test_overfit_stress_is_explicit_paired_and_finite() -> None:
    config = SimConfig(
        design="figure2",
        dgp="nonlinear",
        n=60,
        p=5,
        reps=1,
        k_folds=2,
        learners=("rf",),
        n_estimators=5,
    )

    rows = run_overfit_replication(config, 0)

    assert len(rows) == 2
    assert {row["method"] for row in rows} == {
        "overfit_po_no_split",
        "overfit_po_crossfit",
    }
    assert len({row["fold_hash"] for row in rows}) == 1
    assert all(row["finite"] for row in rows)
    assert all(row["learner"] == "overfit_outcome_tree_plus_rf" for row in rows)


def test_wilson_interval_has_uncertainty_at_boundaries() -> None:
    low0, high0 = wilson_interval(0, 60)
    low1, high1 = wilson_interval(60, 60)

    assert low0 == 0.0
    assert high0 > 0.0
    assert low1 < 1.0
    assert high1 == 1.0


def test_aggregate_counts_failures_in_failure_rate() -> None:
    base = {
        "design": "figure1",
        "dgp": "linear",
        "learner": "rf",
        "method": "po_crossfit",
        "se": 0.1,
        "nuisance_rmse_ell": 0.2,
        "nuisance_rmse_m": 0.1,
    }
    rows = [
        {**base, "finite": True, "bias": 0.0, "covers": True},
        {**base, "finite": True, "bias": 0.2, "covers": False},
        {
            **base,
            "finite": False,
            "bias": math.nan,
            "covers": False,
            "se": math.nan,
            "nuisance_rmse_ell": math.nan,
            "nuisance_rmse_m": math.nan,
        },
    ]

    summary = aggregate_rows(rows)[0]

    assert summary["replications"] == 3
    assert summary["finite_replications"] == 2
    assert summary["failure_rate"] == pytest.approx(1.0 / 3.0)
    assert summary["coverage"] == pytest.approx(0.5)


def test_invalid_config_is_rejected_instead_of_silently_changed() -> None:
    with pytest.raises(ValueError, match="k_folds"):
        SimConfig(n=20, p=5, k_folds=21).validate()
