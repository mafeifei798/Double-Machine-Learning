from __future__ import annotations

from dataclasses import replace
import inspect
import math

import numpy as np
import pytest

import dml_studies.iv_score as repair
from dml_studies.iv_score import (
    IVRepairConfig,
    crossfit_structural_nuisances,
    draw_data,
    estimate_iv_type,
    estimate_outcome_mismatch,
    exact_series_basis,
    outer_fold_ids,
    run_figure1_replication,
    run_figure2_replication,
    self_influence_weight,
    training_iv_pilot,
)


@pytest.mark.parametrize("design", ["figure1", "figure2"])
@pytest.mark.parametrize("dgp", ["linear", "nonlinear"])
def test_dgp_preserves_original_plr_identities(design: str, dgp: str) -> None:
    config = IVRepairConfig(
        design=design,
        dgp=dgp,
        n=120,
        p=20,
        reps=1,
        learners=("lasso",),
        n_estimators=5,
    )
    data = draw_data(config, 0)

    assert np.array_equal(data.d, data.m0 + data.v)
    assert np.array_equal(data.y, config.theta0 * data.d + data.g0 + data.u)
    assert np.array_equal(data.ell0, data.g0 + config.theta0 * data.m0)
    basis = exact_series_basis(data.x)
    for truth in (data.g0, data.m0, data.ell0):
        coefficient, _, rank, _ = np.linalg.lstsq(basis, truth, rcond=None)
        assert rank == 5
        assert np.max(np.abs(basis @ coefficient - truth)) < 3e-13


def test_iv_type_estimator_matches_eq43_closed_form() -> None:
    y = np.asarray([1.2, -0.3, 2.4, 0.7, 1.8, -1.0])
    d = np.asarray([0.8, -0.5, 1.5, 0.2, 1.0, -0.7])
    g = np.asarray([0.1, 0.2, 0.4, -0.1, 0.3, 0.0])
    m = np.asarray([0.3, -0.1, 0.7, 0.0, 0.4, -0.2])
    v = d - m
    expected = float(np.dot(v, y - g) / np.dot(v, d))

    result = estimate_iv_type(y, d, g, m)

    assert result.finite
    assert result.theta == pytest.approx(expected)
    assert result.score_mean == pytest.approx(0.0, abs=2e-15)
    assert result.q_hat == pytest.approx(float(np.mean(v * d)))


def test_historical_mismatch_is_a_distinct_non_dml_ratio() -> None:
    y = np.asarray([1.2, -0.3, 2.4, 0.7, 1.8, -1.0])
    d = np.asarray([0.8, -0.5, 1.5, 0.2, 1.0, -0.7])
    ell = np.asarray([0.1, 0.2, 0.4, -0.1, 0.3, 0.0])
    expected = float(np.dot(d, y - ell) / np.dot(d, d))

    result = estimate_outcome_mismatch(y, d, ell)

    assert result.finite
    assert result.theta == pytest.approx(expected)
    assert result.score_mean == pytest.approx(0.0, abs=2e-15)


def test_training_pilot_is_iv_fixed_point_and_near_truth_at_n1000() -> None:
    config = IVRepairConfig(
        design="figure2",
        dgp="linear",
        n=1_000,
        p=20,
        reps=1,
        learners=("lasso",),
        n_estimators=5,
    )
    data = draw_data(config, 3)
    pilot = training_iv_pilot(
        data.x,
        data.y,
        data.d,
        k_folds=2,
        iterations=2,
        seed=991,
    )

    assert math.isfinite(pilot)
    assert abs(pilot - config.theta0) < 0.12


def test_heldout_outcome_cannot_change_its_own_fold_nuisances() -> None:
    config = IVRepairConfig(
        design="figure2",
        dgp="nonlinear",
        n=200,
        p=20,
        reps=1,
        learners=("lasso",),
        n_estimators=5,
    )
    data = draw_data(config, 0)
    folds = outer_fold_ids(config, 0)
    heldout_fold = int(folds[7])
    changed_y = data.y.copy()
    changed_y[folds == heldout_fold] += np.linspace(
        1_000.0, 2_000.0, int(np.sum(folds == heldout_fold))
    )
    changed = replace(data, y=changed_y)

    before = crossfit_structural_nuisances(config, "lasso", 0, data, folds)
    after = crossfit_structural_nuisances(config, "lasso", 0, changed, folds)
    heldout = folds == heldout_fold

    assert np.array_equal(before.g_hat[heldout], after.g_hat[heldout])
    assert np.array_equal(before.m_hat[heldout], after.m_hat[heldout])
    assert before.theta_pilots[heldout_fold] == after.theta_pilots[heldout_fold]
    assert np.max(np.abs(before.g_hat[~heldout] - after.g_hat[~heldout])) > 1.0


@pytest.mark.parametrize("learner", ["rf", "xgb", "lasso"])
def test_figure1_has_only_mismatch_baseline_and_iv_dml(learner: str) -> None:
    config = IVRepairConfig(
        design="figure1",
        dgp="linear",
        n=160,
        p=20,
        reps=1,
        learners=(learner,),
        n_estimators=5,
    )
    rows = run_figure1_replication(config, 0)

    assert [row["method"] for row in rows] == [
        "outcome_nuisance_mismatch",
        "iv_type_crossfit",
    ]
    assert rows[0]["score_id"] == "non_dml_outcome_nuisance_raw_d"
    assert rows[1]["score_id"] == "eq43_iv_type"
    assert rows[0]["bias"] < -0.10
    assert all(row["finite"] for row in rows)
    assert max(abs(float(row["score_mean"])) for row in rows) < 1e-12


@pytest.mark.parametrize("learner", ["rf", "xgb", "lasso"])
def test_figure2_both_sides_use_iv_and_stress_identity_is_exact(learner: str) -> None:
    config = IVRepairConfig(
        design="figure2",
        dgp="nonlinear",
        n=200,
        p=20,
        reps=1,
        learners=(learner,),
        n_estimators=5,
    )
    rows = run_figure2_replication(config, 0)
    no_split, crossfit = rows
    weight = self_influence_weight(config)

    assert [row["fit_mode"] for row in rows] == ["no_split", "crossfit"]
    assert {row["score_id"] for row in rows} == {"eq43_iv_type"}
    assert no_split["theta"] == pytest.approx(
        (1.0 - weight) * no_split["unstressed_full_theta"], abs=2e-12
    )
    assert abs(no_split["stress_identity_error"]) < 2e-12
    assert crossfit["self_influence_weight"] == 0.0
    assert all(row["finite"] for row in rows)


def test_active_module_has_no_alternate_score_dependency() -> None:
    source = inspect.getsource(repair).casefold()

    assert "eq44" not in source
    assert "partialling_out" not in source
    assert "controlled_mechanisms" not in source
    assert "dml_studies.partialling_out" not in source
    assert source.count("eq43_iv_type") >= 3
