from dataclasses import replace
import inspect
import json
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from sklearn.base import clone

from dml_original import fair_iv as fair
from dml_original.iv_thesis_repair import make_learner, nuisance_truth

PROTOCOL = json.loads((Path(__file__).resolve().parents[1]/"configs/fair_iv_v1.json").read_text())


@pytest.mark.parametrize("design", ["figure1", "figure2"])
@pytest.mark.parametrize("dgp", ["linear", "nonlinear"])
def test_original_functions_and_settings(design, dgp):
    cfg = fair.cell_config(PROTOCOL, design, dgp, development=True)
    assert cfg.n == 1000 and cfg.p == 20 and cfg.theta0 == .5
    assert cfg.lasso_alpha_linear == (.02 if design == "figure1" else .05)
    data = fair.draw(cfg, 0)
    assert np.array_equal(data.y, .5*data.d+data.g0+data.u)
    assert np.array_equal(data.d, data.m0+data.v)
    # Check historical implementations, not just self-consistency of new helpers.
    import importlib.util
    filename = ("figure1_doubleml_docs_style_20260421.py" if design == "figure1"
                else "figure2_compare_learners_common_20260412.py")
    path = Path(__file__).resolve().parents[1]/"legacy_snapshot"/filename
    spec = importlib.util.spec_from_file_location("historical_reference", path)
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    assert np.array_equal(data.m0, getattr(module, f"m0_{'linear' if dgp == 'linear' else 'nonlinear_docs'}")(data.x))
    assert np.array_equal(data.g0, getattr(module, f"g0_{'linear' if dgp == 'linear' else 'nonlinear_docs'}")(data.x))


@pytest.mark.parametrize("learner", ["rf", "xgb", "lasso"])
@pytest.mark.parametrize("split", [False, True])
def test_matches_independently_written_official_algorithm(learner, split):
    cfg = replace(fair.cell_config(PROTOCOL,"figure2","nonlinear",development=True), n=80, n_estimators=5)
    data = fair.draw(cfg, 0)
    folds = fair.make_fold_ids(80, 2, 111) if split else None
    partitions = [(np.arange(80),np.arange(80))] if not split else [
        (np.where(folds != j)[0],np.where(folds == j)[0]) for j in range(2)]
    def cv(role, target):
        seed = fair.seed_for(cfg.seed,cfg.design,cfg.dgp,0,learner,role)
        pred = np.empty(80)
        for tr,te in partitions:
            model = make_learner(cfg,learner,seed)
            model.fit(data.x[tr],target[tr])
            pred[te] = model.predict(data.x[te])
        return pred
    ell, m = cv("ell",data.y), cv("m",data.d)
    psi_a = -(data.d-m)*(data.d-m)
    psi_b = (data.d-m)*(data.y-ell)
    theta_initial = -np.mean(psi_b)/np.mean(psi_a)
    g = cv("g",data.y-theta_initial*data.d)
    pred = fair.official_predictions(cfg,learner,0,data,folds)
    np.testing.assert_array_equal(pred.ell,ell)
    np.testing.assert_array_equal(pred.m,m)
    np.testing.assert_allclose(pred.g,g,rtol=0,atol=1e-14)
    assert pred.pilot == pytest.approx(theta_initial,abs=1e-14)


def test_shared_predictions_and_scores_not_mismatch():
    protocol = {**PROTOCOL, "n": 80,"n_estimators": 5}
    rows = fair.run_replication(protocol,"figure1","linear",0,development=True)
    for learner in protocol["learners"]:
        cell = {r["method"]:r for r in rows if r["learner"] == learner}
        assert set(cell) == set(fair.METHODS)
        assert cell["nonorthogonal_crossfit"]["g_prediction_hash"] == cell["iv_crossfit"]["g_prediction_hash"]
        assert len({r["data_seed"] for r in cell.values()}) == 1
        assert len({r["fold_hash"] for r in cell.values()}) == 1
        assert all(r["artificial_pressure"] is False for r in cell.values())
        assert all(r["treatment_learner"] == learner for r in cell.values())
        assert all(r["finite"] and abs(r["score_mean"]) < 1e-12 for r in cell.values())


def test_forbidden_stress_and_separate_seeds():
    with pytest.raises(ValueError,match="forbidden"):
        fair.cell_config({**PROTOCOL,"artificial_pressure":True},"figure1","linear")
    a = fair.cell_config(PROTOCOL,"figure1","linear")
    b = fair.cell_config(PROTOCOL,"figure1","linear",development=True)
    assert fair.draw(a,0).data_seed != fair.draw(b,0).data_seed


def test_failures_are_retained_not_zeroed():
    with mock.patch.object(fair,"official_predictions",side_effect=FloatingPointError("test")):
        rows = fair.run_replication(PROTOCOL,"figure1","linear",0,development=True)
    assert len(rows) == 9
    assert all(not r["finite"] and "test" in r["failure_reason"] and "theta" not in r for r in rows)


def test_no_oracle_or_series_in_fitting():
    code = inspect.getsource(fair.official_predictions)
    assert "data.g0" not in code and "data.m0" not in code and "SeriesOLS" not in code
    assert "data.y-pilot*data.d" in code
