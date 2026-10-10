import inspect
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from dml_studies import doubleml_reference as fair
from dml_studies import nuisance_comparison as repair

P=json.loads((Path(__file__).resolve().parents[1]/"configs/nuisance_comparison.json").read_text())


def selections():
    return {f"{design}/{dgp}/{family}":{role:repair.candidates(family)[0] for role in ("m","g")}
        for design in ("figure1","figure2") for dgp in ("linear","nonlinear")
        for family in P["ordinary_families"]}


@pytest.mark.parametrize("track,learner",[("ordinary","rf"),("ordinary","xgb"),("ordinary","lasso"),
    ("controlled","series_ols"),("controlled","tree_leaf_2"),("controlled","rf_outcome_series_m")])
def test_holdout_outcome_cannot_change_its_predictions(track,learner):
    config=replace(repair.config_for(P,"figure2","nonlinear",True),n=240,n_estimators=5)
    data=fair.draw(config,0); train=np.arange(120)
    params={role:repair.candidates(learner)[0] for role in ("m","g")} if track=="ordinary" else {}
    first=repair.fit_training(config,track,learner,params,0,data.x,data.y,data.d,train)
    changed=data.y.copy(); changed[120:]+=1000
    second=repair.fit_training(config,track,learner,params,0,data.x,changed,data.d,train)
    np.testing.assert_array_equal(first.g,second.g)
    np.testing.assert_array_equal(first.m,second.m)
    assert first.pilot==second.pilot and first.record==second.record
    if track=="controlled": np.testing.assert_array_equal(first.g,first.ell-first.pilot*first.m)


@pytest.mark.parametrize("design",["figure1","figure2"])
@pytest.mark.parametrize("dgp",["linear","nonlinear"])
def test_all_cells_methods_pairing_and_exact_decomposition(design,dgp):
    protocol={**P,"n":240,"n_estimators":5}
    rows=repair.run_replication(protocol,selections(),design,dgp,0,development=True)
    assert all(r["finite"] for r in rows)
    assert all(abs(r["score_mean"])<1e-10 for r in rows)
    groups={(r["track"],r["learner"]) for r in rows}
    for group in groups:
        pair={r["method"]:r for r in rows if (r["track"],r["learner"])==group}
        a,b=pair["equal_train_reuse"],pair["crossfit"]
        assert a["bank_identity"]==b["bank_identity"] and a["bank_records"]==b["bank_records"]
        assert a["train_n_per_model"]==b["train_n_per_model"]==120
        assert a["prediction_overlap_n"]==240 and b["prediction_overlap_n"]==0
        assert pair["full_no_split"]["train_n_per_model"]==240
        assert pair["nonorthogonal_crossfit"]["g_prediction_hash"]==b["g_prediction_hash"]
        for method in ("full_no_split","equal_train_reuse","crossfit"):
            r=pair[method]
            assert sum(r[k] for k in ("weighted_oracle","weighted_first_order","weighted_second_order"))==pytest.approx(r["bias"],abs=1e-10)
    if design=="figure1":
        assert any(r["method"]=="outcome_mismatch_crossfit" for r in rows)


def test_pilot_is_joint_ols_not_truth():
    config=repair.config_for(P,"figure1","nonlinear",True); data=fair.draw(config,0)
    tr=np.arange(500); design=np.column_stack((data.d[tr],repair.rich_basis(data.x[tr])))
    expected=np.linalg.lstsq(design,data.y[tr],rcond=None)[0][0]
    assert repair.joint_pilot(data.x[tr],data.y[tr],data.d[tr])==expected
    for fn in (repair.fit_training,repair.joint_pilot,repair.selection_job):
        source=inspect.getsource(fn)
        assert "data.g0" not in source and "data.m0" not in source and "theta0" not in source


def test_prediction_selection_has_no_theta_outcomes():
    protocol={**P,"n":240,"n_estimators":5}
    result=repair.selection_job(protocol,"figure1","linear",0,"lasso","g")
    assert len(result["squared_errors"])==4 and result["validation_n"]==120
    assert not {"theta","bias","coverage","m0","g0","theta0"}.intersection(result)
    assert all(np.isfinite(result["squared_errors"]))


def test_disjoint_seeds():
    a=repair.config_for(P,"figure1","linear",True)
    b=repair.config_for(P,"figure1","linear",False)
    assert fair.draw(a,0).data_seed!=fair.draw(b,0).data_seed
