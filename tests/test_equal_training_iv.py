from dataclasses import replace
import inspect
import json
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from dml_original import equal_training_iv as equal
from dml_original import fair_iv as fair

PROTOCOL=json.loads((Path(__file__).resolve().parents[1]/"configs/equal_training_iv_v1.json").read_text())


@pytest.mark.parametrize("design",["figure1","figure2"])
@pytest.mark.parametrize("dgp",["linear","nonlinear"])
def test_same_existing_data_and_folds(design,dgp):
    cfg=equal.config_for(PROTOCOL,design,dgp)
    prior=json.loads((Path(__file__).resolve().parents[1]/"configs/fair_iv_v1.json").read_text())
    a,b=fair.draw(cfg,0),fair.draw(fair.cell_config(prior,design,dgp),0)
    np.testing.assert_array_equal(a.x,b.x); np.testing.assert_array_equal(a.y,b.y)
    assert cfg.lasso_alpha_linear==(.02 if design=="figure1" else .05)


@pytest.mark.parametrize("learner",["rf","xgb","lasso"])
def test_bank_fitted_only_once_with_training_only_pilots(learner):
    cfg=replace(equal.config_for(PROTOCOL,"figure2","nonlinear",True),n=80,n_estimators=5)
    data=fair.draw(cfg,0); folds=fair.make_fold_ids(80,2,111)
    calls=[]; original=equal.make_learner
    def track(*args):
        model=original(*args); original_fit=model.fit
        def fit(x,y):
            calls.append((x.copy(),y.copy())); return original_fit(x,y)
        model.fit=fit; return model
    with mock.patch.object(equal,"make_learner",side_effect=track):
        bank=equal.fit_bank(cfg,learner,0,data,folds)
        reuse=equal.assemble(bank,folds,False); cross=equal.assemble(bank,folds,True)
    assert len(calls)==6
    for j in range(2):
        tr=np.where(folds==j)[0]
        for k in range(3): np.testing.assert_array_equal(calls[3*j+k][0],data.x[tr])
        np.testing.assert_allclose(calls[3*j+2][1],data.y[tr]-bank.pilots[j]*data.d[tr],atol=0,rtol=0)
        seed=fair.seed_for(cfg.seed,cfg.design,cfg.dgp,0,learner,"ell")
        ell=original(cfg,learner,seed).fit(data.x[tr],data.y[tr]).predict(data.x[tr])
        assert bank.pilots[j]==pytest.approx(fair.preliminary_theta(data.y[tr],data.d[tr],ell,bank.m[j,tr]))
        np.testing.assert_array_equal(reuse[0][tr],bank.g[j,tr])
        np.testing.assert_array_equal(cross[0][tr],bank.g[1-j,tr])
    assert np.all(reuse[2]==folds) and np.all(cross[2]!=folds)


def test_heldout_outcomes_cannot_enter_training_bank():
    cfg=replace(equal.config_for(PROTOCOL,"figure1","linear",True),n=80,n_estimators=5)
    data=fair.draw(cfg,0); folds=fair.make_fold_ids(80,2,111)
    a=equal.fit_bank(cfg,"rf",0,data,folds)
    changed=replace(data,y=data.y+1000*(folds==1))
    b=equal.fit_bank(cfg,"rf",0,changed,folds)
    np.testing.assert_array_equal(a.g[0],b.g[0]); np.testing.assert_array_equal(a.m[0],b.m[0])
    assert a.pilots[0]==b.pilots[0]


def test_raw_contract_and_shared_bank():
    protocol={**PROTOCOL,"n":80,"n_estimators":5}
    rows=equal.run_replication(protocol,"figure2","linear",0,development=True)
    assert len(rows)==6
    for learner in protocol["learners"]:
        a,b=[r for r in rows if r["learner"]==learner]
        assert a["bank_identity"]==b["bank_identity"] and a["bank_records"]==b["bank_records"]
        assert a["pilots"]==b["pilots"]
        assert a["prediction_overlap_n"]==80 and b["prediction_overlap_n"]==0
        assert all(r["finite"] and r["train_n_per_model"]==40 and r["evaluation_n"]==80 for r in (a,b))
        for r in (a,b):
            assert abs(r["score_mean"])<1e-12
            assert r["score_at_truth"]==pytest.approx(sum(r[k] for k in ("oracle_score","first_order_score","second_order_score")),abs=1e-12)


def test_failures_retained_and_stress_forbidden():
    with mock.patch.object(equal,"fit_bank",side_effect=FloatingPointError("test")):
        rows=equal.run_replication(PROTOCOL,"figure1","linear",0)
    assert len(rows)==6 and all(not r["finite"] and "test" in r["failure_reason"] for r in rows)
    with pytest.raises(ValueError): equal.config_for({**PROTOCOL,"artificial_pressure":True},"figure1","linear")
    with pytest.raises(ValueError): equal.config_for({**PROTOCOL,"n":81},"figure1","linear")


def test_no_truth_or_stress_training():
    code=inspect.getsource(equal.fit_bank)
    assert "data.g0" not in code and "data.m0" not in code and "SeriesOLS" not in code
    assert "data.y[train] - pilot * data.d[train]" in code
