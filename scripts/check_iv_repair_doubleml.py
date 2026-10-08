#!/usr/bin/env python3
"""Actual-package parity: historical fitting, repaired externally supplied score."""
import json
from dataclasses import replace
from pathlib import Path
import sys
import warnings
import numpy as np
from threadpoolctl import threadpool_limits
from doubleml import DoubleMLData,DoubleMLPLR
import doubleml
from sklearn.ensemble import RandomForestRegressor

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from dml_original import fair_iv as fair,iv_repair_v3 as repair
from dml_original.iv_thesis_repair import make_learner,estimate_iv_type


def main():
    assert doubleml.__version__=="0.10.1"
    old=json.loads((ROOT/"configs/fair_iv_v1.json").read_text())
    protocol=json.loads((ROOT/"configs/iv_repair_v3.json").read_text())
    checks=[]
    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        for design in ("figure1","figure2"):
            for dgp in ("linear","nonlinear"):
                config=replace(fair.cell_config(old,design,dgp,development=True),n=240,n_estimators=20)
                data=fair.draw(config,0); folds=fair.make_fold_ids(config.n,2,9812)
                splits=[(np.flatnonzero(folds!=fold),np.flatnonzero(folds==fold)) for fold in (0,1)]
                dataset=DoubleMLData.from_arrays(data.x,data.y,data.d)
                for family in ("rf","xgb"):
                    models={role:make_learner(config,family,fair.seed_for(config.seed,design,dgp,0,family,role)) for role in ("ell","m","g")}
                    obj=DoubleMLPLR(dataset,ml_l=models["ell"],ml_m=models["m"],ml_g=models["g"],
                        score="IV-type",n_folds=2,draw_sample_splitting=False)
                    obj.set_sample_splitting(splits); obj.fit()
                    predictions=fair.official_predictions(config,family,0,data,folds)
                    for role,name in (("ell","ml_l"),("m","ml_m"),("g","ml_g")):
                        np.testing.assert_allclose(obj.predictions[name][:,0,0],getattr(predictions,role),rtol=0,atol=1e-12)
                    estimate=estimate_iv_type(data.y,data.d,predictions.g,predictions.m)
                    np.testing.assert_allclose(obj.coef[0],estimate.theta,rtol=0,atol=1e-12)
                    np.testing.assert_allclose(obj.se[0],estimate.se,rtol=0,atol=1e-12)
                    checks.append(dict(type="actual_fit_old_pooled_algorithm",design=design,dgp=dgp,family=family,max_tolerance=1e-12))
                new_config=replace(repair.config_for(protocol,design,dgp,True),n=240,n_estimators=5)
                # Repaired bank is deliberately not the official fitting algorithm.
                for track,learner in (("controlled","series_ols"),("controlled","tree_leaf_2"),("ordinary","rf")):
                    params={role:repair.candidates("rf")[0] for role in ("m","g")} if track=="ordinary" else {}
                    bank=tuple(repair.fit_training(new_config,track,learner,params,0,data.x,data.y,data.d,np.flatnonzero(folds==half)) for half in (0,1))
                    g,m,_,_=repair.assemble(bank,folds,True)
                    estimate=estimate_iv_type(data.y,data.d,g,m)
                    obj=DoubleMLPLR(dataset,ml_l=RandomForestRegressor(n_estimators=5),ml_m=RandomForestRegressor(n_estimators=5),
                        ml_g=RandomForestRegressor(n_estimators=5),score="IV-type",n_folds=2,draw_sample_splitting=False)
                    obj.set_sample_splitting(splits)
                    obj.fit(external_predictions={"d": {"ml_m":m[:,None],"ml_g":g[:,None]}})
                    np.testing.assert_allclose(obj.coef[0],estimate.theta,rtol=0,atol=1e-12)
                    np.testing.assert_allclose(obj.se[0],estimate.se,rtol=0,atol=1e-12)
                    checks.append(dict(type="repaired_external_predictions_score_only",design=design,dgp=dgp,track=track,learner=learner,max_tolerance=1e-12))
    print(json.dumps(dict(status="PASS",doubleml=doubleml.__version__,checks=checks,
        warnings=[dict(category=w.category.__name__,message=str(w.message)) for w in captured]),indent=2))


if __name__=="__main__": main()
