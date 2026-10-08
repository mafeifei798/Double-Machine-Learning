#!/usr/bin/env python3
"""Actual DoubleML external-prediction score parity, not fitting parity."""
from dataclasses import replace
import json
from pathlib import Path
import sys
import warnings
import numpy as np
from threadpoolctl import threadpool_limits
from doubleml import DoubleMLData,DoubleMLPLR
import doubleml
from sklearn.ensemble import RandomForestRegressor
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from dml_original import two_claims_v4 as study,fair_iv as fair
from dml_original.iv_thesis_repair import estimate_iv_type


def main():
    assert doubleml.__version__=='0.10.1'
    protocol=json.loads((ROOT/'configs/two_claims_v4.json').read_text()); protocol['n']=240
    for family in ('rf','xgb'):
        for role in ('m','g','ell'): protocol[family][role]['n_estimators']=5
    checks=[]
    with threadpool_limits(limits=1),warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        for design in ('figure1','figure2'):
            for dgp in ('linear','nonlinear'):
                c=study.config_for(protocol,design,dgp,True);data=fair.draw(c,0)
                folds=fair.make_fold_ids(c.n,2,18120)
                splits=[(np.flatnonzero(folds!=f),np.flatnonzero(folds==f)) for f in (0,1)]
                dataset=DoubleMLData.from_arrays(data.x,data.y,data.d)
                for family in protocol['learners']:
                    bank=tuple(study.fit_training(protocol,c,family,0,data.x,data.y,data.d,
                        np.flatnonzero(folds==h)) for h in (0,1))
                    g,m=study.assemble(bank,folds,True); result=estimate_iv_type(data.y,data.d,g,m)
                    obj=DoubleMLPLR(dataset,ml_l=RandomForestRegressor(n_estimators=5),
                        ml_m=RandomForestRegressor(n_estimators=5),ml_g=RandomForestRegressor(n_estimators=5),
                        score='IV-type',n_folds=2,draw_sample_splitting=False)
                    obj.set_sample_splitting(splits)
                    obj.fit(external_predictions={'d':{'ml_m':m[:,None],'ml_g':g[:,None]}})
                    np.testing.assert_allclose(obj.coef[0],result.theta,rtol=0,atol=1e-12)
                    np.testing.assert_allclose(obj.se[0],result.se,rtol=0,atol=1e-12)
                    checks.append(dict(design=design,dgp=dgp,family=family,
                        theta_difference=float(obj.coef[0]-result.theta),se_difference=float(obj.se[0]-result.se)))
    print(json.dumps(dict(status='PASS',type='external_predictions_score_only',checks=checks,
        warnings=[dict(category=w.category.__name__,message=str(w.message)) for w in captured]),indent=2))


if __name__=='__main__':main()
