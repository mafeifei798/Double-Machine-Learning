#!/usr/bin/env python3
"""24 actual-package IV score/SE checks using external nuisance predictions."""
import json
from pathlib import Path
import sys
import warnings
import numpy as np
from threadpoolctl import threadpool_limits
from sklearn.ensemble import RandomForestRegressor
from doubleml import DoubleMLData,DoubleMLPLR
import doubleml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from dml_original import harder_v5 as study,two_claims_v4 as base,fair_iv as fair
from dml_original.iv_thesis_repair import estimate_iv_type


def main():
    assert doubleml.__version__=='0.10.1'
    p=json.loads((ROOT/'configs/harder_v5.json').read_text());p['n']=240;p['seed']=p['preflight_seed']
    for f in ('rf','xgb'):
        for role in ('m','g','ell'):p[f][role]['n_estimators']=5
    checks=[]
    with threadpool_limits(limits=1),warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        for design in ('figure1','figure2'):
            for dgp in ('linear','nonlinear'):
                c=base.config_for(p,design,dgp);folds=fair.make_fold_ids(c.n,2,19234)
                splits=[(np.flatnonzero(folds!=h),np.flatnonzero(folds==h)) for h in (0,1)]
                for level,data in study.draw_pair(p,c,0).items():
                    for family in p['learners']:
                        bank=tuple(study.fit_training(p,c,family,0,data.x,data.y,data.d,np.flatnonzero(folds==h)) for h in (0,1))
                        g,m=base.assemble(bank,folds,True);estimate=estimate_iv_type(data.y,data.d,g,m)
                        model=DoubleMLPLR(DoubleMLData.from_arrays(data.x,data.y,data.d),
                            ml_l=RandomForestRegressor(n_estimators=5),ml_m=RandomForestRegressor(n_estimators=5),
                            ml_g=RandomForestRegressor(n_estimators=5),score='IV-type',n_folds=2,draw_sample_splitting=False)
                        model.set_sample_splitting(splits);model.fit(external_predictions={'d':{'ml_m':m[:,None],'ml_g':g[:,None]}})
                        np.testing.assert_allclose(model.coef[0],estimate.theta,rtol=0,atol=1e-12)
                        np.testing.assert_allclose(model.se[0],estimate.se,rtol=0,atol=1e-12)
                        checks.append(dict(level=level,design=design,dgp=dgp,family=family,max_tolerance=1e-12))
    print(json.dumps(dict(status='PASS',type='external_predictions_score_only',checks=checks,
        warnings=[dict(category=w.category.__name__,message=str(w.message)) for w in captured]),indent=2))


if __name__=='__main__':main()
