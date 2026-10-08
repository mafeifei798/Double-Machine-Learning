#!/usr/bin/env python3
"""Actual DoubleML external-prediction parity, both K and both DGP levels."""
import json
from pathlib import Path
import sys
import warnings
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from threadpoolctl import threadpool_limits
from doubleml import DoubleMLPLR,DoubleMLData
import doubleml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from dml_original import sensitivity_v6 as s, harder_v5 as h
from dml_original.iv_thesis_repair import estimate_iv_type


def main():
    assert doubleml.__version__=='0.10.1'
    p=json.loads((ROOT/'configs/sensitivity_v6.json').read_text());q=json.loads((ROOT/p['base_config']).read_text())
    p.update(seed=p['preflight_seed'],sample_sizes=[100],anchor_n=100)
    for family in ('rf','xgb'):
        for role in ('m','g','ell'):q[family][role]['n_estimators']=5
    checks=[]
    with threadpool_limits(limits=1),warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        for k in (2,5):
            for design in ('figure1','figure2'):
                for dgp in ('linear','nonlinear'):
                    c=s.config(p,design,dgp,100,k);e=s.effective(p,q,100,k,'base');folds=s.folds_for(p,c,0)
                    splits=[(np.flatnonzero(folds!=j),np.flatnonzero(folds==j)) for j in range(k)]
                    for level,data in s.draw(p,e,c,0).items():
                        for family in p['learners']:
                            bank=[h.fit_training(e,c,family,0,data.x,data.y,data.d,train) for train,_ in splits]
                            g,m=s.assemble(bank,folds,True);v=estimate_iv_type(data.y,data.d,g,m)
                            model=DoubleMLPLR(DoubleMLData.from_arrays(data.x,data.y,data.d),ml_l=RandomForestRegressor(n_estimators=5),
                                ml_m=RandomForestRegressor(n_estimators=5),ml_g=RandomForestRegressor(n_estimators=5),
                                score='IV-type',n_folds=k,draw_sample_splitting=False)
                            model.set_sample_splitting(splits);model.fit(external_predictions={'d':{'ml_m':m[:,None],'ml_g':g[:,None]}})
                            np.testing.assert_allclose(model.coef[0],v.theta,rtol=0,atol=1e-12)
                            np.testing.assert_allclose(model.se[0],v.se,rtol=0,atol=1e-12)
                            checks.append(dict(k=k,design=design,dgp=dgp,level=level,family=family,tolerance=1e-12))
    print(json.dumps(dict(status='PASS',type='external_predictions_score_only',checks=checks,
        warnings=[dict(category=w.category.__name__,message=str(w.message)) for w in captured]),indent=2))


if __name__=='__main__':main()
