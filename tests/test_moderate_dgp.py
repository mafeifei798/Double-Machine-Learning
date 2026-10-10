import json
import inspect
from pathlib import Path
import sys
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from dml_studies import moderate_dgp as study,orthogonalization_crossfitting as base,doubleml_reference as fair
from dml_studies.iv_score import nuisance_truth


@pytest.fixture
def protocol():
    p=json.loads((ROOT/'configs/moderate_dgp.json').read_text());p['n']=240;p['seed']=p['preflight_seed']
    for f in ('rf','xgb'):
        for role in ('m','g','ell'):p[f][role]['n_estimators']=5
    return p


@pytest.mark.parametrize('design',['figure1','figure2'])
@pytest.mark.parametrize('dgp',['linear','nonlinear'])
def test_truth_and_common_random_numbers(protocol,design,dgp):
    c=base.config_for(protocol,design,dgp);pair=study.draw_pair(protocol,c,0);a,b=pair.values()
    for role in ('x','u','v'):np.testing.assert_array_equal(getattr(a,role),getattr(b,role))
    expected=nuisance_truth(design,dgp,a.x,c.theta0)
    for role,value in zip(('g0','m0','ell0'),expected):np.testing.assert_array_equal(getattr(a,role),value)
    assert not np.array_equal(a.d,b.d) and not np.array_equal(a.y,b.y)
    np.testing.assert_allclose(a.y,.5*a.d+a.g0+a.u,rtol=0,atol=1e-14)
    np.testing.assert_allclose(b.y,.5*b.d+b.g0+b.u,rtol=0,atol=1e-14)


@pytest.mark.parametrize('family',['lasso','rf','xgb'])
@pytest.mark.parametrize('design',['figure1','figure2'])
@pytest.mark.parametrize('level',['baseline','moderate'])
def test_response_isolation(protocol,family,design,level):
    c=base.config_for(protocol,design,'nonlinear');data=study.draw_pair(protocol,c,0)[level];train=np.arange(120)
    a=study.fit_training(protocol,c,family,0,data.x,data.y,data.d,train)
    y=data.y.copy();d=data.d.copy();y[120:]+=200;d[120:]-=150
    b=study.fit_training(protocol,c,family,0,data.x,y,d,train)
    np.testing.assert_array_equal(a.g,b.g);np.testing.assert_array_equal(a.m,b.m);assert a.record==b.record


@pytest.mark.parametrize('design',['figure1','figure2'])
@pytest.mark.parametrize('dgp',['linear','nonlinear'])
def test_exact_method_bank_pairing(protocol,design,dgp):
    rows=study.run_replication(protocol,design,dgp,0)
    assert len(rows)==len(protocol['levels'])*3*len(base.methods(design))
    assert len({r['fold_hash'] for r in rows})==1
    for role in ('x_hash','u_hash','v_hash'):assert len({r[role] for r in rows})==1
    for level in protocol['levels']:
        for family in protocol['learners']:
            group={r['method']:r for r in rows if r['level']==level and r['learner']==family}
            a=group['naive_ml' if design=='figure1' else 'equal_train_reuse'];b=group['dml_crossfit']
            assert a['bank_records']==b['bank_records'] and a['bank_identity']==b['bank_identity']
            assert b['prediction_overlap_n']==0
            if design=='figure1':assert a['g_prediction_hash']==b['g_prediction_hash']
            else:assert a['prediction_overlap_n']==240
    assert all(r['finite'] and abs(r['score_mean'])<1e-10 for r in rows)
    for row in rows:
        if row['method']!='naive_ml':
            assert abs(row['bias']-sum(row[k] for k in ('weighted_oracle','weighted_first_order','weighted_second_order')))<1e-10


@pytest.mark.parametrize('dgp',['linear','nonlinear'])
def test_treatment_truth_remains_monotone(protocol,dgp):
    c=base.config_for(protocol,'figure1',dgp)
    for column in protocol['treatment_columns']:
        x=np.zeros((100,20));x[:,column]=np.linspace(-3,3,100)
        _,m,_=study.truth(protocol,c,x,'moderate');assert np.min(np.diff(m))>0


@pytest.mark.parametrize('family',['lasso','rf','xgb'])
def test_model_families_and_inputs(protocol,family):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Lasso
    from xgboost import XGBRegressor
    x=np.zeros((40,20))
    for role in ('m','g','ell'):
        model=base.model_for(protocol,family,role,'nonlinear',123)
        assert isinstance(model.named_steps['lasso'],Lasso) if family=='lasso' else isinstance(model,RandomForestRegressor if family=='rf' else XGBRegressor)
    assert study.features(protocol,x,family,'m','nonlinear').shape==(40,8 if family=='lasso' else 4)
    assert 'level' not in inspect.signature(study.fit_training).parameters
    source=inspect.getsource(study.fit_training)
    assert all(v not in source for v in ('g0','m0','theta0','truth('))
