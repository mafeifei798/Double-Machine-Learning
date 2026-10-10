import copy
from dataclasses import replace
import json
from pathlib import Path
import sys
import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from dml_studies import orthogonalization_crossfitting as study, doubleml_reference as fair
from dml_studies.iv_score import estimate_iv_type


@pytest.fixture
def protocol():
    p=json.loads((ROOT/'configs/orthogonalization_crossfitting.json').read_text())
    p['n']=240
    for family in ('rf','xgb'):
        for role in ('m','g','ell'): p[family][role]['n_estimators']=5
    return p


@pytest.mark.parametrize('family',['lasso','rf','xgb'])
@pytest.mark.parametrize('role',['m','g','ell'])
def test_real_families(protocol,family,role):
    model=study.model_for(protocol,family,role,'nonlinear',123)
    if family=='lasso': assert isinstance(model.named_steps['lasso'],Lasso)
    elif family=='rf': assert isinstance(model,RandomForestRegressor)
    else:
        from xgboost import XGBRegressor
        assert isinstance(model,XGBRegressor)


@pytest.mark.parametrize('family',['lasso','rf','xgb'])
@pytest.mark.parametrize('design',['figure1','figure2'])
def test_no_test_response_leak(protocol,family,design):
    c=study.config_for(protocol,design,'nonlinear',True); data=fair.draw(c,0)
    train=np.arange(120); a=study.fit_training(protocol,c,family,0,data.x,data.y,data.d,train)
    y=data.y.copy(); d=data.d.copy(); y[120:]+=100; d[120:]-=50
    b=study.fit_training(protocol,c,family,0,data.x,y,d,train)
    np.testing.assert_array_equal(a.g,b.g); np.testing.assert_array_equal(a.m,b.m)
    assert a.record==b.record
    expected=data.y[train]-a.record['pilot']*data.d[train] if design=='figure1' else data.y[train]
    assert a.record['target_sha256']==fair.vector_hash(expected)


@pytest.mark.parametrize('design',['figure1','figure2'])
@pytest.mark.parametrize('dgp',['linear','nonlinear'])
def test_all_cells_and_paired_identity(protocol,design,dgp):
    rows=study.run_replication(protocol,design,dgp,0)
    assert {(r['learner'],r['method']) for r in rows}=={(l,m) for l in protocol['learners'] for m in study.methods(design)}
    for family in protocol['learners']:
        r={v['method']:v for v in rows if v['learner']==family}
        a=r['naive_ml' if design=='figure1' else 'equal_train_reuse']; b=r['dml_crossfit']
        assert a['bank_identity']==b['bank_identity']
        assert b['prediction_overlap_n']==0
        if design=='figure1': assert a['g_prediction_hash']==b['g_prediction_hash']
        else: assert a['prediction_overlap_n']==240 and a['train_n_per_model']==b['train_n_per_model']==120
    for r in rows:
        assert r['finite'] and abs(r['score_mean'])<1e-10
        if r['method']!='naive_ml':
            assert abs(sum(r[k] for k in ('weighted_oracle','weighted_first_order','weighted_second_order'))-r['bias'])<1e-10


@pytest.mark.parametrize('family',['rf','xgb','lasso'])
def test_monotonic_treatment(protocol,family):
    c=study.config_for(protocol,'figure1','linear',True); data=fair.draw(c,0)
    model=study.model_for(protocol,family,'m','linear',123)
    model.fit(study.features(data.x,family,'m','linear'),data.d)
    for column in (0,2):
        grid=np.zeros((100,20)); grid[:,column]=np.linspace(-3,3,100)
        assert np.min(np.diff(model.predict(study.features(grid,family,'m','linear'))))>=-1e-12


def test_truth_not_in_fit_interface():
    import inspect
    assert tuple(inspect.signature(study.fit_training).parameters)==('protocol','config','family','rep','x','y','d','train')
    source=inspect.getsource(study.fit_training)+inspect.getsource(study.joint_pilot)
    assert all(name not in source for name in ('g0','m0','theta0','lstsq','SeriesOLS'))
