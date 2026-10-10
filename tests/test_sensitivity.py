from copy import deepcopy
import json
from pathlib import Path
import sys
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from dml_studies import sensitivity as s, moderate_dgp as h, orthogonalization_crossfitting as b


@pytest.fixture
def inputs():
    p=json.loads((ROOT/'configs/sensitivity.json').read_text());q=json.loads((ROOT/p['base_config']).read_text())
    p.update(seed=p['preflight_seed'],sample_sizes=[100,200],anchor_n=200,replications=2)
    for family in ('rf','xgb'):
        for role in ('m','g','ell'):q[family][role]['n_estimators']=5
    return p,q


def test_grid_frozen():
    p=json.loads((ROOT/'configs/sensitivity.json').read_text())
    assert len(set(s.scenarios(p)))==10
    assert p['replications']==200 and p['seed']!=p['preflight_seed']
    assert p['sample_sizes']==[300,1000,3000] and p['fold_counts']==[2,5]


@pytest.mark.parametrize('design',['figure1','figure2'])
@pytest.mark.parametrize('dgp',['linear','nonlinear'])
def test_nested_data_and_presets(inputs,design,dgp):
    p,q=inputs;a=s.draw(p,q,s.config(p,design,dgp,100,2),0);z=s.draw(p,q,s.config(p,design,dgp,200,5),0)
    for level in p['levels']:
        for field in ('x','u','v','g0','m0','y','d'):np.testing.assert_array_equal(getattr(a[level],field),getattr(z[level],field)[:100])
    for field in ('x','u','v'):np.testing.assert_array_equal(getattr(a['baseline'],field),getattr(a['moderate'],field))


@pytest.mark.parametrize('k',[2,5])
def test_single_model_assignment(k):
    n=100;folds=np.repeat(np.arange(k),n//k)
    bank=[b.Predictions(np.full(n,j),np.full(n,100+j),{}) for j in range(k)]
    cg,cm=s.assemble(bank,folds,True);rg,rm=s.assemble(bank,folds,False)
    np.testing.assert_array_equal(cg,folds);np.testing.assert_array_equal(rg,(folds+1)%k)
    np.testing.assert_array_equal(cm,cg+100);np.testing.assert_array_equal(rm,rg+100)
    assert np.all(rg!=folds)


@pytest.mark.parametrize('preset',['base','outcome_weak','outcome_strong','treatment_weak','treatment_strong'])
def test_one_role_changed(inputs,preset):
    p,q=inputs;prior=deepcopy(q);e=s.effective(p,q,200,2,preset);assert q==prior
    assert e['pilot']==q['pilot'] and e['treatment_columns']==[0,2,4,6]
    if preset.startswith('outcome'):
        for family in ('rf','xgb'):assert e[family]['m']==q[family]['m']
        assert e['lasso']['m_alpha']==q['lasso']['m_alpha']
    elif preset.startswith('treatment'):
        for family in ('rf','xgb'):
            for role in ('g','ell'):assert e[family][role]==q[family][role]
        for name in ('g_alpha_linear','g_alpha_nonlinear','ell_alpha'):assert e['lasso'][name]==q['lasso'][name]


@pytest.mark.parametrize('k',[2,5])
@pytest.mark.parametrize('design',['figure1','figure2'])
@pytest.mark.parametrize('family',['lasso','rf','xgb'])
def test_training_response_isolation(inputs,k,design,family):
    p,q=inputs;c=s.config(p,design,'nonlinear',200,k);e=s.effective(p,q,200,k,'treatment_strong')
    data=s.draw(p,e,c,0)['moderate'];folds=s.folds_for(p,c,0);train=np.flatnonzero(folds!=0);hold=np.flatnonzero(folds==0)
    a=h.fit_training(e,c,family,0,data.x,data.y,data.d,train)
    y=data.y.copy();d=data.d.copy();y[hold]+=999;d[hold]-=999
    z=h.fit_training(e,c,family,0,data.x,y,d,train)
    np.testing.assert_array_equal(a.g,z.g);np.testing.assert_array_equal(a.m,z.m);assert a.record==z.record


@pytest.mark.parametrize('design',['figure1','figure2'])
@pytest.mark.parametrize('k',[2,5])
def test_run_bank_pairing(inputs,design,k):
    p,q=inputs;rows=s.run_replication(p,q,design,'nonlinear',200,k,'base',0)
    assert len(rows)==2*3*len(b.methods(design))
    assert len({r['fold_hash'] for r in rows})==1
    for level in p['levels']:
        for family in p['learners']:
            g={r['method']:r for r in rows if r['level']==level and r['learner']==family}
            cf=g['dml_crossfit'];left=g['naive_ml' if design=='figure1' else 'equal_train_reuse']
            assert cf['bank_records']==left['bank_records'] and cf['train_n_per_model']==200*(k-1)//k
            assert cf['prediction_overlap_n']==0
            if design=='figure1':assert cf['g_prediction_hash']==left['g_prediction_hash']
            else:assert left['prediction_overlap_n']==200
    assert all(r['finite'] and r['se']>0 for r in rows)


def test_full_and_rf_aliases(inputs):
    p,q=inputs;p['learners']=['rf']
    a=s.run_replication(p,q,'figure2','linear',200,2,'base',0)
    z=s.run_replication(p,q,'figure2','linear',200,5,'base',0)
    w=s.run_replication(p,q,'figure2','linear',200,2,'outcome_weak',0)
    for level in p['levels']:
        aa=next(r for r in a if r['level']==level and r['method']=='full_no_split')
        zz=next(r for r in z if r['level']==level and r['method']=='full_no_split')
        assert aa['bank_records']==zz['bank_records'] and aa['theta']==zz['theta']
        for method in b.methods('figure2'):
            aa=next(r for r in a if r['level']==level and r['method']==method)
            ww=next(r for r in w if r['level']==level and r['method']==method)
            assert aa['bank_records']==ww['bank_records'] and aa['theta']==ww['theta']
