"""IV sensitivity: matched draws, K-bank reuse, frozen role-specific changes."""
from __future__ import annotations
from copy import deepcopy
import numpy as np
from . import harder_v5 as hard, two_claims_v4 as base, fair_iv as fair
from .iv_thesis_repair import IVRepairConfig, PLRData, toeplitz_cov, estimate_iv_type, estimate_outcome_mismatch


def scenarios(p):
    return [(n,k,'base') for n in p['sample_sizes'] for k in p['fold_counts']] + [
        (p['anchor_n'],p['anchor_k'],s) for s in p['presets'] if s!='base']


def effective(p, original, n, k, preset):
    q=deepcopy(original);q.update(n=n,k_folds=k,seed=p['seed'],levels=p['levels'],learners=p['learners'])
    change=p['preset_overrides'][preset]
    for name,role in [('rf_g_leaf','g'),('rf_ell_leaf','ell'),('rf_m_leaf','m')]:
        if name in change:q['rf'][role]['min_samples_leaf']=change[name]
    for name,role in [('xgb_g_depth','g'),('xgb_ell_depth','ell')]:
        if name in change:q['xgb'][role]['max_depth']=change[name]
    if 'xgb_m_lambda' in change:q['xgb']['m']['reg_lambda']=change['xgb_m_lambda']
    if 'lasso_m_alpha_multiplier' in change:q['lasso']['m_alpha']*=change['lasso_m_alpha_multiplier']
    if 'lasso_outcome_alpha_multiplier' in change:
        for name in ('g_alpha_linear','g_alpha_nonlinear','ell_alpha'):q['lasso'][name]*=change['lasso_outcome_alpha_multiplier']
    return q


def config(p,design,dgp,n,k):
    return IVRepairConfig(design=design,dgp=dgp,n=n,p=20,k_folds=k,theta0=.5,reps=p['replications'],seed=p['seed'])


def draw(p,q,c,rep):
    seed=fair.seed_for(p['seed'],c.design,c.dgp,rep,'sensitivity-v6-nested-data')
    rng=np.random.default_rng(seed);max_n=max(p['sample_sizes'])
    x=rng.multivariate_normal(np.zeros(c.p),toeplitz_cov(c.p),size=max_n)[:c.n]
    u=rng.normal(size=max_n)[:c.n];v=rng.normal(size=max_n)[:c.n]
    pair={}
    for level in p['levels']:
        g,m,ell=hard.truth(q,c,x,level);d=m+v
        pair[level]=PLRData(x,c.theta0*d+g+u,d,g,m,ell,u,v,seed)
    return pair


def folds_for(p,c,rep):
    return fair.make_fold_ids(c.n,c.k_folds,fair.seed_for(p['seed'],c.design,c.dgp,c.n,rep,'sensitivity-v6-common-permutation'))


def assemble(bank,folds,crossfit):
    source=folds if crossfit else (folds+1)%len(bank);index=np.arange(len(folds))
    return tuple(np.vstack([getattr(b,role) for b in bank])[source,index] for role in ('g','m'))


def run_replication(p,original,design,dgp,n,k,preset,rep):
    q=effective(p,original,n,k,preset);c=config(p,design,dgp,n,k)
    pair=draw(p,q,c,rep);folds=folds_for(p,c,rep);rows=[]
    common=dict(n=n,k_folds=k,preset=preset,p=20,theta0=.5,stage=p['stage'],protocol_id=p['protocol_id'],
        design=design,dgp=dgp,replication=rep,fold_hash=fair.fold_digest(folds),artificial_pressure=False,
        controlled_known_treatment_support=True,train_n_per_model=n*(k-1)//k,
        bank_rule='model j excludes fold j; reuse source=(fold+1)%K; one prediction per row')
    for level,data in pair.items():
        hashes={role+'_hash':fair.vector_hash(getattr(data,role)) for role in ('x','u','v')}
        for family in p['learners']:
            bank=tuple(hard.fit_training(q,c,family,rep,data.x,data.y,data.d,np.flatnonzero(folds!=j)) for j in range(k))
            records=[b.record for b in bank];cf_g,cf_m=assemble(bank,folds,True)
            full=None if design=='figure1' else hard.fit_training(q,c,family,rep,data.x,data.y,data.d,np.arange(n))
            for method in base.methods(design):
                if method=='full_no_split':g,m=full.g,full.m;used=[full.record];overlap=n;train_n=n
                else:
                    g,m=(cf_g,cf_m) if method!='equal_train_reuse' else assemble(bank,folds,False)
                    used=records;overlap=n if method=='equal_train_reuse' else 0;train_n=n*(k-1)//k
                estimate=estimate_outcome_mismatch(data.y,data.d,g) if method=='naive_ml' else estimate_iv_type(data.y,data.d,g,m)
                row=dict(common,level=level,learner=family,treatment_learner=family,outcome_learner=family,method=method,
                    data_seed=data.data_seed,bank_records=used,bank_identity=base.bank_hash(used),
                    prediction_overlap_n=overlap,finite=estimate.finite,failure_reason=estimate.failure_reason,**hashes)
                row['train_n_per_model']=train_n
                if estimate.finite:
                    error=estimate.theta-.5;a=g-data.g0;b=m-data.m0
                    row.update(theta=estimate.theta,se=estimate.se,score_mean=estimate.score_mean,jacobian=estimate.q_hat,
                        bias=error,abs_error=abs(error),squared_error=error*error,covers=abs(error)<=1.96*estimate.se,
                        g_prediction_hash=fair.vector_hash(g),m_prediction_hash=fair.vector_hash(m),
                        nuisance_rmse_g=float(np.sqrt(np.mean(a*a))),nuisance_rmse_m=float(np.sqrt(np.mean(b*b))))
                    if method!='naive_ml':
                        components=np.array([np.mean(data.v*data.u),np.mean(-data.v*a-b*data.u),np.mean(b*a)])/estimate.q_hat
                        if abs(components.sum()-error)>1e-10:raise RuntimeError('IV decomposition mismatch')
                        row.update(zip(('weighted_oracle','weighted_first_order','weighted_second_order'),map(float,components)))
                rows.append(row)
    return rows
