"""Three-family controlled IV experiments; truth never enters fitting."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from . import fair_iv as fair
from .iv_thesis_repair import IVRepairConfig, sigmoid, estimate_iv_type, estimate_outcome_mismatch


def config_for(protocol, design, dgp, preflight=False):
    if protocol['artificial_pressure'] or protocol['k_folds'] != 2:
        raise ValueError('no artificial pressure; exactly two folds')
    return IVRepairConfig(design=design,dgp=dgp,n=protocol['n'],p=protocol['p'],
        reps=protocol['replications'][design],theta0=protocol['theta0'],
        seed=protocol['preflight_seed' if preflight else 'seed'])


def features(x, family, role, dgp):
    if role == 'm':
        raw=x[:,[0,2]]
        return np.column_stack((raw,sigmoid(raw))) if family=='lasso' and dgp=='nonlinear' else raw
    if family=='lasso' and (role=='ell' or dgp=='nonlinear'):
        return np.column_stack((x,sigmoid(x)))
    return x


def lasso(alpha, positive=False):
    return make_pipeline(StandardScaler(),Lasso(alpha=alpha,positive=positive,
        max_iter=200000,tol=1e-7,selection='cyclic'))


def model_for(protocol,family,role,dgp,seed):
    if family=='rf':
        return RandomForestRegressor(random_state=seed,n_jobs=1,**protocol['rf'][role])
    if family=='xgb':
        from xgboost import XGBRegressor
        return XGBRegressor(random_state=seed,n_jobs=1,objective='reg:squarederror',
            tree_method='hist',max_bin=512,verbosity=0,**protocol['xgb'][role])
    if family=='lasso':
        key='g_alpha_'+dgp if role=='g' else role+'_alpha'
        return lasso(protocol['lasso'][key],positive=role=='m')
    raise ValueError(family)


def joint_pilot(protocol,x,y,d):
    z=np.column_stack((d,x,sigmoid(x)))
    fitted=lasso(protocol['pilot']['alpha']).fit(z,y)
    pilot=float(fitted.named_steps['lasso'].coef_[0]/fitted.named_steps['standardscaler'].scale_[0])
    if not np.isfinite(pilot): raise FloatingPointError('nonfinite pilot')
    return pilot


@dataclass(frozen=True)
class Predictions:
    g: np.ndarray
    m: np.ndarray
    record: dict


def fit_training(protocol,config,family,rep,x,y,d,train):
    # This API has no causal truth or nuisance truth argument.
    pilot=joint_pilot(protocol,x[train],y[train],d[train])
    role='g' if config.design=='figure1' else 'ell'
    target=y[train]-pilot*d[train] if role=='g' else y[train]
    predictions={}
    for channel,response in (('m',d[train]),(role,target)):
        z=features(x,family,channel,config.dgp)
        seed=fair.seed_for(config.seed,config.design,config.dgp,rep,family,channel,'v4')
        fitted=model_for(protocol,family,channel,config.dgp,seed).fit(z[train],response)
        predictions[channel]=fitted.predict(z)
    m=predictions['m']; g=predictions[role] if role=='g' else predictions['ell']-pilot*m
    if not np.isfinite(g).all() or not np.isfinite(m).all():
        raise FloatingPointError('nonfinite predictions')
    record=dict(train_n=len(train),training_indices_sha256=fair.vector_hash(train),
        pilot=pilot,pilot_family='lasso',target_sha256=fair.vector_hash(target),
        construction='direct_training_Y_minus_pilot_D' if role=='g' else 'training_ell_minus_pilot_m',
        treatment_family=family,outcome_family=family,
        g_all_sha256=fair.vector_hash(g),m_all_sha256=fair.vector_hash(m))
    return Predictions(g,m,record)


def assemble(bank,folds,crossfit):
    source=1-folds if crossfit else folds; index=np.arange(len(folds))
    return tuple(np.vstack([getattr(b,role) for b in bank])[source,index] for role in ('g','m'))


def bank_hash(records):
    return hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest()


def methods(design):
    return ('naive_ml','dml_crossfit') if design=='figure1' else ('full_no_split','equal_train_reuse','dml_crossfit')


def run_replication(protocol,design,dgp,rep):
    config=config_for(protocol,design,dgp); data=fair.draw(config,rep)
    folds=fair.make_fold_ids(config.n,2,fair.seed_for(config.seed,design,dgp,rep,'v4folds'))
    rows=[]
    for family in protocol['learners']:
        bank=tuple(fit_training(protocol,config,family,rep,data.x,data.y,data.d,
            np.flatnonzero(folds==half)) for half in (0,1))
        records=[b.record for b in bank]
        cf_g,cf_m=assemble(bank,folds,True)
        full=None if design=='figure1' else fit_training(protocol,config,family,rep,
            data.x,data.y,data.d,np.arange(config.n))
        for method in methods(design):
            if method=='full_no_split':
                g,m=full.g,full.m; used=[full.record]; overlap=config.n; train_n=config.n
            else:
                g,m=(cf_g,cf_m) if method!='equal_train_reuse' else assemble(bank,folds,False)
                used=records; overlap=config.n if method=='equal_train_reuse' else 0; train_n=config.n//2
            # The historically named helper computes the genuine naive ratio
            # here because its input is STRUCTURAL g, never outcome ell.
            result=estimate_outcome_mismatch(data.y,data.d,g) if method=='naive_ml' else estimate_iv_type(data.y,data.d,g,m)
            row=dict(protocol_id=protocol['protocol_id'],design=design,dgp=dgp,replication=rep,
                learner=family,treatment_learner=family,outcome_learner=family,method=method,
                n=config.n,p=config.p,theta0=config.theta0,data_seed=data.data_seed,
                fold_hash=fair.fold_digest(folds),train_n_per_model=train_n,
                prediction_overlap_n=overlap,artificial_pressure=False,
                controlled_known_treatment_support=True,bank_records=used,bank_identity=bank_hash(used),
                finite=result.finite,failure_reason=result.failure_reason)
            if result.finite:
                error=result.theta-config.theta0; a=g-data.g0; b=m-data.m0
                row.update(theta=result.theta,se=result.se,score_mean=result.score_mean,jacobian=result.q_hat,
                    bias=error,abs_error=abs(error),squared_error=error**2,covers=abs(error)<=1.96*result.se,
                    g_prediction_hash=fair.vector_hash(g),m_prediction_hash=fair.vector_hash(m),
                    nuisance_rmse_g=float(np.sqrt(np.mean(a*a))),nuisance_rmse_m=float(np.sqrt(np.mean(b*b))))
                if method!='naive_ml':
                    components=np.array([np.mean(data.v*data.u),np.mean(-data.v*a-b*data.u),np.mean(b*a)])/result.q_hat
                    if abs(components.sum()-error)>1e-10: raise RuntimeError('IV decomposition failed')
                    row.update(weighted_oracle=float(components[0]),weighted_first_order=float(components[1]),
                        weighted_second_order=float(components[2]))
            rows.append(row)
    return rows
