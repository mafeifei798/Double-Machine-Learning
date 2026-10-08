"""Matched moderate DGP expansion; no oracle information in fitting."""
from __future__ import annotations
import numpy as np
from . import two_claims_v4 as base, fair_iv as fair
from .iv_thesis_repair import PLRData, nuisance_truth, toeplitz_cov, sigmoid, estimate_iv_type, estimate_outcome_mismatch


def truth(protocol,config,x,level):
    g,m,_=nuisance_truth(config.design,config.dgp,x,config.theta0)
    if level=='moderate':
        difficulty=protocol['difficulty']
        if config.dgp=='linear':
            cm=difficulty['linear_m_raw_additions'];cg=difficulty['linear_g_raw_additions']
            m=m+cm[0]*x[:,4]+cm[1]*x[:,6];g=g+cg[0]*x[:,4]+cg[1]*x[:,6]
        else:
            cm=difficulty['nonlinear_m_raw4_sigmoid6_additions'];cg=difficulty['nonlinear_g_sigmoid4_raw6_additions']
            m=m+cm[0]*x[:,4]+cm[1]*sigmoid(x[:,6])
            g=g+cg[0]*sigmoid(x[:,4])+cg[1]*x[:,6]+difficulty['nonlinear_g_tanh0_times_tanh4']*np.tanh(x[:,0])*np.tanh(x[:,4])
    elif level!='baseline':raise ValueError(level)
    return g,m,g+config.theta0*m


def draw_pair(protocol,config,rep):
    seed=fair.seed_for(config.seed,config.design,config.dgp,config.n,config.p,rep,'harder-v5-data')
    rng=np.random.default_rng(seed)
    x=rng.multivariate_normal(np.zeros(config.p),toeplitz_cov(config.p),size=config.n)
    u,v=rng.normal(size=config.n),rng.normal(size=config.n)
    pair={}
    for level in protocol['levels']:
        g,m,ell=truth(protocol,config,x,level);d=m+v
        pair[level]=PLRData(x,config.theta0*d+g+u,d,g,m,ell,u,v,seed)
    return pair


def features(protocol,x,family,role,dgp):
    if role!='m':return base.features(x,family,role,dgp)
    raw=x[:,protocol['treatment_columns']]
    return np.column_stack((raw,sigmoid(raw))) if family=='lasso' and dgp=='nonlinear' else raw


def fit_training(protocol,config,family,rep,x,y,d,train):
    pilot=base.joint_pilot(protocol,x[train],y[train],d[train])
    role='g' if config.design=='figure1' else 'ell'
    target=y[train]-pilot*d[train] if role=='g' else y[train]
    predictions={}
    for channel,response in (('m',d[train]),(role,target)):
        z=features(protocol,x,family,channel,config.dgp)
        seed=fair.seed_for(config.seed,config.design,config.dgp,rep,family,channel,'harder-v5')
        fitted=base.model_for(protocol,family,channel,config.dgp,seed).fit(z[train],response)
        predictions[channel]=fitted.predict(z)
    m=predictions['m'];g=predictions[role] if role=='g' else predictions['ell']-pilot*m
    if not np.isfinite(g).all() or not np.isfinite(m).all():raise FloatingPointError('nonfinite predictions')
    record=dict(train_n=len(train),training_indices_sha256=fair.vector_hash(train),pilot=pilot,pilot_family='lasso',
        target_sha256=fair.vector_hash(target),construction='direct_training_Y_minus_pilot_D' if role=='g' else 'training_ell_minus_pilot_m',
        treatment_family=family,outcome_family=family,treatment_columns=protocol['treatment_columns'],
        g_all_sha256=fair.vector_hash(g),m_all_sha256=fair.vector_hash(m))
    return base.Predictions(g,m,record)


def run_replication(protocol,design,dgp,rep):
    config=base.config_for(protocol,design,dgp)
    pair=draw_pair(protocol,config,rep)
    folds=fair.make_fold_ids(config.n,2,fair.seed_for(config.seed,design,dgp,rep,'harder-v5-folds'))
    rows=[]
    for level,data in pair.items():
        common=dict(x_hash=fair.vector_hash(data.x),u_hash=fair.vector_hash(data.u),v_hash=fair.vector_hash(data.v))
        for family in protocol['learners']:
            bank=tuple(fit_training(protocol,config,family,rep,data.x,data.y,data.d,
                np.flatnonzero(folds==half)) for half in (0,1))
            records=[b.record for b in bank];cf_g,cf_m=base.assemble(bank,folds,True)
            full=None if design=='figure1' else fit_training(protocol,config,family,rep,data.x,data.y,data.d,np.arange(config.n))
            for method in base.methods(design):
                if method=='full_no_split':g,m=full.g,full.m;used=[full.record];overlap=config.n;train_n=config.n
                else:
                    g,m=(cf_g,cf_m) if method!='equal_train_reuse' else base.assemble(bank,folds,False)
                    used=records;overlap=config.n if method=='equal_train_reuse' else 0;train_n=config.n//2
                result=estimate_outcome_mismatch(data.y,data.d,g) if method=='naive_ml' else estimate_iv_type(data.y,data.d,g,m)
                row=dict(protocol_id=protocol['protocol_id'],stage=protocol['stage'],level=level,design=design,dgp=dgp,
                    replication=rep,learner=family,treatment_learner=family,outcome_learner=family,n=config.n,p=config.p,
                    method=method,theta0=config.theta0,data_seed=data.data_seed,fold_hash=fair.fold_digest(folds),
                    train_n_per_model=train_n,prediction_overlap_n=overlap,artificial_pressure=False,
                    controlled_known_treatment_support=True,bank_records=used,bank_identity=base.bank_hash(used),
                    finite=result.finite,failure_reason=result.failure_reason,**common)
                if result.finite:
                    error=result.theta-config.theta0;a=g-data.g0;b=m-data.m0
                    row.update(theta=result.theta,se=result.se,score_mean=result.score_mean,jacobian=result.q_hat,
                        bias=error,abs_error=abs(error),squared_error=error**2,covers=abs(error)<=1.96*result.se,
                        g_prediction_hash=fair.vector_hash(g),m_prediction_hash=fair.vector_hash(m),
                        nuisance_rmse_g=float(np.sqrt(np.mean(a*a))),nuisance_rmse_m=float(np.sqrt(np.mean(b*b))),
                        true_g_sd=float(np.std(data.g0,ddof=1)),true_m_sd=float(np.std(data.m0,ddof=1)))
                    if method!='naive_ml':
                        components=np.array([np.mean(data.v*data.u),np.mean(-data.v*a-b*data.u),np.mean(b*a)])/result.q_hat
                        if abs(components.sum()-error)>1e-10:raise RuntimeError('IV decomposition failed')
                        row.update(weighted_oracle=float(components[0]),weighted_first_order=float(components[1]),weighted_second_order=float(components[2]))
                rows.append(row)
    return rows
