"""IV-type experiments with ordinary and controlled nuisance learners."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import hashlib
import json

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from . import doubleml_reference as fair
from .iv_score import (IVRepairConfig, SeriesOLS, exact_series_basis,
    sigmoid, LassoNuisance, estimate_iv_type, estimate_outcome_mismatch)

METHODS = ("nonorthogonal_crossfit", "full_no_split", "equal_train_reuse", "crossfit")


def config_for(protocol, design, dgp, development=False):
    if protocol["artificial_pressure"] or protocol["k_folds"] != 2:
        raise ValueError("requires no pressure and two folds")
    return IVRepairConfig(design=design,dgp=dgp,n=protocol["n"],p=protocol["p"],
        reps=protocol["replications"][design],theta0=protocol["theta0"],
        seed=protocol["development_seed" if development else "seed"],
        n_estimators=protocol["n_estimators"])


def rich_basis(x):
    return np.column_stack((np.ones(len(x)), x, sigmoid(x)))


def joint_pilot(x, y, d, *, controlled=False):
    # Neither structural truth nor the true causal parameter is an input.
    basis = exact_series_basis(x) if controlled else rich_basis(x)
    design = np.column_stack((d, basis))
    coefficient, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    if rank != design.shape[1] or not np.isfinite(coefficient).all():
        raise FloatingPointError("rank-deficient/nonfinite training pilot")
    return float(coefficient[0])


def candidates(family):
    if family == "rf":
        return [dict(max_features="sqrt",max_depth=8,min_samples_leaf=2)] + [
            dict(max_features=1.,max_depth=depth,min_samples_leaf=leaf)
            for depth,leaf in [(5,2),(None,2),(None,5),(None,10),(None,20)]]
    if family == "xgb":
        return [dict(learning_rate=.01,max_depth=3)] + [
            dict(learning_rate=lr,max_depth=depth) for lr in (.05,.1) for depth in (2,3)]
    if family == "lasso":
        return [dict(alpha=a) for a in (.005,.02,.06,.1)]
    raise ValueError(family)


def model_for(config, family, params, seed):
    if family == "rf":
        return RandomForestRegressor(n_estimators=config.n_estimators,random_state=seed,n_jobs=1,**params)
    if family == "xgb":
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=config.n_estimators,random_state=seed,n_jobs=1,
            subsample=.9,colsample_bytree=.9,reg_lambda=1.,min_child_weight=2.,
            objective="reg:squarederror",verbosity=0,**params)
    if family == "lasso":
        return LassoNuisance(linear=config.dgp=="linear",alpha=params["alpha"],seed=seed)
    raise ValueError(family)


def controlled_outcome(config, learner, seed):
    if learner == "series_ols":
        return SeriesOLS()
    if learner.startswith("tree_leaf_"):
        return DecisionTreeRegressor(criterion="squared_error",max_depth=None,
            min_samples_leaf=int(learner.rsplit("_",1)[1]),random_state=seed)
    if learner == "rf_outcome_series_m":
        return RandomForestRegressor(n_estimators=config.n_estimators,max_features=1.,
            max_depth=None,min_samples_leaf=2,random_state=seed,n_jobs=1)
    raise ValueError(learner)


def selection_job(protocol, design, dgp, rep, family, role):
    """Score nuisance candidates by validation prediction loss.

    Pilots use training observations only. Causal-effect accuracy and
    ground-truth quantities are not selection criteria.
    """
    config = config_for(protocol,design,dgp,True)
    data = fair.draw(config,rep)
    outer = fair.make_fold_ids(config.n,2,fair.seed_for(config.seed,design,dgp,rep,"folds"))
    subset = np.flatnonzero(outer==0)
    inner = fair.make_fold_ids(len(subset),2,fair.seed_for(config.seed,design,dgp,rep,"selection"))
    losses = np.zeros(len(candidates(family)))
    for fold in (0,1):
        train,valid=subset[inner!=fold],subset[inner==fold]
        pilot=joint_pilot(data.x[train],data.y[train],data.d[train]) if role=="g" else None
        target=data.d if role=="m" else data.y-pilot*data.d
        for index,params in enumerate(candidates(family)):
            seed=fair.seed_for(config.seed,design,dgp,rep,family,role,"selection_model")
            model=model_for(config,family,params,seed)
            model.fit(data.x[train],target[train])
            losses[index]+=float(np.sum((model.predict(data.x[valid])-target[valid])**2))
    return dict(design=design,dgp=dgp,replication=rep,family=family,role=role,
        data_seed=data.data_seed,validation_n=len(subset),squared_errors=losses.tolist(),
        candidates=candidates(family),selection_uses="observed_validation_prediction_loss_only")


@dataclass(frozen=True)
class Predictions:
    g: np.ndarray
    m: np.ndarray
    ell: np.ndarray | None
    pilot: float
    record: dict[str,Any]


def fit_training(config, track, learner, params, rep, x, y, d, train):
    pilot=joint_pilot(x[train],y[train],d[train],controlled=track=="controlled")
    seed=lambda role: fair.seed_for(config.seed,config.design,config.dgp,rep,track,learner,role)
    if track=="controlled":
        m_model=SeriesOLS().fit(x[train],d[train])
        ell_model=controlled_outcome(config,learner,seed("ell")).fit(x[train],y[train])
        m=m_model.predict(x); ell=ell_model.predict(x)
        g=ell-pilot*m
        target_hash=fair.vector_hash(y[train])
        construction="ell_hat_minus_training_pilot_times_m_hat"
    elif track=="ordinary":
        m_model=model_for(config,learner,params["m"],seed("m")).fit(x[train],d[train])
        target=y[train]-pilot*d[train]
        g_model=model_for(config,learner,params["g"],seed("g")).fit(x[train],target)
        m=m_model.predict(x); g=g_model.predict(x); ell=None
        target_hash=fair.vector_hash(target)
        construction="direct_Y_minus_training_pilot_D"
    else: raise ValueError(track)
    if not np.isfinite(g).all() or not np.isfinite(m).all():
        raise FloatingPointError("nonfinite predictions")
    record=dict(train_n=len(train),training_indices_sha256=fair.vector_hash(train),
        pilot=pilot,target_sha256=target_hash,construction=construction,
        g_all_sha256=fair.vector_hash(g),m_all_sha256=fair.vector_hash(m))
    return Predictions(g,m,ell,pilot,record)


def assemble(bank, folds, crossfit):
    source=1-folds if crossfit else folds
    index=np.arange(len(folds))
    g=np.vstack([b.g for b in bank])[source,index]
    m=np.vstack([b.m for b in bank])[source,index]
    ell=None if bank[0].ell is None else np.vstack([b.ell for b in bank])[source,index]
    return g,m,ell,source


def summarize_estimate(data, g, m, result):
    if not result.finite: return dict(finite=False,failure_reason=result.failure_reason)
    return dict(finite=True,failure_reason="",theta=result.theta,se=result.se,
        score_mean=result.score_mean,jacobian=result.q_hat,
        g_prediction_hash=fair.vector_hash(g),m_prediction_hash=fair.vector_hash(m))


def run_replication(protocol, selections, design, dgp, rep, *, development=False):
    config=config_for(protocol,design,dgp,development)
    data=fair.draw(config,rep)
    folds=fair.make_fold_ids(config.n,2,fair.seed_for(config.seed,design,dgp,rep,"folds"))
    controls=["series_ols"] if design=="figure1" else ["series_ols"]+[
        f"tree_leaf_{leaf}" for leaf in protocol["tree_leaves"]]+["rf_outcome_series_m"]
    specifications=[("ordinary",l) for l in protocol["ordinary_families"]]+[("controlled",l) for l in controls]
    rows=[]
    for track,learner in specifications:
        params=selections[f"{design}/{dgp}/{learner}"] if track=="ordinary" else {}
        base=dict(protocol_id=protocol["protocol_id"],development=development,design=design,
            dgp=dgp,replication=rep,n=config.n,p=config.p,theta0=config.theta0,
            track=track,learner=learner,data_seed=data.data_seed,fold_hash=fair.fold_digest(folds),
            artificial_pressure=False,selected_params=params,
            treatment_learner=learner if track=="ordinary" else "series_ols_correctly_specified",
            pilot_mode=protocol["pilot" if track=="ordinary" else "controlled_pilot"])
        try:
            bank=tuple(fit_training(config,track,learner,params,rep,data.x,data.y,data.d,
                np.flatnonzero(folds==half)) for half in (0,1))
            records=[b.record for b in bank]
            identity=hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest()
            full=fit_training(config,track,learner,params,rep,data.x,data.y,data.d,np.arange(config.n))
            for method in METHODS+( ("outcome_mismatch_crossfit",) if track=="controlled" and design=="figure1" else () ):
                row={**base,"method":method}
                if method=="full_no_split":
                    g,m,ell=full.g,full.m,full.ell
                    row.update(train_n_per_model=config.n,prediction_overlap_n=config.n,
                        bank_records=[full.record],bank_identity=hashlib.sha256(json.dumps([full.record],sort_keys=True).encode()).hexdigest())
                else:
                    g,m,ell,source=assemble(bank,folds,method!="equal_train_reuse")
                    row.update(train_n_per_model=config.n//2,
                        prediction_overlap_n=config.n if method=="equal_train_reuse" else 0,
                        bank_records=records,bank_identity=identity,source_half_sha256=fair.vector_hash(source))
                if method=="nonorthogonal_crossfit": result=estimate_outcome_mismatch(data.y,data.d,g)
                elif method=="outcome_mismatch_crossfit": result=estimate_outcome_mismatch(data.y,data.d,ell)
                else: result=estimate_iv_type(data.y,data.d,g,m)
                row.update(summarize_estimate(data,g,m,result))
                if result.finite:
                    error=result.theta-config.theta0
                    a,b=g-data.g0,m-data.m0
                    row.update(bias=error,abs_error=abs(error),squared_error=error**2,
                        covers=abs(error)<=1.96*result.se,nuisance_rmse_g=float(np.sqrt(np.mean(a*a))),
                        nuisance_rmse_m=float(np.sqrt(np.mean(b*b))))
                    if method in ("full_no_split","equal_train_reuse","crossfit"):
                        components=[float(np.mean(data.v*data.u)),float(np.mean(-data.v*a-b*data.u)),float(np.mean(b*a))]
                        if abs(sum(components)/result.q_hat-error)>1e-10:
                            raise RuntimeError("exact IV decomposition failure")
                        row.update(weighted_oracle=components[0]/result.q_hat,
                            weighted_first_order=components[1]/result.q_hat,
                            weighted_second_order=components[2]/result.q_hat)
                rows.append(row)
        except (FloatingPointError,ValueError,np.linalg.LinAlgError) as exc:
            expected=list(METHODS)+( ["outcome_mismatch_crossfit"] if track=="controlled" and design=="figure1" else [] )
            # Preserve every method, not partial successful output.
            rows=[r for r in rows if (r["track"],r["learner"])!=(track,learner)]
            rows.extend({**base,"method":m,"finite":False,"failure_reason":repr(exc)} for m in expected)
    return rows
