"""IV-type comparisons with learned nuisance functions.

Adapt the pinned official DoubleML pooled-pilot algorithm to the original
thesis DGPs/learner settings. Only the final structural IV-type score is DML.
The preliminary residual ratio is internal and not a reported estimator.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from .iv_thesis_repair import (
    IVRepairConfig, PLRData, Estimate, make_learner, make_fold_ids,
    fold_digest, nuisance_truth, toeplitz_cov, estimate_iv_type,
    estimate_outcome_mismatch,
)

METHODS = ("nonorthogonal_crossfit", "iv_no_split", "iv_crossfit")


def seed_for(base: int, *labels: object) -> int:
    payload = "\0".join(map(str, ("fair-iv-v1", base, *labels))).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def cell_config(protocol: dict[str, Any], design: str, dgp: str, *, development: bool = False) -> IVRepairConfig:
    if design not in ("figure1", "figure2") or dgp not in ("linear", "nonlinear"):
        raise ValueError("invalid cell")
    if protocol["artificial_pressure"] is not False:
        raise ValueError("artificial pressure is forbidden")
    if tuple(protocol["methods"]) != METHODS or protocol["pilot"] != "official_pooled_residual_ratio":
        raise ValueError("unexpected method/pilot contract")
    return IVRepairConfig(
        design=design, dgp=dgp, n=protocol["n"], p=protocol["p"],
        reps=protocol["replications"][design], theta0=protocol["theta0"],
        seed=protocol["development_seed" if development else "seed"],
        k_folds=protocol["k_folds"], learners=tuple(protocol["learners"]),
        n_estimators=protocol["n_estimators"], rf_max_depth=protocol["rf_max_depth"],
        rf_min_samples_leaf=protocol["rf_min_samples_leaf"],
        xgb_learning_rate=protocol["xgb_learning_rate"],
        lasso_alpha_linear=protocol["lasso_alpha_linear"][design],
        lasso_alpha_nonlinear=protocol["lasso_alpha_nonlinear"],
    )


def draw(config: IVRepairConfig, rep: int) -> PLRData:
    config.validate()
    data_seed = seed_for(config.seed, config.design, config.dgp, config.n, config.p, rep, "data")
    rng = np.random.default_rng(data_seed)
    x = rng.multivariate_normal(np.zeros(config.p), toeplitz_cov(config.p), size=config.n)
    u, v = rng.normal(size=config.n), rng.normal(size=config.n)
    g, m, ell = nuisance_truth(config.design, config.dgp, x, config.theta0)
    d = m + v
    return PLRData(x, config.theta0*d+g+u, d, g, m, ell, u, v, data_seed)


def preliminary_theta(y: np.ndarray, d: np.ndarray, ell: np.ndarray, m: np.ndarray) -> float:
    residual = d-m
    denominator = float(np.mean(residual**2))
    if denominator <= 1e-12 or not math.isfinite(denominator):
        raise FloatingPointError("degenerate pilot residual")
    theta = float(np.mean(residual*(y-ell))/denominator)
    if not math.isfinite(theta):
        raise FloatingPointError("nonfinite pilot")
    return theta


@dataclass(frozen=True)
class Predictions:
    ell: np.ndarray
    m: np.ndarray
    g: np.ndarray
    pilot: float


def predict_target(config: IVRepairConfig, learner: str, rep: int, data: PLRData,
                   target: np.ndarray, folds: np.ndarray | None, role: str) -> np.ndarray:
    # A clone-like fixed random state is used in each fold, as in the official
    # implementation. Use identical role seeds across full/cross-fit schemes.
    model_seed = seed_for(config.seed, config.design, config.dgp, rep, learner, role)
    result = np.full(config.n, np.nan)
    splits = [(np.arange(config.n), np.arange(config.n))] if folds is None else [
        (np.flatnonzero(folds != fold), np.flatnonzero(folds == fold))
        for fold in range(config.k_folds)
    ]
    coverage = np.zeros(config.n, dtype=int)
    for train, test in splits:
        if folds is not None and np.intersect1d(train, test).size:
            raise RuntimeError("train-test overlap")
        model = make_learner(config, learner, model_seed)
        model.fit(data.x[train], target[train])
        result[test] = model.predict(data.x[test])
        coverage[test] += 1
    if not np.all(coverage == 1) or not np.isfinite(result).all():
        raise FloatingPointError("incomplete/nonfinite nuisance predictions")
    return result


def official_predictions(config: IVRepairConfig, learner: str, rep: int,
                         data: PLRData, folds: np.ndarray | None) -> Predictions:
    ell = predict_target(config, learner, rep, data, data.y, folds, "ell")
    m = predict_target(config, learner, rep, data, data.d, folds, "m")
    pilot = preliminary_theta(data.y, data.d, ell, m)
    g = predict_target(config, learner, rep, data, data.y-pilot*data.d, folds, "g")
    return Predictions(ell, m, g, pilot)


def vector_hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest()


def run_replication(protocol: dict[str, Any], design: str, dgp: str, rep: int,
                    *, development: bool = False) -> list[dict[str, Any]]:
    config = cell_config(protocol, design, dgp, development=development)
    if not 0 <= rep < config.reps:
        raise ValueError("replication outside fixed universe")
    data = draw(config, rep)
    folds = make_fold_ids(config.n, config.k_folds,
                         seed_for(config.seed, design, dgp, rep, "folds"))
    rows: list[dict[str, Any]] = []
    for learner in config.learners:
        estimates: dict[str, tuple[Estimate, Predictions] | str] = {}
        for mode in ("no_split", "crossfit"):
            try:
                pred = official_predictions(config, learner, rep, data, None if mode == "no_split" else folds)
                estimates[f"iv_{mode}"] = (estimate_iv_type(data.y, data.d, pred.g, pred.m), pred)
                if mode == "crossfit":
                    # Name of imported ratio helper is historical; here its
                    # input is STRUCTURAL g, so this is the valid naive score,
                    # NOT the old ell/raw-D mismatch diagnostic.
                    estimates["nonorthogonal_crossfit"] = (
                        estimate_outcome_mismatch(data.y, data.d, pred.g), pred)
            except (FloatingPointError, ValueError) as exc:
                estimates[f"iv_{mode}"] = f"{type(exc).__name__}: {exc}"
                if mode == "crossfit":
                    estimates["nonorthogonal_crossfit"] = estimates["iv_crossfit"]
        for method in METHODS:
            item = estimates[method]
            row: dict[str, Any] = dict(
                protocol_id=protocol["protocol_id"], development=development,
                design=design, dgp=dgp, n=config.n, p=config.p,
                replication=rep, theta0=config.theta0, learner=learner, method=method,
                data_seed=data.data_seed, fold_hash=fold_digest(folds),
                artificial_pressure=False, treatment_learner=learner,
                pilot_mode="official_pooled_residual_ratio",
            )
            if isinstance(item, str):
                row.update(finite=False, failure_reason=item)
            else:
                estimate, pred = item
                row.update(finite=estimate.finite, failure_reason=estimate.failure_reason,
                           pilot=pred.pilot, g_prediction_hash=vector_hash(pred.g),
                           m_prediction_hash=vector_hash(pred.m))
                if estimate.finite:
                    error = estimate.theta-config.theta0
                    row.update(theta=estimate.theta, se=estimate.se, bias=error,
                               abs_error=abs(error), squared_error=error**2,
                               covers=abs(error) <= 1.96*estimate.se,
                               score_mean=estimate.score_mean, jacobian=estimate.q_hat,
                               nuisance_rmse_g=float(np.sqrt(np.mean((pred.g-data.g0)**2))),
                               nuisance_rmse_m=float(np.sqrt(np.mean((pred.m-data.m0)**2))))
            rows.append(row)
    return rows
