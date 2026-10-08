"""Shared model-bank IV diagnostic; training size and fitted models are paired."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any

import numpy as np

from . import fair_iv as fair
from .iv_thesis_repair import make_learner, estimate_iv_type

METHODS = ("equal_train_reuse", "equal_train_crossfit")


def config_for(protocol, design, dgp, development=False):
    if tuple(protocol["methods"]) != METHODS or protocol["pilot"] != "shared_bank_training_half_residual_ratio":
        raise ValueError("wrong equal-training method/pilot")
    if protocol["n"] % 2 or protocol["k_folds"] != 2 or protocol["data_namespace"] != "fair-iv-v1":
        raise ValueError("requires even n and original two-fold data namespace")
    adapter = {**protocol, "methods": list(fair.METHODS), "pilot": "official_pooled_residual_ratio"}
    return fair.cell_config(adapter, design, dgp, development=development)


@dataclass(frozen=True)
class Bank:
    g: np.ndarray  # training-half index by evaluated observation
    m: np.ndarray
    pilots: tuple[float, float]
    records: tuple[dict, dict]
    identity: str


def fit_bank(config, learner, rep, data, folds):
    g = np.empty((2, config.n)); m = np.empty_like(g)
    pilots = []; records = []
    for train_half in range(2):
        train = np.flatnonzero(folds == train_half)
        if len(train) != config.n // 2:
            raise ValueError("unbalanced training halves")
        def fit(role, target):
            seed = fair.seed_for(config.seed, config.design, config.dgp, rep, learner, role)
            model = make_learner(config, learner, seed)
            model.fit(data.x[train], target)
            return model
        ell_model = fit("ell", data.y[train])
        m_model = fit("m", data.d[train])
        ell_train = ell_model.predict(data.x[train])
        m[train_half] = m_model.predict(data.x)
        pilot = fair.preliminary_theta(data.y[train], data.d[train], ell_train, m[train_half, train])
        g_model = fit("g", data.y[train] - pilot * data.d[train])
        g[train_half] = g_model.predict(data.x)
        if not np.isfinite(g[train_half]).all() or not np.isfinite(m[train_half]).all():
            raise FloatingPointError("nonfinite bank predictions")
        pilots.append(pilot)
        records.append(dict(training_half=train_half, train_n=len(train),
                            training_indices_sha256=fair.vector_hash(train), pilot=pilot,
                            structural_target_sha256=fair.vector_hash(data.y[train]-pilot*data.d[train]),
                            g_all_sha256=fair.vector_hash(g[train_half]),
                            m_all_sha256=fair.vector_hash(m[train_half])))
    identity = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    return Bank(g, m, tuple(pilots), tuple(records), identity)


def assemble(bank, folds, crossfit):
    source_half = 1-folds if crossfit else folds
    indices = np.arange(len(folds))
    return bank.g[source_half, indices], bank.m[source_half, indices], source_half


def run_replication(protocol: dict[str, Any], design: str, dgp: str, rep: int, *, development=False):
    config = config_for(protocol, design, dgp, development)
    if not 0 <= rep < config.reps:
        raise ValueError("replication outside universe")
    data = fair.draw(config, rep)
    folds = fair.make_fold_ids(config.n, 2, fair.seed_for(config.seed, design, dgp, rep, "folds"))
    rows = []
    for learner in config.learners:
        bank = None; failure = None
        try:
            bank = fit_bank(config, learner, rep, data, folds)
        except (FloatingPointError, ValueError) as exc:
            failure = f"{type(exc).__name__}: {exc}"
        for method in METHODS:
            crossfit = method == "equal_train_crossfit"
            row = dict(protocol_id=protocol["protocol_id"], development=development, design=design,
                       dgp=dgp, n=config.n, p=config.p, theta0=config.theta0, replication=rep,
                       learner=learner, method=method, data_seed=data.data_seed,
                       fold_hash=fair.fold_digest(folds), artificial_pressure=False,
                       treatment_learner=learner, train_n_per_model=config.n//2,
                       evaluation_n=config.n, pilot_mode=protocol["pilot"], fit_calls_per_bank=6,
                       every_row_held_out=crossfit)
            if bank is None:
                row.update(finite=False, failure_reason=failure)
            else:
                g, m, source = assemble(bank, folds, crossfit)
                result = estimate_iv_type(data.y, data.d, g, m)
                row.update(finite=result.finite, failure_reason=result.failure_reason,
                           bank_identity=bank.identity, bank_records=bank.records,
                           pilots=bank.pilots, source_half_sha256=fair.vector_hash(source),
                           prediction_overlap_n=int(np.sum(source == folds)),
                           g_prediction_hash=fair.vector_hash(g), m_prediction_hash=fair.vector_hash(m))
                if result.finite:
                    error = result.theta-config.theta0
                    a, b = g-data.g0, m-data.m0
                    oracle = float(np.mean(data.v*data.u))
                    first = float(np.mean(-data.v*a-b*data.u))
                    second = float(np.mean(b*a))
                    truth_score = float(np.mean((data.d-m)*(data.y-config.theta0*data.d-g)))
                    if not math.isclose(truth_score, oracle+first+second, abs_tol=1e-12):
                        raise RuntimeError("score decomposition failed")
                    row.update(theta=result.theta, se=result.se, bias=error, abs_error=abs(error),
                               squared_error=error**2, covers=abs(error)<=1.96*result.se,
                               score_mean=result.score_mean, jacobian=result.q_hat,
                               score_at_truth=truth_score, oracle_score=oracle,
                               first_order_score=first, second_order_score=second,
                               nuisance_rmse_g=float(np.sqrt(np.mean(a*a))),
                               nuisance_rmse_m=float(np.sqrt(np.mean(b*b))))
            rows.append(row)
    return rows
