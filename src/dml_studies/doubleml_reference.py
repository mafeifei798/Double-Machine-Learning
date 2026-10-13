"""Shared data generation and deterministic identities for the DML studies."""
from __future__ import annotations

import hashlib
import numpy as np

from .iv_score import (
    IVRepairConfig, PLRData, make_fold_ids, fold_digest, nuisance_truth, toeplitz_cov,
)


def seed_for(base: int, *labels: object) -> int:
    payload = "\0".join(map(str, ("fair-iv-v1", base, *labels))).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")

def draw(config: IVRepairConfig, rep: int) -> PLRData:
    config.validate()
    data_seed = seed_for(config.seed, config.design, config.dgp, config.n, config.p, rep, "data")
    rng = np.random.default_rng(data_seed)
    x = rng.multivariate_normal(np.zeros(config.p), toeplitz_cov(config.p), size=config.n)
    u, v = rng.normal(size=config.n), rng.normal(size=config.n)
    g, m, ell = nuisance_truth(config.design, config.dgp, x, config.theta0)
    d = m + v
    return PLRData(x, config.theta0*d+g+u, d, g, m, ell, u, v, data_seed)

def vector_hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest()
