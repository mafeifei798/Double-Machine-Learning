"""Structural IV-type estimation, nuisance models, and simulation data utilities."""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler


LEARNERS = ("rf", "xgb", "lasso")


def stable_seed(base: int, *labels: object) -> int:
    """Map an ordered semantic key to a deterministic uint32 seed."""

    if isinstance(base, bool) or not isinstance(base, (int, np.integer)):
        raise ValueError("base seed must be an integer")
    if not 0 <= int(base) <= np.iinfo(np.uint32).max:
        raise ValueError("base seed must fit uint32")
    payload = [b"thesis-iv-repair-v2", str(int(base)).encode("utf-8")]
    payload.extend(str(label).encode("utf-8") for label in labels)
    return int.from_bytes(hashlib.sha256(b"\0".join(payload)).digest()[:4], "little")

@dataclass(frozen=True)
class IVRepairConfig:
    """Configuration for one IV-type simulation cell."""

    design: Literal["figure1", "figure2"]
    dgp: Literal["linear", "nonlinear"]
    n: int = 1_000
    p: int = 20
    reps: int = 1_000
    theta0: float = 0.5
    seed: int = 1_987_654_321
    k_folds: int = 2
    pilot_k_folds: int = 2
    pilot_iterations: int = 2
    learners: tuple[str, ...] = LEARNERS
    n_estimators: int = 300
    rf_max_depth: int = 8
    rf_min_samples_leaf: int = 2
    xgb_learning_rate: float = 0.01
    lasso_alpha_linear: float = 0.05
    lasso_alpha_nonlinear: float = 0.06
    self_influence_epsilon: float = 0.25

    def validate(self) -> None:
        if self.design not in {"figure1", "figure2"}:
            raise ValueError("design must be figure1 or figure2")
        if self.dgp not in {"linear", "nonlinear"}:
            raise ValueError("dgp must be linear or nonlinear")
        if self.n < 40 or self.p < 3 or self.reps < 1:
            raise ValueError("require n >= 40, p >= 3, and reps >= 1")
        if not 2 <= self.k_folds <= self.n // 2:
            raise ValueError("k_folds is incompatible with n")
        smallest_outer_train = self.n - math.ceil(self.n / self.k_folds)
        if not 2 <= self.pilot_k_folds <= smallest_outer_train // 2:
            raise ValueError("pilot_k_folds is incompatible with outer training size")
        if self.pilot_iterations < 1:
            raise ValueError("pilot_iterations must be positive")
        if self.n_estimators < 1:
            raise ValueError("n_estimators must be positive")
        if not 1 <= self.rf_min_samples_leaf <= self.n:
            raise ValueError("rf_min_samples_leaf is invalid")
        if not 0.0 < self.self_influence_epsilon < 0.5:
            raise ValueError("self_influence_epsilon must lie in (0, 0.5)")
        unknown = set(self.learners) - set(LEARNERS)
        if unknown or not self.learners:
            raise ValueError(f"invalid learners: {sorted(unknown)}")

@dataclass(frozen=True)
class PLRData:
    x: np.ndarray
    y: np.ndarray
    d: np.ndarray
    g0: np.ndarray
    m0: np.ndarray
    ell0: np.ndarray
    u: np.ndarray
    v: np.ndarray
    data_seed: int

@dataclass(frozen=True)
class Estimate:
    theta: float
    se: float
    score_mean: float
    q_hat: float
    finite: bool
    failure_reason: str

def toeplitz_cov(p: int, rho: float = 0.7) -> np.ndarray:
    index = np.arange(p)
    return rho ** np.abs(index[:, None] - index[None, :])

def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=float), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))

def nuisance_truth(
    design: str, dgp: str, x: np.ndarray, theta0: float = 0.5
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return structural g0, treatment m0, and outcome regression ell0."""

    if design == "figure1":
        if dgp == "linear":
            m0 = 0.9 * x[:, 0] + 0.20 * x[:, 2]
            g0 = 0.25 * x[:, 0] + 0.25 * x[:, 2]
        else:
            m0 = 0.9 * x[:, 0] + 0.20 * sigmoid(x[:, 2])
            g0 = sigmoid(x[:, 0]) + 0.25 * x[:, 2]
    elif design == "figure2":
        if dgp == "linear":
            m0 = 1.2 * x[:, 0] + 0.35 * x[:, 2]
            g0 = 0.40 * x[:, 0] + 0.40 * x[:, 2]
        else:
            m0 = 1.2 * x[:, 0] + 0.35 * sigmoid(x[:, 2])
            g0 = 1.2 * sigmoid(x[:, 0]) + 0.40 * x[:, 2]
    else:
        raise ValueError("unknown design")
    return g0, m0, g0 + theta0 * m0

def make_fold_ids(n: int, k_folds: int, seed: int) -> np.ndarray:
    if not 2 <= k_folds <= n:
        raise ValueError("require 2 <= k_folds <= n")
    order = np.random.default_rng(seed).permutation(n)
    fold_id = np.empty(n, dtype=np.int16)
    for fold, indices in enumerate(np.array_split(order, k_folds)):
        fold_id[indices] = fold
    if set(fold_id.tolist()) != set(range(k_folds)):
        raise RuntimeError("fold construction failed exact coverage")
    return fold_id

def fold_digest(fold_id: np.ndarray) -> str:
    array = np.asarray(fold_id, dtype="<i2")
    payload = b"thesis-iv-repair-fold-v1\0"
    payload += int(array.size).to_bytes(8, "little") + array.tobytes()
    return hashlib.sha256(payload).hexdigest()[:16]

def exact_series_basis(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] < 3 or not np.isfinite(x).all():
        raise ValueError("x must be finite, two-dimensional, and have >=3 columns")
    basis = np.column_stack(
        (
            np.ones(x.shape[0]),
            x[:, 0],
            x[:, 2],
            sigmoid(x[:, 0]),
            sigmoid(x[:, 2]),
        )
    )
    if not np.isfinite(basis).all():
        raise FloatingPointError("non-finite series basis")
    return basis

class SeriesOLS:
    """Unregularized OLS on the fixed five-column nuisance basis."""

    def __init__(self) -> None:
        self.coef_: np.ndarray | None = None
        self.rank_: int | None = None

    def fit(self, x: np.ndarray, target: np.ndarray) -> "SeriesOLS":
        design = exact_series_basis(x)
        target = np.asarray(target, dtype=float)
        if target.shape != (design.shape[0],) or not np.isfinite(target).all():
            raise ValueError("target must be finite and aligned with x")
        coefficient, _residuals, rank, _singular = np.linalg.lstsq(
            design, target, rcond=None
        )
        if int(rank) != design.shape[1]:
            raise np.linalg.LinAlgError(
                f"series design is rank deficient: {rank}/{design.shape[1]}"
            )
        if not np.isfinite(coefficient).all():
            raise FloatingPointError("non-finite series coefficient")
        self.coef_ = np.asarray(coefficient, dtype=float)
        self.rank_ = int(rank)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("SeriesOLS must be fit before predict")
        prediction = exact_series_basis(x) @ self.coef_
        if not np.isfinite(prediction).all():
            raise FloatingPointError("non-finite series prediction")
        return np.asarray(prediction, dtype=float)

def _failure(reason: str) -> Estimate:
    return Estimate(math.nan, math.nan, math.nan, math.nan, False, reason)

def estimate_iv_type(
    y: np.ndarray, d: np.ndarray, g_hat: np.ndarray, m_hat: np.ndarray
) -> Estimate:
    """Estimate theta with Eq. 4.3 and an uncentered HC0 sandwich SE."""

    arrays = [np.asarray(value, dtype=float) for value in (y, d, g_hat, m_hat)]
    if any(value.ndim != 1 for value in arrays):
        return _failure("inputs_must_be_vectors")
    if len({value.shape for value in arrays}) != 1 or not all(
        np.isfinite(value).all() for value in arrays
    ):
        return _failure("nonfinite_or_shape_mismatch")
    yv, dv, gv, mv = arrays
    v_hat = dv - mv
    q_hat = float(np.mean(v_hat * dv))
    scale = math.sqrt(
        max(float(np.mean(v_hat * v_hat) * np.mean(dv * dv)), 0.0)
    )
    if not math.isfinite(q_hat) or abs(q_hat) <= 1e-12 * max(scale, 1.0):
        return _failure("weak_sample_jacobian")
    theta = float(np.mean(v_hat * (yv - gv)) / q_hat)
    score = v_hat * (yv - gv - theta * dv)
    se = math.sqrt(float(np.mean(score * score)) / (yv.size * q_hat * q_hat))
    if not all(math.isfinite(value) for value in (theta, se)) or se <= 0.0:
        return _failure("nonfinite_estimate")
    return Estimate(theta, se, float(np.mean(score)), q_hat, True, "")

def estimate_outcome_mismatch(
    y: np.ndarray, d: np.ndarray, ell_hat: np.ndarray
) -> Estimate:
    """Compute the raw-treatment residual ratio.

    Structural g gives the non-orthogonal ML estimator. Outcome regression
    ell gives the historical nuisance-mismatch diagnostic.
    """

    yv, dv, lv = (np.asarray(value, dtype=float) for value in (y, d, ell_hat))
    if yv.ndim != 1 or dv.shape != yv.shape or lv.shape != yv.shape:
        return _failure("nonfinite_or_shape_mismatch")
    if not all(np.isfinite(value).all() for value in (yv, dv, lv)):
        return _failure("nonfinite_or_shape_mismatch")
    q_hat = float(np.mean(dv * dv))
    if q_hat <= np.finfo(float).tiny:
        return _failure("weak_sample_jacobian")
    theta = float(np.mean(dv * (yv - lv)) / q_hat)
    score = dv * (yv - lv - theta * dv)
    se = math.sqrt(float(np.mean(score * score)) / (yv.size * q_hat * q_hat))
    if not all(math.isfinite(value) for value in (theta, se)) or se <= 0.0:
        return _failure("nonfinite_estimate")
    return Estimate(theta, se, float(np.mean(score)), q_hat, True, "")

def _lasso_feature_map(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    features = [x, x * x]
    interactions = []
    for left in range(min(6, x.shape[1])):
        for right in range(left + 1, min(6, x.shape[1])):
            interactions.append((x[:, left] * x[:, right])[:, None])
    if interactions:
        features.append(np.hstack(interactions))
    mapped = np.clip(np.hstack(features), -5.0, 5.0)
    if not np.isfinite(mapped).all():
        raise FloatingPointError("non-finite Lasso feature map")
    return mapped

class LassoNuisance:
    def __init__(self, *, linear: bool, alpha: float, seed: int) -> None:
        self.linear = linear
        self.scaler = StandardScaler()
        self.model = Lasso(
            alpha=alpha,
            fit_intercept=True,
            max_iter=25_000,
            tol=1e-4,
            random_state=seed,
        )

    def _features(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=float) if self.linear else _lasso_feature_map(x)

    def fit(self, x: np.ndarray, target: np.ndarray) -> "LassoNuisance":
        values = self.scaler.fit_transform(self._features(x))
        self.model.fit(np.clip(values, -10.0, 10.0), np.asarray(target, dtype=float))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        values = self.scaler.transform(self._features(x))
        prediction = self.model.predict(np.clip(values, -10.0, 10.0))
        return np.asarray(prediction, dtype=float)
