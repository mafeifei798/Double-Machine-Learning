"""Source-aligned repairs of the original undergraduate DML simulations.

The data-generating processes, learner families, and two headline comparisons are
kept close to the March-April 2026 scripts.  The core estimator restores the
Robinson partialling-out score already present in the 3 March prototype:

    v = D - m_hat(X)
    r = Y - ell_hat(X)
    theta_hat = sum(v * r) / sum(v ** 2)

The module deliberately stays small.  It is an undergraduate simulation, not a
general causal-inference framework.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor


LEARNERS = ("rf", "xgb", "lasso")


@dataclass(frozen=True)
class SimConfig:
    """Configuration for one original-style Monte Carlo experiment."""

    design: str = "figure1"  # figure1 | figure2
    dgp: str = "nonlinear"  # linear | nonlinear
    n: int = 500
    p: int = 20
    reps: int = 300
    theta0: float = 0.5
    seed: int = 20260406
    k_folds: int = 2
    learners: tuple[str, ...] = LEARNERS
    n_estimators: int = 300
    rf_max_depth: int = 8
    min_samples_leaf: int = 2
    xgb_learning_rate: float = 0.01
    lasso_alpha_linear: float = 0.02
    lasso_alpha_nonlinear: float = 0.06

    def validate(self) -> None:
        if self.design not in {"figure1", "figure2"}:
            raise ValueError("design must be 'figure1' or 'figure2'")
        if self.dgp not in {"linear", "nonlinear"}:
            raise ValueError("dgp must be 'linear' or 'nonlinear'")
        if self.n < 20 or self.p < 5 or self.reps < 1:
            raise ValueError("require n >= 20, p >= 5, and reps >= 1")
        if not 2 <= self.k_folds <= self.n:
            raise ValueError("k_folds must satisfy 2 <= k_folds <= n")
        unknown = set(self.learners) - set(LEARNERS)
        if unknown:
            raise ValueError(f"unknown learners: {sorted(unknown)}")
        if self.n_estimators < 1:
            raise ValueError("n_estimators must be positive")


@dataclass(frozen=True)
class EstimationResult:
    theta: float
    se: float
    score_mean: float
    jacobian: float
    residual_signal: float
    finite: bool
    failure_reason: str


def _seed(base: int, *parts: int) -> int:
    value = np.random.SeedSequence([base, *parts]).generate_state(1, dtype=np.uint32)[0]
    return int(value)


def toeplitz_cov(p: int, rho: float = 0.7) -> np.ndarray:
    idx = np.arange(p)
    return rho ** np.abs(idx[:, None] - idx[None, :])


def sigmoid(z: np.ndarray) -> np.ndarray:
    z_clip = np.clip(z, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z_clip))


def nuisance_truth(config: SimConfig, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return structural g0(X) and treatment regression m0(X)."""

    if config.design == "figure1":
        if config.dgp == "linear":
            m0 = 0.9 * x[:, 0] + 0.20 * x[:, 2]
            g0 = 0.25 * x[:, 0] + 0.25 * x[:, 2]
        else:
            m0 = 0.9 * x[:, 0] + 0.20 * sigmoid(x[:, 2])
            g0 = sigmoid(x[:, 0]) + 0.25 * x[:, 2]
    else:
        if config.dgp == "linear":
            m0 = 1.2 * x[:, 0] + 0.35 * x[:, 2]
            g0 = 0.40 * x[:, 0] + 0.40 * x[:, 2]
        else:
            m0 = 1.2 * x[:, 0] + 0.35 * sigmoid(x[:, 2])
            g0 = 1.2 * sigmoid(x[:, 0]) + 0.40 * x[:, 2]
    return g0, m0


def draw_data(
    config: SimConfig, rep: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Draw one paired synthetic dataset and return X, Y, D, g0, m0, ell0."""

    rng = np.random.default_rng(_seed(config.seed, rep, 1))
    x = rng.multivariate_normal(
        mean=np.zeros(config.p), cov=toeplitz_cov(config.p), size=config.n
    )
    u = rng.normal(size=config.n)
    v = rng.normal(size=config.n)
    g0, m0 = nuisance_truth(config, x)
    d = m0 + v
    y = config.theta0 * d + g0 + u
    ell0 = config.theta0 * m0 + g0
    return x, y, d, g0, m0, ell0


def make_fold_ids(n: int, k_folds: int, seed: int) -> np.ndarray:
    """Create one partition that is reused by every learner and nuisance target."""

    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    fold_id = np.empty(n, dtype=np.int16)
    for fold, indices in enumerate(np.array_split(order, k_folds)):
        fold_id[indices] = fold
    return fold_id


def fold_hash(fold_id: np.ndarray, k_folds: int) -> str:
    payload = b"dml-original-v1\0" + int(k_folds).to_bytes(4, "little")
    payload += int(fold_id.size).to_bytes(8, "little") + fold_id.astype("<i2").tobytes()
    return hashlib.sha256(payload).hexdigest()[:16]


def lasso_feature_map(x: np.ndarray) -> np.ndarray:
    feats = [x, x * x]
    interactions = []
    active = min(6, x.shape[1])
    for left in range(active):
        for right in range(left + 1, active):
            interactions.append((x[:, left] * x[:, right])[:, None])
    if interactions:
        feats.append(np.hstack(interactions))
    mapped = np.hstack(feats)
    if not np.isfinite(mapped).all():
        raise FloatingPointError("non-finite Lasso feature map")
    return np.clip(mapped, -5.0, 5.0)


class LassoNuisance:
    def __init__(self, linear_mode: bool, alpha: float, random_state: int):
        self.linear_mode = linear_mode
        self.scaler = StandardScaler()
        self.model = Lasso(
            alpha=alpha,
            fit_intercept=True,
            max_iter=25_000,
            tol=1e-4,
            random_state=random_state,
        )

    def _features(self, x: np.ndarray) -> np.ndarray:
        return x if self.linear_mode else lasso_feature_map(x)

    def fit(self, x: np.ndarray, target: np.ndarray) -> "LassoNuisance":
        scaled = self.scaler.fit_transform(self._features(x))
        if not np.isfinite(scaled).all() or not np.isfinite(target).all():
            raise FloatingPointError("non-finite Lasso training data")
        self.model.fit(np.clip(scaled, -10.0, 10.0), target)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        scaled = self.scaler.transform(self._features(x))
        if not np.isfinite(scaled).all():
            raise FloatingPointError("non-finite Lasso prediction features")
        prediction = np.asarray(self.model.predict(np.clip(scaled, -10.0, 10.0)))
        if not np.isfinite(prediction).all():
            raise FloatingPointError("non-finite Lasso prediction")
        return prediction


def make_learner(config: SimConfig, learner: str, seed: int) -> Any:
    if learner == "rf":
        return RandomForestRegressor(
            n_estimators=config.n_estimators,
            max_depth=config.rf_max_depth,
            min_samples_leaf=config.min_samples_leaf,
            max_features="sqrt",
            random_state=seed,
            n_jobs=1,
        )
    if learner == "xgb":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("xgboost is required when learner='xgb'") from exc
        return XGBRegressor(
            n_estimators=config.n_estimators,
            max_depth=3,
            learning_rate=config.xgb_learning_rate,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            min_child_weight=2.0,
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=1,
            verbosity=0,
        )
    if learner == "lasso":
        alpha = (
            config.lasso_alpha_linear
            if config.dgp == "linear"
            else config.lasso_alpha_nonlinear
        )
        return LassoNuisance(config.dgp == "linear", alpha, seed)
    raise ValueError(f"unknown learner: {learner}")


def _fit_predict(
    config: SimConfig,
    learner: str,
    x_train: np.ndarray,
    target: np.ndarray,
    x_test: np.ndarray,
    seed: int,
) -> np.ndarray:
    model = make_learner(config, learner, seed)
    model.fit(x_train, target)
    prediction = np.asarray(model.predict(x_test), dtype=float)
    if prediction.shape != (x_test.shape[0],) or not np.isfinite(prediction).all():
        raise FloatingPointError("nuisance prediction is non-finite or has the wrong shape")
    return prediction


def nuisance_predictions(
    config: SimConfig,
    learner: str,
    rep: int,
    x: np.ndarray,
    y: np.ndarray,
    d: np.ndarray,
    fold_id: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return in-sample and out-of-fold predictions for ell=E[Y|X] and m=E[D|X]."""

    learner_index = LEARNERS.index(learner) + 1
    ell_full = _fit_predict(
        config, learner, x, y, x, _seed(config.seed, rep, 100, learner_index, 1)
    )
    m_full = _fit_predict(
        config, learner, x, d, x, _seed(config.seed, rep, 100, learner_index, 2)
    )

    ell_oof = np.empty(config.n, dtype=float)
    m_oof = np.empty(config.n, dtype=float)
    predicted = np.zeros(config.n, dtype=np.int8)
    for fold in range(config.k_folds):
        test = np.flatnonzero(fold_id == fold)
        train = np.flatnonzero(fold_id != fold)
        if test.size == 0 or train.size == 0 or np.intersect1d(train, test).size:
            raise RuntimeError("invalid cross-fitting partition")
        ell_oof[test] = _fit_predict(
            config,
            learner,
            x[train],
            y[train],
            x[test],
            _seed(config.seed, rep, 200, learner_index, fold, 1),
        )
        m_oof[test] = _fit_predict(
            config,
            learner,
            x[train],
            d[train],
            x[test],
            _seed(config.seed, rep, 200, learner_index, fold, 2),
        )
        predicted[test] += 1
    if not np.all(predicted == 1):
        raise RuntimeError("each observation must receive exactly one OOF prediction")
    return ell_full, m_full, ell_oof, m_oof


def _failure(reason: str) -> EstimationResult:
    return EstimationResult(
        theta=math.nan,
        se=math.nan,
        score_mean=math.nan,
        jacobian=math.nan,
        residual_signal=math.nan,
        finite=False,
        failure_reason=reason,
    )


def partialling_out(
    y: np.ndarray, d: np.ndarray, ell_hat: np.ndarray, m_hat: np.ndarray
) -> EstimationResult:
    """Robinson partialling-out estimate and uncentered HC0-style sandwich SE."""

    arrays = (y, d, ell_hat, m_hat)
    if any(a.shape != y.shape for a in arrays) or not all(np.isfinite(a).all() for a in arrays):
        return _failure("non_finite_or_shape_mismatch")
    v_hat = d - m_hat
    r_hat = y - ell_hat
    q_hat = float(np.mean(v_hat * v_hat))
    scale = max(float(np.mean(d * d)), 1.0)
    numerical_floor = 100.0 * np.finfo(float).eps * scale
    if not math.isfinite(q_hat) or q_hat <= numerical_floor:
        return _failure("degenerate_residual_treatment_signal")
    theta = float(np.mean(v_hat * r_hat) / q_hat)
    score = v_hat * (r_hat - theta * v_hat)
    se_squared = float(np.mean(score * score) / (y.size * q_hat * q_hat))
    if not math.isfinite(theta) or not math.isfinite(se_squared) or se_squared < 0.0:
        return _failure("non_finite_estimate_or_variance")
    return EstimationResult(
        theta=theta,
        se=math.sqrt(se_squared),
        score_mean=float(np.mean(score)),
        jacobian=-q_hat,
        residual_signal=q_hat / max(float(np.var(d)), np.finfo(float).tiny),
        finite=True,
        failure_reason="",
    )


def outcome_only_nonorthogonal(
    y: np.ndarray, d: np.ndarray, ell_hat: np.ndarray
) -> EstimationResult:
    """Historical outcome-only residualization baseline, explicitly non-orthogonal."""

    arrays = (y, d, ell_hat)
    if any(a.shape != y.shape for a in arrays) or not all(np.isfinite(a).all() for a in arrays):
        return _failure("non_finite_or_shape_mismatch")
    q_hat = float(np.mean(d * d))
    numerical_floor = 100.0 * np.finfo(float).eps * max(q_hat, 1.0)
    if not math.isfinite(q_hat) or q_hat <= numerical_floor:
        return _failure("degenerate_raw_treatment_signal")
    residual = y - ell_hat
    theta = float(np.mean(d * residual) / q_hat)
    score = d * (residual - theta * d)
    se_squared = float(np.mean(score * score) / (y.size * q_hat * q_hat))
    if not math.isfinite(theta) or not math.isfinite(se_squared) or se_squared < 0.0:
        return _failure("non_finite_estimate_or_variance")
    return EstimationResult(
        theta=theta,
        se=math.sqrt(se_squared),
        score_mean=float(np.mean(score)),
        jacobian=-q_hat,
        residual_signal=1.0,
        finite=True,
        failure_reason="",
    )


def _row(
    config: SimConfig,
    rep: int,
    learner: str,
    method: str,
    result: EstimationResult,
    nuisance_rmse_ell: float,
    nuisance_rmse_m: float,
    partition_hash: str,
) -> dict[str, Any]:
    theta = result.theta
    se = result.se
    finite = result.finite
    ci_low = theta - 1.96 * se if finite else math.nan
    ci_high = theta + 1.96 * se if finite else math.nan
    return {
        "design": config.design,
        "dgp": config.dgp,
        "replication": rep,
        "n": config.n,
        "p": config.p,
        "theta0": config.theta0,
        "learner": learner,
        "method": method,
        "k_folds": config.k_folds,
        "theta": theta,
        "se": se,
        "bias": theta - config.theta0 if finite else math.nan,
        "abs_error": abs(theta - config.theta0) if finite else math.nan,
        "squared_error": (theta - config.theta0) ** 2 if finite else math.nan,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "covers": bool(ci_low <= config.theta0 <= ci_high) if finite else False,
        "score_mean": result.score_mean,
        "jacobian": result.jacobian,
        "residual_signal": result.residual_signal,
        "nuisance_rmse_ell": nuisance_rmse_ell,
        "nuisance_rmse_m": nuisance_rmse_m,
        "fold_hash": partition_hash,
        "finite": finite,
        "failure_reason": result.failure_reason,
    }


def run_replication(config: SimConfig, rep: int) -> list[dict[str, Any]]:
    config.validate()
    x, y, d, _g0, m0, ell0 = draw_data(config, rep)
    fold_id = make_fold_ids(config.n, config.k_folds, _seed(config.seed, rep, 2))
    partition_hash = fold_hash(fold_id, config.k_folds)
    rows: list[dict[str, Any]] = []

    oracle = partialling_out(y, d, ell0, m0)
    rows.append(_row(config, rep, "oracle", "oracle_po", oracle, 0.0, 0.0, partition_hash))

    for learner in config.learners:
        try:
            ell_full, m_full, ell_oof, m_oof = nuisance_predictions(
                config, learner, rep, x, y, d, fold_id
            )
            full_rmse_ell = float(np.sqrt(np.mean((ell_full - ell0) ** 2)))
            full_rmse_m = float(np.sqrt(np.mean((m_full - m0) ** 2)))
            oof_rmse_ell = float(np.sqrt(np.mean((ell_oof - ell0) ** 2)))
            oof_rmse_m = float(np.sqrt(np.mean((m_oof - m0) ** 2)))

            rows.append(
                _row(
                    config,
                    rep,
                    learner,
                    "outcome_only_nonorthogonal",
                    outcome_only_nonorthogonal(y, d, ell_oof),
                    oof_rmse_ell,
                    math.nan,
                    partition_hash,
                )
            )
            rows.append(
                _row(
                    config,
                    rep,
                    learner,
                    "po_no_split",
                    partialling_out(y, d, ell_full, m_full),
                    full_rmse_ell,
                    full_rmse_m,
                    partition_hash,
                )
            )
            rows.append(
                _row(
                    config,
                    rep,
                    learner,
                    "po_crossfit",
                    partialling_out(y, d, ell_oof, m_oof),
                    oof_rmse_ell,
                    oof_rmse_m,
                    partition_hash,
                )
            )
        except Exception as exc:  # one learner failure must not erase paired data
            reason = f"{type(exc).__name__}:{exc}"
            for method in (
                "outcome_only_nonorthogonal",
                "po_no_split",
                "po_crossfit",
            ):
                rows.append(
                    _row(
                        config,
                        rep,
                        learner,
                        method,
                        _failure(reason),
                        math.nan,
                        math.nan,
                        partition_hash,
                    )
                )
    return rows


def run_experiment(config: SimConfig) -> list[dict[str, Any]]:
    config.validate()
    rows: list[dict[str, Any]] = []
    for rep in range(config.reps):
        rows.extend(run_replication(config, rep))
    return rows


def run_overfit_replication(config: SimConfig, rep: int) -> list[dict[str, Any]]:
    """Source-motivated stress: an overfit outcome tree with a regularized RF for m.

    The original paper's Figure 2 deliberately studies an outcome nuisance that
    overfits observations used again in the score.  This transparent analogue
    keeps the same estimator and data while changing only whether predictions are
    in-sample or out-of-fold.  It is a stress test, not the headline learner
    comparison.
    """

    if config.design != "figure2":
        raise ValueError("the overfit stress is defined only for the Figure 2 design")
    x, y, d, _g0, m0, ell0 = draw_data(config, rep)
    fold_id = make_fold_ids(config.n, config.k_folds, _seed(config.seed, rep, 2))
    partition_hash = fold_hash(fold_id, config.k_folds)

    ell_full_model = DecisionTreeRegressor(
        min_samples_leaf=2, random_state=_seed(config.seed, rep, 800, 1)
    )
    ell_full_model.fit(x, y)
    ell_full = np.asarray(ell_full_model.predict(x), dtype=float)
    m_full = _fit_predict(
        config, "rf", x, d, x, _seed(config.seed, rep, 800, 2)
    )

    ell_oof = np.empty(config.n, dtype=float)
    m_oof = np.empty(config.n, dtype=float)
    predicted = np.zeros(config.n, dtype=np.int8)
    for fold in range(config.k_folds):
        test = np.flatnonzero(fold_id == fold)
        train = np.flatnonzero(fold_id != fold)
        outcome_tree = DecisionTreeRegressor(
            min_samples_leaf=2,
            random_state=_seed(config.seed, rep, 801, fold, 1),
        )
        outcome_tree.fit(x[train], y[train])
        ell_oof[test] = outcome_tree.predict(x[test])
        m_oof[test] = _fit_predict(
            config,
            "rf",
            x[train],
            d[train],
            x[test],
            _seed(config.seed, rep, 801, fold, 2),
        )
        predicted[test] += 1
    if not np.all(predicted == 1):
        raise RuntimeError("each observation must receive exactly one OOF prediction")

    learner = "overfit_outcome_tree_plus_rf"
    return [
        _row(
            config,
            rep,
            learner,
            "overfit_po_no_split",
            partialling_out(y, d, ell_full, m_full),
            float(np.sqrt(np.mean((ell_full - ell0) ** 2))),
            float(np.sqrt(np.mean((m_full - m0) ** 2))),
            partition_hash,
        ),
        _row(
            config,
            rep,
            learner,
            "overfit_po_crossfit",
            partialling_out(y, d, ell_oof, m_oof),
            float(np.sqrt(np.mean((ell_oof - ell0) ** 2))),
            float(np.sqrt(np.mean((m_oof - m0) ** 2))),
            partition_hash,
        ),
    ]


def run_overfit_stress(config: SimConfig) -> list[dict[str, Any]]:
    config.validate()
    rows: list[dict[str, Any]] = []
    for rep in range(config.reps):
        rows.extend(run_overfit_replication(config, rep))
    return rows


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return math.nan, math.nan
    if not 0 <= successes <= total:
        raise ValueError("successes must be between zero and total")
    p_hat = successes / total
    denominator = 1.0 + z * z / total
    center = (p_hat + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(
        p_hat * (1.0 - p_hat) / total + z * z / (4.0 * total * total)
    ) / denominator
    low = 0.0 if successes == 0 else max(0.0, center - radius)
    high = 1.0 if successes == total else min(1.0, center + radius)
    return low, high


def aggregate_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["design"], row["dgp"], row["learner"], row["method"])
        groups.setdefault(key, []).append(row)

    summary: list[dict[str, Any]] = []
    for key in sorted(groups):
        group = groups[key]
        finite = [row for row in group if row["finite"]]
        total = len(group)
        count = len(finite)
        if count == 0:
            metrics = {
                name: math.nan
                for name in (
                    "bias",
                    "bias_mcse",
                    "rmse",
                    "rmse_mcse",
                    "empirical_sd",
                    "mean_se",
                    "se_sd_ratio",
                    "coverage",
                    "coverage_mcse",
                    "coverage_wilson_low",
                    "coverage_wilson_high",
                    "mean_ci_length",
                    "mean_nuisance_rmse_ell",
                    "mean_nuisance_rmse_m",
                )
            }
        else:
            errors = np.asarray([row["bias"] for row in finite], dtype=float)
            ses = np.asarray([row["se"] for row in finite], dtype=float)
            squared = errors * errors
            rmse = float(np.sqrt(np.mean(squared)))
            empirical_sd = float(np.std(errors, ddof=1)) if count > 1 else math.nan
            successes = sum(bool(row["covers"]) for row in finite)
            coverage = successes / count
            low, high = wilson_interval(successes, count)
            ell_values = np.asarray(
                [row["nuisance_rmse_ell"] for row in finite], dtype=float
            )
            m_values = np.asarray([row["nuisance_rmse_m"] for row in finite], dtype=float)
            metrics = {
                "bias": float(np.mean(errors)),
                "bias_mcse": empirical_sd / math.sqrt(count) if count > 1 else math.nan,
                "rmse": rmse,
                "rmse_mcse": (
                    float(np.std(squared, ddof=1)) / (2.0 * rmse * math.sqrt(count))
                    if count > 1 and rmse > 0.0
                    else math.nan
                ),
                "empirical_sd": empirical_sd,
                "mean_se": float(np.mean(ses)),
                "se_sd_ratio": (
                    float(np.mean(ses)) / empirical_sd
                    if math.isfinite(empirical_sd) and empirical_sd > 0.0
                    else math.nan
                ),
                "coverage": coverage,
                "coverage_mcse": math.sqrt(coverage * (1.0 - coverage) / count),
                "coverage_wilson_low": low,
                "coverage_wilson_high": high,
                "mean_ci_length": float(np.mean(2.0 * 1.96 * ses)),
                "mean_nuisance_rmse_ell": (
                    float(np.nanmean(ell_values))
                    if np.isfinite(ell_values).any()
                    else math.nan
                ),
                "mean_nuisance_rmse_m": (
                    float(np.nanmean(m_values)) if np.isfinite(m_values).any() else math.nan
                ),
            }
        summary.append(
            {
                "design": key[0],
                "dgp": key[1],
                "learner": key[2],
                "method": key[3],
                "replications": total,
                "finite_replications": count,
                "failure_rate": (total - count) / total,
                **metrics,
            }
        )
    return summary


def config_dict(config: SimConfig) -> dict[str, Any]:
    output = asdict(config)
    output["learners"] = list(config.learners)
    return output


def select_rows(
    rows: Iterable[dict[str, Any]], learner: str, method: str
) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row["learner"] == learner and row["method"] == method and row["finite"]
    ]
