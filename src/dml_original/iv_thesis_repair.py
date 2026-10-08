"""IV-type-only repair of the two original thesis simulation stories.

The active causal estimator in this module is always the structural score

    (D - m_hat(X)) * (Y - theta * D - g_hat(X)).

Figure 1 keeps the historically flawed outcome-nuisance/raw-treatment ratio as
an explicitly non-DML diagnostic and compares it with cross-fitted IV-type DML.
Figure 2 uses the same IV-type estimator on both sides and adds a prespecified
own-observation stress only to the no-split outcome nuisance.  The stress is a
controlled mechanism check; it is not a claim that every ordinary no-split
learner must be biased.

All structural-outcome learner targets are ``Y - theta_pilot * D``.  Every
outer-fold pilot is computed inside the corresponding training complement by
fixed-point updates that themselves use only the IV-type ratio.  A held-out
outcome therefore cannot enter its own nuisance prediction, even through a
pooled pilot scalar.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler


LEARNERS = ("rf", "xgb", "lasso")
FIGURE1_METHODS = ("outcome_nuisance_mismatch", "iv_type_crossfit")
FIGURE2_FIT_MODES = ("no_split", "crossfit")


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
    """One DGP/design cell in the independent thesis repair."""

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


@dataclass(frozen=True)
class CrossfitNuisances:
    g_hat: np.ndarray
    m_hat: np.ndarray
    theta_pilots: tuple[float, ...]
    fold_id: np.ndarray
    fold_hash: str
    min_series_rank: int


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


def draw_data(config: IVRepairConfig, rep: int) -> PLRData:
    config.validate()
    data_seed = stable_seed(
        config.seed, config.design, config.dgp, config.n, config.p, rep, "data"
    )
    rng = np.random.default_rng(data_seed)
    x = rng.multivariate_normal(
        mean=np.zeros(config.p), cov=toeplitz_cov(config.p), size=config.n
    )
    u = rng.normal(size=config.n)
    v = rng.normal(size=config.n)
    g0, m0, ell0 = nuisance_truth(config.design, config.dgp, x, config.theta0)
    d = m0 + v
    y = config.theta0 * d + g0 + u
    return PLRData(x, y, d, g0, m0, ell0, u, v, data_seed)


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


def crossfit_series(
    x: np.ndarray, target: np.ndarray, fold_id: np.ndarray
) -> tuple[np.ndarray, tuple[int, ...]]:
    x = np.asarray(x, dtype=float)
    target = np.asarray(target, dtype=float)
    fold_id = np.asarray(fold_id)
    if target.shape != (x.shape[0],) or fold_id.shape != target.shape:
        raise ValueError("x, target, and fold_id are not aligned")
    folds = sorted(int(value) for value in np.unique(fold_id))
    if folds != list(range(len(folds))) or len(folds) < 2:
        raise ValueError("fold IDs must begin at zero and be consecutive")
    prediction = np.empty(target.size, dtype=float)
    coverage = np.zeros(target.size, dtype=np.int8)
    ranks: list[int] = []
    for fold in folds:
        test = fold_id == fold
        train = ~test
        model = SeriesOLS().fit(x[train], target[train])
        prediction[test] = model.predict(x[test])
        coverage[test] += 1
        assert model.rank_ is not None
        ranks.append(model.rank_)
    if not np.all(coverage == 1) or not np.isfinite(prediction).all():
        raise RuntimeError("series cross-fit failed exact finite coverage")
    return prediction, tuple(ranks)


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
    """Historical non-DML ratio using an outcome nuisance with raw treatment."""

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


def make_learner(config: IVRepairConfig, learner: str, seed: int) -> Any:
    if learner == "rf":
        return RandomForestRegressor(
            n_estimators=config.n_estimators,
            max_depth=config.rf_max_depth,
            min_samples_leaf=config.rf_min_samples_leaf,
            max_features="sqrt",
            random_state=seed,
            n_jobs=1,
        )
    if learner == "xgb":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("xgboost is required for learner='xgb'") from exc
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
        return LassoNuisance(
            linear=config.dgp == "linear", alpha=alpha, seed=seed
        )
    raise ValueError(f"unknown learner: {learner}")


def _fit_predict_learner(
    config: IVRepairConfig,
    learner: str,
    x_train: np.ndarray,
    target_train: np.ndarray,
    x_test: np.ndarray,
    seed: int,
) -> np.ndarray:
    model = make_learner(config, learner, seed)
    model.fit(x_train, target_train)
    prediction = np.asarray(model.predict(x_test), dtype=float)
    if prediction.shape != (x_test.shape[0],) or not np.isfinite(prediction).all():
        raise FloatingPointError("learner prediction is non-finite or misaligned")
    return prediction


def training_iv_pilot(
    x: np.ndarray,
    y: np.ndarray,
    d: np.ndarray,
    *,
    k_folds: int,
    iterations: int,
    seed: int,
) -> float:
    """Training-only fixed-point pilot, using the IV-type ratio at every update."""

    fold_id = make_fold_ids(x.shape[0], k_folds, seed)
    m_oof, _ = crossfit_series(x, d, fold_id)
    theta = 0.0
    for _iteration in range(iterations):
        target = y - theta * d
        g_oof, _ = crossfit_series(x, target, fold_id)
        result = estimate_iv_type(y, d, g_oof, m_oof)
        if not result.finite:
            raise RuntimeError(f"training-only IV pilot failed: {result.failure_reason}")
        theta = result.theta
    return float(theta)


def outer_fold_ids(config: IVRepairConfig, rep: int) -> np.ndarray:
    seed = stable_seed(
        config.seed, config.design, config.dgp, config.n, config.p, rep, "outer_folds"
    )
    return make_fold_ids(config.n, config.k_folds, seed)


def crossfit_structural_nuisances(
    config: IVRepairConfig,
    learner: str,
    rep: int,
    data: PLRData,
    fold_id: np.ndarray | None = None,
) -> CrossfitNuisances:
    """Cross-fit structural g and series m with fold-local IV pilots."""

    config.validate()
    if learner not in config.learners:
        raise ValueError("learner is outside the configured universe")
    folds = outer_fold_ids(config, rep) if fold_id is None else np.asarray(fold_id)
    if folds.shape != (config.n,) or set(folds.tolist()) != set(range(config.k_folds)):
        raise ValueError("outer fold vector does not match config")
    g_hat = np.empty(config.n, dtype=float)
    m_hat = np.empty(config.n, dtype=float)
    coverage = np.zeros(config.n, dtype=np.int8)
    pilots: list[float] = []
    ranks: list[int] = []
    for fold in range(config.k_folds):
        test = folds == fold
        train = ~test
        pilot_seed = stable_seed(
            config.seed,
            config.design,
            config.dgp,
            config.n,
            config.p,
            rep,
            "outer",
            fold,
            "pilot",
        )
        pilot = training_iv_pilot(
            data.x[train],
            data.y[train],
            data.d[train],
            k_folds=config.pilot_k_folds,
            iterations=config.pilot_iterations,
            seed=pilot_seed,
        )
        learner_seed = stable_seed(
            config.seed,
            config.design,
            config.dgp,
            config.n,
            config.p,
            rep,
            learner,
            "outer",
            fold,
            "structural_g",
        )
        structural_target = data.y[train] - pilot * data.d[train]
        g_hat[test] = _fit_predict_learner(
            config,
            learner,
            data.x[train],
            structural_target,
            data.x[test],
            learner_seed,
        )
        m_model = SeriesOLS().fit(data.x[train], data.d[train])
        m_hat[test] = m_model.predict(data.x[test])
        assert m_model.rank_ is not None
        ranks.append(m_model.rank_)
        pilots.append(pilot)
        coverage[test] += 1
    if not np.all(coverage == 1) or not np.isfinite(g_hat).all() or not np.isfinite(m_hat).all():
        raise RuntimeError("outer cross-fit failed exact finite coverage")
    return CrossfitNuisances(
        g_hat,
        m_hat,
        tuple(pilots),
        np.asarray(folds, dtype=np.int16),
        fold_digest(folds),
        min(ranks),
    )


def full_sample_structural_nuisances(
    config: IVRepairConfig, learner: str, rep: int, data: PLRData
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Fit structural g and series m on all rows and predict those same rows."""

    pilot = training_iv_pilot(
        data.x,
        data.y,
        data.d,
        k_folds=config.pilot_k_folds,
        iterations=config.pilot_iterations,
        seed=stable_seed(
            config.seed,
            config.design,
            config.dgp,
            config.n,
            config.p,
            rep,
            "full_pilot",
        ),
    )
    target = data.y - pilot * data.d
    g_hat = _fit_predict_learner(
        config,
        learner,
        data.x,
        target,
        data.x,
        stable_seed(
            config.seed,
            config.design,
            config.dgp,
            config.n,
            config.p,
            rep,
            learner,
            "full_structural_g",
        ),
    )
    m_model = SeriesOLS().fit(data.x, data.d)
    assert m_model.rank_ is not None
    return g_hat, m_model.predict(data.x), pilot, m_model.rank_


def self_influence_weight(config: IVRepairConfig) -> float:
    """Article-inspired N^(-1/2+epsilon) stress weight."""

    return float(config.n ** (-0.5 + config.self_influence_epsilon))


def _estimate_fields(result: Estimate, theta0: float) -> dict[str, Any]:
    if not result.finite:
        return {
            "theta": math.nan,
            "se": math.nan,
            "bias": math.nan,
            "abs_error": math.nan,
            "z_error": math.nan,
            "ci_low": math.nan,
            "ci_high": math.nan,
            "covers": False,
            "score_mean": math.nan,
            "q_hat": math.nan,
            "finite": False,
            "failure_reason": result.failure_reason,
        }
    low = result.theta - 1.96 * result.se
    high = result.theta + 1.96 * result.se
    return {
        "theta": result.theta,
        "se": result.se,
        "bias": result.theta - theta0,
        "abs_error": abs(result.theta - theta0),
        "z_error": (result.theta - theta0) / result.se,
        "ci_low": low,
        "ci_high": high,
        "covers": bool(low <= theta0 <= high),
        "score_mean": result.score_mean,
        "q_hat": result.q_hat,
        "finite": True,
        "failure_reason": "",
    }


def _base_row(
    config: IVRepairConfig,
    rep: int,
    data: PLRData,
    learner: str,
    method: str,
    score_id: str,
    result: Estimate,
) -> dict[str, Any]:
    return {
        "protocol_id": "thesis_iv_repair_v2",
        "experiment": config.design,
        "dgp": config.dgp,
        "n": config.n,
        "p": config.p,
        "replication": rep,
        "theta0": config.theta0,
        "learner": learner,
        "method": method,
        "score_id": score_id,
        "data_seed": data.data_seed,
        **_estimate_fields(result, config.theta0),
    }


def run_figure1_replication(config: IVRepairConfig, rep: int) -> list[dict[str, Any]]:
    if config.design != "figure1":
        raise ValueError("Figure 1 runner requires design='figure1'")
    config.validate()
    data = draw_data(config, rep)
    mismatch = estimate_outcome_mismatch(data.y, data.d, data.ell0)
    if not mismatch.finite:
        raise RuntimeError(f"mismatch baseline failed: {mismatch.failure_reason}")
    rows: list[dict[str, Any]] = []
    for learner in config.learners:
        nuisance = crossfit_structural_nuisances(config, learner, rep, data)
        iv_result = estimate_iv_type(data.y, data.d, nuisance.g_hat, nuisance.m_hat)
        if not iv_result.finite:
            raise RuntimeError(f"IV-type Figure 1 failed: {iv_result.failure_reason}")
        mismatch_row = _base_row(
            config,
            rep,
            data,
            learner,
            "outcome_nuisance_mismatch",
            "non_dml_outcome_nuisance_raw_d",
            mismatch,
        )
        mismatch_row.update(
            {
                "fit_mode": "oracle_outcome_nuisance",
                "fold_hash": nuisance.fold_hash,
                "theta_pilot_mean": float(np.mean(nuisance.theta_pilots)),
                "theta_pilot_min": min(nuisance.theta_pilots),
                "theta_pilot_max": max(nuisance.theta_pilots),
                "series_rank_m_min": nuisance.min_series_rank,
                "nuisance_rmse_g": 0.0,
                "nuisance_rmse_m": "",
                "self_influence_weight": 0.0,
            }
        )
        iv_row = _base_row(
            config,
            rep,
            data,
            learner,
            "iv_type_crossfit",
            "eq43_iv_type",
            iv_result,
        )
        iv_row.update(
            {
                "fit_mode": "crossfit",
                "fold_hash": nuisance.fold_hash,
                "theta_pilot_mean": float(np.mean(nuisance.theta_pilots)),
                "theta_pilot_min": min(nuisance.theta_pilots),
                "theta_pilot_max": max(nuisance.theta_pilots),
                "series_rank_m_min": nuisance.min_series_rank,
                "nuisance_rmse_g": float(
                    np.sqrt(np.mean((nuisance.g_hat - data.g0) ** 2))
                ),
                "nuisance_rmse_m": float(
                    np.sqrt(np.mean((nuisance.m_hat - data.m0) ** 2))
                ),
                "self_influence_weight": 0.0,
            }
        )
        rows.extend((mismatch_row, iv_row))
    expected = len(config.learners) * len(FIGURE1_METHODS)
    if len(rows) != expected or any(not bool(row["finite"]) for row in rows):
        raise RuntimeError("Figure 1 exact-cover/finite check failed")
    return rows


def run_figure2_replication(config: IVRepairConfig, rep: int) -> list[dict[str, Any]]:
    if config.design != "figure2":
        raise ValueError("Figure 2 runner requires design='figure2'")
    config.validate()
    data = draw_data(config, rep)
    weight = self_influence_weight(config)
    rows: list[dict[str, Any]] = []
    for learner in config.learners:
        full_g, full_m, full_pilot, full_rank = full_sample_structural_nuisances(
            config, learner, rep, data
        )
        crossfit = crossfit_structural_nuisances(config, learner, rep, data)
        base_full = estimate_iv_type(data.y, data.d, full_g, full_m)
        stressed_g = full_g + weight * (data.y - full_g)
        no_split = estimate_iv_type(data.y, data.d, stressed_g, full_m)
        crossfit_result = estimate_iv_type(
            data.y, data.d, crossfit.g_hat, crossfit.m_hat
        )
        if not base_full.finite or not no_split.finite or not crossfit_result.finite:
            raise RuntimeError("non-finite Figure 2 estimate")
        expected_stressed = (1.0 - weight) * base_full.theta
        if not math.isclose(no_split.theta, expected_stressed, rel_tol=2e-12, abs_tol=2e-12):
            raise RuntimeError("own-observation stress scaling identity failed")
        no_split_row = _base_row(
            config,
            rep,
            data,
            learner,
            "iv_type_no_split_stress",
            "eq43_iv_type",
            no_split,
        )
        no_split_row.update(
            {
                "fit_mode": "no_split",
                "fold_hash": crossfit.fold_hash,
                "theta_pilot_mean": full_pilot,
                "theta_pilot_min": full_pilot,
                "theta_pilot_max": full_pilot,
                "series_rank_m_min": full_rank,
                "nuisance_rmse_g": float(
                    np.sqrt(np.mean((stressed_g - data.g0) ** 2))
                ),
                "nuisance_rmse_m": float(np.sqrt(np.mean((full_m - data.m0) ** 2))),
                "self_influence_weight": weight,
                "unstressed_full_theta": base_full.theta,
                "stress_identity_error": no_split.theta - expected_stressed,
            }
        )
        crossfit_row = _base_row(
            config,
            rep,
            data,
            learner,
            "iv_type_crossfit",
            "eq43_iv_type",
            crossfit_result,
        )
        crossfit_row.update(
            {
                "fit_mode": "crossfit",
                "fold_hash": crossfit.fold_hash,
                "theta_pilot_mean": float(np.mean(crossfit.theta_pilots)),
                "theta_pilot_min": min(crossfit.theta_pilots),
                "theta_pilot_max": max(crossfit.theta_pilots),
                "series_rank_m_min": crossfit.min_series_rank,
                "nuisance_rmse_g": float(
                    np.sqrt(np.mean((crossfit.g_hat - data.g0) ** 2))
                ),
                "nuisance_rmse_m": float(
                    np.sqrt(np.mean((crossfit.m_hat - data.m0) ** 2))
                ),
                "self_influence_weight": 0.0,
                "unstressed_full_theta": base_full.theta,
                "stress_identity_error": 0.0,
            }
        )
        rows.extend((no_split_row, crossfit_row))
    expected = len(config.learners) * len(FIGURE2_FIT_MODES)
    if len(rows) != expected or any(not bool(row["finite"]) for row in rows):
        raise RuntimeError("Figure 2 exact-cover/finite check failed")
    return rows


def run_replication(config: IVRepairConfig, rep: int) -> list[dict[str, Any]]:
    if config.design == "figure1":
        return run_figure1_replication(config, rep)
    return run_figure2_replication(config, rep)


def run_cell(
    config: IVRepairConfig, rep_ids: Sequence[int] | None = None
) -> list[dict[str, Any]]:
    config.validate()
    selected = range(config.reps) if rep_ids is None else tuple(int(rep) for rep in rep_ids)
    if not selected or min(selected) < 0 or max(selected) >= config.reps:
        raise ValueError("replication IDs fall outside the configured universe")
    rows = [row for rep in selected for row in run_replication(config, rep)]
    rows_per_rep = len(config.learners) * 2
    if len(rows) != len(selected) * rows_per_rep:
        raise RuntimeError("cell exact-cover check failed")
    return rows


__all__ = [
    "CrossfitNuisances",
    "Estimate",
    "FIGURE1_METHODS",
    "FIGURE2_FIT_MODES",
    "IVRepairConfig",
    "LEARNERS",
    "PLRData",
    "SeriesOLS",
    "crossfit_series",
    "crossfit_structural_nuisances",
    "draw_data",
    "estimate_iv_type",
    "estimate_outcome_mismatch",
    "exact_series_basis",
    "fold_digest",
    "full_sample_structural_nuisances",
    "make_fold_ids",
    "nuisance_truth",
    "outer_fold_ids",
    "run_cell",
    "run_figure1_replication",
    "run_figure2_replication",
    "run_replication",
    "self_influence_weight",
    "stable_seed",
    "toeplitz_cov",
    "training_iv_pilot",
]
