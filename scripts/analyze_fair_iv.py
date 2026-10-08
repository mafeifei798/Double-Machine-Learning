#!/usr/bin/env python3
"""Read-only verification and complete, unfiltered Monte Carlo reporting."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

METHODS = ("nonorthogonal_crossfit", "iv_no_split", "iv_crossfit")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean_interval(values):
    x = np.asarray(values, dtype=float)
    mean = float(x.mean())
    half = float(stats.t.ppf(.975,len(x)-1)*x.std(ddof=1)/math.sqrt(len(x))) if len(x)>1 else None
    return dict(mean=mean,low=mean-half if half is not None else None,
                high=mean+half if half is not None else None)


def wilson(success,total):
    z = float(stats.norm.ppf(.975)); p=success/total
    center = (p+z*z/(2*total))/(1+z*z/total)
    half = z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/(1+z*z/total)
    return dict(rate=p, low=center-half, high=center+half)


def load_verified(root):
    frozen=json.loads((root/"FROZEN_MANIFEST.json").read_text())
    ledger=json.loads((root/"RESULT_LEDGER.json").read_text())
    status=json.loads((root/"STATUS.json").read_text())
    identity=frozen["identity"]; protocol=identity["protocol"]
    if status["status"] != "complete":
        raise RuntimeError("not a complete run; no final comparison")
    expected={(design,dgp,rep) for design in ("figure1","figure2") for dgp in ("linear","nonlinear")
              for rep in range(min(identity["development_reps"],protocol["replications"][design])
                               if identity["development"] else protocol["replications"][design])}
    seen=set(); rows=[]; warnings=[]
    disk={str(p.relative_to(root)) for p in root.glob("figure*_*/*rep_*.json")}
    if disk != set(ledger):
        raise RuntimeError("replication ledger exact-cover mismatch")
    for rel,sha in sorted(ledger.items()):
        path=root/rel
        if digest(path) != sha:
            raise RuntimeError(f"result hash mismatch: {rel}")
        artifact=json.loads(path.read_text()); key=tuple(artifact["key"])
        if key not in expected or key in seen or artifact["identity_sha256"] != frozen["identity_sha256"]:
            raise RuntimeError(f"duplicate/incompatible replication: {rel}")
        seen.add(key)
        cells={(r["learner"],r["method"]) for r in artifact["rows"]}
        target={(l,m) for l in protocol["learners"] for m in METHODS}
        if cells != target or len(artifact["rows"]) != len(target):
            raise RuntimeError("method/learner exact cover mismatch")
        for row in artifact["rows"]:
            if (row["design"],row["dgp"],row["replication"]) != key or row["artificial_pressure"]:
                raise RuntimeError("row identity/stress contract mismatch")
            if row["treatment_learner"] != row["learner"] or row["protocol_id"] != protocol["protocol_id"]:
                raise RuntimeError("wrong treatment learner/protocol")
            if row["finite"]:
                if not all(math.isfinite(row[k]) for k in ("theta","se","bias","abs_error","squared_error","pilot","score_mean")):
                    raise RuntimeError("nonfinite marked finite")
                if row["se"] <= 0 or abs(row["score_mean"]) > 1e-10:
                    raise RuntimeError("invalid score/SE")
                error=row["theta"]-row["theta0"]
                if abs(row["bias"]-error)>1e-12 or abs(row["abs_error"]-abs(error))>1e-12:
                    raise RuntimeError("bias/MAE identity failed")
                if abs(row["squared_error"]-error**2)>1e-12 or row["covers"] != (abs(error)<=1.96*row["se"]):
                    raise RuntimeError("MSE/coverage identity failed")
            elif not row["failure_reason"]:
                raise RuntimeError("unexplained numerical failure")
        for learner in protocol["learners"]:
            pair={r["method"]:r for r in artifact["rows"] if r["learner"]==learner}
            a,b=pair["nonorthogonal_crossfit"],pair["iv_crossfit"]
            if a["finite"] and b["finite"] and a["g_prediction_hash"] != b["g_prediction_hash"]:
                raise RuntimeError("naive/IV did not share structural g")
        rows.extend(artifact["rows"])
        warnings.extend(artifact["warnings"])
    if seen != expected:
        raise RuntimeError("missing/unexpected replication IDs")
    return frozen,rows,warnings


def analyze(root):
    frozen,rows,warnings=load_verified(root)
    analysis=root/"analysis"; analysis.mkdir(exist_ok=True)
    failures=[r for r in rows if not r["finite"]]
    if failures:
        (analysis/"FAILURES.json").write_text(json.dumps(failures,indent=2))
        raise RuntimeError(f"{len(failures)} failed estimates: retained; final comparison blocked")
    grouped=defaultdict(list)
    for row in rows: grouped[(row["design"],row["dgp"],row["learner"],row["method"])].append(row)
    summaries=[]; contrasts=[]
    for key,group in sorted(grouped.items()):
        group.sort(key=lambda r:r["replication"])
        summaries.append(dict(design=key[0],dgp=key[1],learner=key[2],method=key[3],
                              R=len(group),mean_theta=float(np.mean([r["theta"] for r in group])),
                              bias=mean_interval([r["bias"] for r in group]),
                              mc_sd=float(np.std([r["theta"] for r in group],ddof=1)),
                              mae=float(np.mean([r["abs_error"] for r in group])),
                              rmse=float(np.sqrt(np.mean([r["squared_error"] for r in group]))),
                              mean_se=float(np.mean([r["se"] for r in group])),
                              coverage=wilson(sum(r["covers"] for r in group),len(group))))
    for design in ("figure1","figure2"):
        for dgp in ("linear","nonlinear"):
            for learner in frozen["identity"]["protocol"]["learners"]:
                for baseline in ("nonorthogonal_crossfit","iv_no_split"):
                    a=grouped[(design,dgp,learner,baseline)]; b=grouped[(design,dgp,learner,"iv_crossfit")]
                    if [r["replication"] for r in a] != [r["replication"] for r in b]:
                        raise RuntimeError("unpaired contrast")
                    # Negatives favor cross-fitting; signed bias differences
                    # alone do not establish reduction in absolute bias.
                    contrasts.append(dict(design=design,dgp=dgp,learner=learner,baseline=baseline,
                        comparator="iv_crossfit",direction="crossfit minus baseline",
                        signed_bias_difference=mean_interval([rb["bias"]-ra["bias"] for ra,rb in zip(a,b)]),
                        mae_difference=mean_interval([rb["abs_error"]-ra["abs_error"] for ra,rb in zip(a,b)]),
                        mse_difference=mean_interval([rb["squared_error"]-ra["squared_error"] for ra,rb in zip(a,b)]),
                        absolute_mean_bias_difference=abs(np.mean([r["bias"] for r in b]))-abs(np.mean([r["bias"] for r in a]))))
    # Equal weighting across learner x DGP cells; not pooled MAE nor mean of
    # per-cell percentage reductions. Bootstrap replication blocks jointly
    # across learners to retain their shared-data dependence.
    aggregates=[]
    rng=np.random.default_rng(2026100401)
    for design in ("figure1","figure2"):
        for baseline in ("nonorthogonal_crossfit","iv_no_split"):
            matrices=[]
            for dgp in ("linear","nonlinear"):
                matrices.append(np.array([[[r["bias"] for r in grouped[(design,dgp,l,m)]]
                                           for m in (baseline,"iv_crossfit")]
                                          for l in frozen["identity"]["protocol"]["learners"]]))
            observed=np.mean([np.abs(a.mean(axis=2)).mean(axis=0) for a in matrices],axis=0)
            draws=[]
            for _ in range(2000):
                values=np.mean([np.abs(a[:,:,rng.integers(0,a.shape[2],size=a.shape[2])].mean(axis=2)).mean(axis=0)
                                for a in matrices],axis=0)
                draws.append(float(100*(1-values[1]/values[0])))
            aggregates.append(dict(design=design,baseline=baseline,comparator="iv_crossfit",
                baseline_absolute_mean_bias=float(observed[0]),crossfit_absolute_mean_bias=float(observed[1]),
                reduction_percent=float(100*(1-observed[1]/observed[0])),
                paired_block_bootstrap_percentile95=list(map(float,np.quantile(draws,[.025,.975])))))
    (analysis/"summary.json").write_text(json.dumps(summaries,indent=2,allow_nan=False))
    (analysis/"paired_contrasts.json").write_text(json.dumps(contrasts,indent=2,allow_nan=False))
    (analysis/"aggregate_bias.json").write_text(json.dumps(aggregates,indent=2,allow_nan=False))
    (analysis/"warnings.json").write_text(json.dumps(warnings,indent=2))
    report=["# Fair IV-type comparison: complete results", "",
            "DEVELOPMENT ONLY; not formal evidence." if frozen["identity"]["development"] else "FORMAL: fixed configuration; all replication IDs and result hashes verified.",
            "", "No artificial residual pressure; learned RF/XGB/Lasso for both nuisance channels.",
            "Original DGPs and hyperparameters; n=1,000; two folds. The official pooled initial scalar is disclosed in the protocol.",
            "", "Bias is signed Monte Carlo mean error, not MAE. Intervals below are Monte Carlo uncertainty for the mean bias, not individual-estimate confidence intervals.", "",
            "| Design | DGP | Learner | Method | R | Mean | Bias [MC 95% CI] | RMSE | MAE | Coverage [Wilson 95% CI] |",
            "|---|---|---|---|---:|---:|---|---:|---:|---|"]
    for s in summaries:
        b,c=s["bias"],s["coverage"]
        report.append(f"| {s['design']} | {s['dgp']} | {s['learner']} | {s['method']} | {s['R']} | {s['mean_theta']:.6f} | {b['mean']:+.6f} [{b['low']:+.6f}, {b['high']:+.6f}] | {s['rmse']:.6f} | {s['mae']:.6f} | {c['rate']:.3f} [{c['low']:.3f}, {c['high']:.3f}] |")
    report.extend(["", "## Equal-weight absolute-mean-bias contrasts", "",
                   "A negative reduction means the cross-fit estimator had larger absolute mean bias. Do not select favorable cells or reinterpret this as universal robustness.", ""])
    for a in aggregates:
        report.append(f"- {a['design']}, {a['baseline']} vs IV cross-fit: {a['baseline_absolute_mean_bias']:.6f} -> {a['crossfit_absolute_mean_bias']:.6f}; reduction {a['reduction_percent']:.2f}% (paired block-bootstrap 95% interval {a['paired_block_bootstrap_percentile95'][0]:.2f}% to {a['paired_block_bootstrap_percentile95'][1]:.2f}%).")
    report.extend(["", "Every per-cell paired MAE/MSE/bias contrast, including adverse results, is in paired_contrasts.json.",
                   "The bootstrap interval is descriptive finite-Monte-Carlo uncertainty, not a generalization guarantee. No scientific success gate is based on obtaining four desired conclusions.",
                   f"", f"Rows: {len(rows)}; failed estimates: {len(failures)}; recorded warnings: {len(warnings)}."])
    (analysis/"RESULTS.md").write_text("\n".join(report)+"\n")
    plot(root,grouped,summaries)
    outputs={str(p.relative_to(root)):digest(p) for p in analysis.iterdir() if p.is_file() and p.name!="ANALYSIS_LEDGER.json"}
    (analysis/"ANALYSIS_LEDGER.json").write_text(json.dumps(outputs,indent=2))
    print(json.dumps(dict(rows=len(rows),summary_cells=len(summaries),failures=len(failures),aggregates=aggregates),indent=2))


def plot(root,grouped,summaries):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lookup={(s["design"],s["dgp"],s["learner"],s["method"]):s for s in summaries}
    for design in ("figure1","figure2"):
        left="nonorthogonal_crossfit" if design=="figure1" else "iv_no_split"
        for dgp in ("linear","nonlinear"):
            fig,axes=plt.subplots(3,2,figsize=(12,10))
            for i,learner in enumerate(("rf","xgb","lasso")):
                values=[np.array([r["bias"]/r["se"] for r in grouped[(design,dgp,learner,m)]]) for m in (left,"iv_crossfit")]
                lo=min(-4.,*(float(v.min()) for v in values)); hi=max(4.,*(float(v.max()) for v in values))
                bins=np.linspace(lo-.1,hi+.1,37)
                for j,method in enumerate((left,"iv_crossfit")):
                    ax=axes[i,j]; ax.hist(values[j],bins=bins,density=True,color="steelblue",alpha=.8,edgecolor="white")
                    x=np.linspace(lo,hi,500); ax.plot(x,stats.norm.pdf(x),color="red",label="N(0,1)")
                    ax.axvline(0,color="black",linewidth=1); ax.set_xlim(lo-.1,hi+.1)
                    s=lookup[(design,dgp,learner,method)]
                    ax.set_title(f"{learner.upper()} | {method}\nmean={s['mean_theta']:.4f}; bias={s['bias']['mean']:+.4f}; coverage={s['coverage']['rate']:.3f}")
                    ax.set_xlabel("(estimate - true effect) / estimated SE"); ax.set_ylabel("Density")
            fig.suptitle(f"{design} | {dgp} | Ordinary learned nuisances; no artificial pressure\nTrue effect=0.5, n=1000, 2-fold; all replications retained",fontsize=13)
            fig.tight_layout(rect=(0,0,1,.94))
            fig.savefig(root/"analysis"/f"{design}_{dgp}.png",dpi=160)
            plt.close(fig)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--root",type=Path,required=True)
    analyze(parser.parse_args().root.resolve())
