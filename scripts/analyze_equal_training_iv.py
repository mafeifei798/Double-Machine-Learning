#!/usr/bin/env python3
"""Complete paired reporting for the frozen shared-bank experiment."""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats
from analyze_fair_iv import digest, mean_interval, wilson

METHODS=("equal_train_reuse","equal_train_crossfit")


def load_verified(root):
    frozen=json.loads((root/"FROZEN_MANIFEST.json").read_text())
    ledger=json.loads((root/"RESULT_LEDGER.json").read_text())
    status=json.loads((root/"STATUS.json").read_text())
    identity=frozen["identity"]; protocol=identity["protocol"]
    if status["status"]!="complete": raise RuntimeError("incomplete run")
    if hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()!=frozen["identity_sha256"]:
        raise RuntimeError("invalid frozen identity")
    for rel,sha in identity["source_hashes"].items():
        if digest(root.parents[1]/rel)!=sha: raise RuntimeError(f"source changed: {rel}")
    expected={(d,g,r) for d in ("figure1","figure2") for g in ("linear","nonlinear")
              for r in range(2 if identity["development"] else protocol["replications"][d])}
    disk={str(p.relative_to(root)) for p in root.glob("figure*_*/*rep_*.json")}
    if disk!=set(ledger) or len(ledger)!=len(expected): raise RuntimeError("ledger universe")
    rows=[]; warnings=[]; seen=set()
    for rel,sha in sorted(ledger.items()):
        path=root/rel
        if digest(path)!=sha: raise RuntimeError("result hash")
        artifact=json.loads(path.read_text()); key=tuple(artifact["key"])
        if key in seen or key not in expected or artifact["identity_sha256"]!=frozen["identity_sha256"]:
            raise RuntimeError("replication identity")
        seen.add(key); cell=artifact["rows"]
        target={(l,m) for l in protocol["learners"] for m in METHODS}
        if len(cell)!=len(target) or {(r["learner"],r["method"]) for r in cell}!=target:
            raise RuntimeError("cell exact cover")
        for r in cell:
            cross=r["method"]==METHODS[1]
            if (r["design"],r["dgp"],r["replication"])!=key or r["artificial_pressure"]:
                raise RuntimeError("row key/stress")
            if r["train_n_per_model"]!=protocol["n"]//2 or r["evaluation_n"]!=protocol["n"]:
                raise RuntimeError("unequal training/evaluation sizes")
            if r["pilot_mode"]!=protocol["pilot"] or r["fit_calls_per_bank"]!=6 or r["every_row_held_out"]!=cross:
                raise RuntimeError("pilot/fitting contract")
            if r["finite"]:
                numeric=("theta","se","bias","abs_error","squared_error","score_mean","jacobian",
                         "score_at_truth","oracle_score","first_order_score","second_order_score")
                if not all(math.isfinite(r[k]) for k in numeric) or r["se"]<=0: raise RuntimeError("nonfinite")
                error=r["theta"]-r["theta0"]
                if abs(error-r["bias"])>1e-12 or abs(abs(error)-r["abs_error"])>1e-12 or abs(error**2-r["squared_error"])>1e-12:
                    raise RuntimeError("error identities")
                if r["covers"]!=(abs(error)<=1.96*r["se"]) or abs(r["score_mean"])>1e-10:
                    raise RuntimeError("SE/score identity")
                if abs(r["score_at_truth"]-sum(r[k] for k in ("oracle_score","first_order_score","second_order_score")))>1e-12:
                    raise RuntimeError("decomposition identity")
                if r["prediction_overlap_n"]!=(0 if cross else protocol["n"]): raise RuntimeError("overlap contract")
            elif not r["failure_reason"]: raise RuntimeError("unexplained failure")
        for learner in protocol["learners"]:
            pair={r["method"]:r for r in cell if r["learner"]==learner}; a,b=(pair[m] for m in METHODS)
            if a["finite"] and b["finite"]:
                for k in ("bank_identity","bank_records","pilots","data_seed","fold_hash"):
                    if a[k]!=b[k]: raise RuntimeError(f"unpaired bank: {k}")
                banksha=hashlib.sha256(json.dumps(a["bank_records"],sort_keys=True).encode()).hexdigest()
                if banksha!=a["bank_identity"] or any(x["train_n"]!=protocol["n"]//2 for x in a["bank_records"]):
                    raise RuntimeError("bank receipt")
        rows.extend(cell); warnings.extend(artifact["warnings"])
    if seen!=expected: raise RuntimeError("missing replication")
    return frozen,rows,warnings


def analyze(root):
    frozen,rows,warnings=load_verified(root)
    analysis=root/"analysis"; analysis.mkdir(exist_ok=True)
    failed=[r for r in rows if not r["finite"]]
    if failed:
        (analysis/"FAILURES.json").write_text(json.dumps(failed,indent=2))
        raise RuntimeError("failures retained; complete report blocked")
    groups=defaultdict(list)
    for r in rows: groups[(r["design"],r["dgp"],r["learner"],r["method"])].append(r)
    summaries=[]; pairs=[]
    for key,group in sorted(groups.items()):
        group.sort(key=lambda r:r["replication"])
        summaries.append(dict(design=key[0],dgp=key[1],learner=key[2],method=key[3],R=len(group),
            mean_theta=float(np.mean([r["theta"] for r in group])),bias=mean_interval([r["bias"] for r in group]),
            mc_sd=float(np.std([r["theta"] for r in group],ddof=1)),
            mae=float(np.mean([r["abs_error"] for r in group])),rmse=float(np.sqrt(np.mean([r["squared_error"] for r in group]))),
            mean_se=float(np.mean([r["se"] for r in group])),coverage=wilson(sum(r["covers"] for r in group),len(group)),
            diagnostics={k:mean_interval([r[k] for r in group]) for k in
                         ("nuisance_rmse_g","nuisance_rmse_m","oracle_score","first_order_score","second_order_score","score_at_truth")}))
    protocol=frozen["identity"]["protocol"]
    for d in ("figure1","figure2"):
        for g in ("linear","nonlinear"):
            for learner in protocol["learners"]:
                a,b=(groups[(d,g,learner,m)] for m in METHODS)
                if [r["replication"] for r in a]!=[r["replication"] for r in b]: raise RuntimeError("unpaired IDs")
                pairs.append(dict(design=d,dgp=g,learner=learner,direction="crossfit minus equal-training reuse",
                    signed_bias_difference=mean_interval([y["bias"]-x["bias"] for x,y in zip(a,b)]),
                    mae_difference=mean_interval([y["abs_error"]-x["abs_error"] for x,y in zip(a,b)]),
                    mse_difference=mean_interval([y["squared_error"]-x["squared_error"] for x,y in zip(a,b)]),
                    absolute_mean_bias_difference=float(abs(np.mean([r["bias"] for r in b]))-abs(np.mean([r["bias"] for r in a])))))
    aggregates=[]; rng=np.random.default_rng(protocol["bootstrap_seed"])
    for d in ("figure1","figure2"):
        matrices=[np.array([[[r["bias"] for r in groups[(d,g,l,m)]] for m in METHODS] for l in protocol["learners"]])
                  for g in ("linear","nonlinear")]
        observed=np.mean([np.abs(a.mean(axis=2)).mean(axis=0) for a in matrices],axis=0)
        boot=[]
        for _ in range(protocol["bootstrap_reps"]):
            v=np.mean([np.abs(a[:,:,rng.integers(0,a.shape[2],a.shape[2])].mean(axis=2)).mean(axis=0) for a in matrices],axis=0)
            boot.append(float(v[1]-v[0]))
        aggregates.append(dict(design=d,reuse_absolute_mean_bias=float(observed[0]),crossfit_absolute_mean_bias=float(observed[1]),
                               difference=float(observed[1]-observed[0]),paired_block_bootstrap95=list(map(float,np.quantile(boot,[.025,.975])))))
    for name,value in (("summary.json",summaries),("paired_contrasts.json",pairs),("aggregate_bias.json",aggregates),("warnings.json",warnings)):
        (analysis/name).write_text(json.dumps(value,indent=2,allow_nan=False))
    report=["# Equal-training IV-type: complete paired results","",
            "Development only." if frozen["identity"]["development"] else "Formal follow-up; all planned replication IDs retained and hashes verified.",
            "","Same data and same trained A/B model bank for both arms. Each model trains on 500 observations; both arms estimate on 1000.",
            "Training-half-only pilots are shared by both arms. No artificial pressure, oracle fitting or Series OLS replacement.",
            "Reuse is an equal-training mechanism comparator, NOT ordinary full-sample No-split. This crossfit pilot differs from the earlier official pooled pilot.",
            "Bias intervals describe Monte Carlo uncertainty. Nominal sandwich coverage is descriptive, not assumed valid for reuse.","",
            "| Design | DGP | Learner | Arm | R | Mean | Bias [MC95] | RMSE | MAE | Nominal coverage |",
            "|---|---|---|---|---:|---:|---|---:|---:|---:|"]
    for s in summaries:
        b=s["bias"]; report.append(f"| {s['design']} | {s['dgp']} | {s['learner']} | {s['method']} | {s['R']} | {s['mean_theta']:.6f} | {b['mean']:+.6f} [{b['low']:+.6f}, {b['high']:+.6f}] | {s['rmse']:.6f} | {s['mae']:.6f} | {s['coverage']['rate']:.3f} |")
    report.extend(["","## Equal-weight absolute Monte Carlo mean bias","","Average of six learner/DGP absolute mean biases per design; NOT MAE. Negative differences favor cross-fitting.",""])
    for a in aggregates:
        report.append(f"- {a['design']}: reuse {a['reuse_absolute_mean_bias']:.6f}; crossfit {a['crossfit_absolute_mean_bias']:.6f}; difference {a['difference']:+.6f}, paired bootstrap95 {a['paired_block_bootstrap95']}.")
    report.extend(["","## Paired contrasts","","| Design | DGP | Learner | CF-minus-reuse bias [MC95] | MAE difference [MC95] | MSE difference [MC95] |",
                   "|---|---|---|---|---|---|"])
    for p in pairs:
        cells=[f"{p[k]['mean']:+.6f} [{p[k]['low']:+.6f}, {p[k]['high']:+.6f}]" for k in ("signed_bias_difference","mae_difference","mse_difference")]
        report.append(f"| {p['design']} | {p['dgp']} | {p['learner']} | "+" | ".join(cells)+" |")
    report.extend(["","All nuisance RMSE and score-decomposition means/intervals are in summary.json. They are descriptive diagnostics only.",
                   "No cells, seeds or hyperparameters are selected based on the results; no exact-unbiasedness or universal robustness claim is implied.",
                   f"Rows: {len(rows)}; failures: {len(failed)}; warnings: {len(warnings)}."])
    (analysis/"RESULTS.md").write_text("\n".join(report)+"\n")
    plot(root,groups,summaries)
    ledger={str(p.relative_to(root)):digest(p) for p in analysis.iterdir() if p.is_file() and p.name!="ANALYSIS_LEDGER.json"}
    (analysis/"ANALYSIS_LEDGER.json").write_text(json.dumps(ledger,indent=2))
    print(json.dumps(dict(rows=len(rows),summary_cells=len(summaries),failed=len(failed),warnings=len(warnings),aggregates=aggregates),indent=2))


def plot(root,groups,summaries):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lookup={(s["design"],s["dgp"],s["learner"],s["method"]):s for s in summaries}
    labels=("Equal-training reuse","Equal-training crossfit")
    for d in ("figure1","figure2"):
        for g in ("linear","nonlinear"):
            fig,axes=plt.subplots(3,2,figsize=(12,10))
            for i,l in enumerate(("rf","xgb","lasso")):
                vals=[np.array([r["bias"]/r["se"] for r in groups[(d,g,l,m)]]) for m in METHODS]
                lo=min(-4.,*(float(v.min()) for v in vals)); hi=max(4.,*(float(v.max()) for v in vals))
                bins=np.linspace(lo-.1,hi+.1,37)
                for j,m in enumerate(METHODS):
                    ax=axes[i,j]; ax.hist(vals[j],bins=bins,density=True,color="steelblue",alpha=.8,edgecolor="white")
                    x=np.linspace(lo,hi,500); ax.plot(x,stats.norm.pdf(x),color="red"); ax.axvline(0,color="black",lw=1)
                    ax.set_xlim(lo-.1,hi+.1); s=lookup[(d,g,l,m)]
                    ax.set_title(f"{l.upper()} | {labels[j]}\nmean={s['mean_theta']:.4f}; bias={s['bias']['mean']:+.4f}; coverage={s['coverage']['rate']:.3f}")
                    ax.set_xlabel("(estimate - true effect) / estimated SE"); ax.set_ylabel("Density")
            fig.suptitle(f"{d} | {g} | Shared model bank; no artificial pressure\nTrue effect=0.5; train/model=500; estimate/arm=1000; all replications retained",fontsize=13)
            fig.tight_layout(rect=(0,0,1,.94)); fig.savefig(root/"analysis"/f"{d}_{g}.png",dpi=160); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(14,6),sharey=True)
    categories=[(l,g) for l in ("rf","xgb","lasso") for g in ("linear","nonlinear")]
    for ax,d in zip(axes,("figure1","figure2")):
        for j,m in enumerate(METHODS):
            ss=[lookup[(d,g,l,m)] for l,g in categories]; y=np.arange(6)+(j-.5)*.18
            x=np.array([s["bias"]["mean"] for s in ss])
            err=np.array([[s["bias"]["mean"]-s["bias"]["low"] for s in ss],
                          [s["bias"]["high"]-s["bias"]["mean"] for s in ss]])
            ax.errorbar(x,y,xerr=err,fmt="o",capsize=3,label=labels[j])
        ax.axvline(0,color="black",lw=1); ax.set_yticks(np.arange(6),[f"{l.upper()} / {g}" for l,g in categories])
        ax.set_title(d); ax.set_xlabel("Signed mean bias with Monte Carlo 95% interval"); ax.legend()
    axes[0].invert_yaxis(); fig.suptitle("Equal training size and shared fitted models; true effect=0.5")
    fig.tight_layout(); fig.savefig(root/"analysis"/"bias_comparison.png",dpi=160); plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--root",type=Path,required=True)
    analyze(parser.parse_args().root.resolve())
