#!/usr/bin/env python3
"""Unfiltered, paired analysis with prespecified practical-zero criterion."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy import stats
from analyze_fair_iv import digest,mean_interval,wilson

ROOT=Path(__file__).resolve().parents[1]


def analyze():
    root=ROOT/"output/formal"
    frozen=json.loads((root/"FROZEN_MANIFEST.json").read_text())
    ledger=json.loads((root/"RESULT_LEDGER.json").read_text())
    status=json.loads((root/"STATUS.json").read_text()); identity=frozen["identity"]
    protocol=identity["protocol"]
    assert status["status"]=="complete"
    assert hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()==frozen["identity_sha256"]
    for path,sha in identity["source_hashes"].items(): assert digest(ROOT/path)==sha,path
    assert digest(ROOT/"output/selection/SELECTED.json")==identity["selection_sha256"]
    expected={(d,g,r) for d in ("figure1","figure2") for g in ("linear","nonlinear") for r in range(protocol["replications"][d])}
    assert set(ledger)=={p.name for p in root.glob("figure*.json")}
    seen=set(); rows=[]; warnings=[]
    for path,sha in sorted(ledger.items()):
        assert digest(root/path)==sha
        receipt=json.loads((root/path).read_text()); key=tuple(receipt["key"])
        assert key in expected and key not in seen and receipt["identity_sha256"]==frozen["identity_sha256"]
        seen.add(key); rows.extend(receipt["value"]); warnings.extend(receipt["warnings"])
    assert seen==expected
    out=root/"analysis"; out.mkdir(exist_ok=True)
    failed=[r for r in rows if not r["finite"]]
    (out/"FAILURES.json").write_text(json.dumps(failed,indent=2))
    (out/"warnings.json").write_text(json.dumps(warnings,indent=2))
    if failed: raise RuntimeError("failed estimates retained; no complete-case success report")
    grouped=defaultdict(list)
    for row in rows: grouped[(row["design"],row["dgp"],row["track"],row["learner"],row["method"])].append(row)
    margin=protocol["bias_equivalence_margin"]; summaries=[]; contrasts=[]
    for key,group in sorted(grouped.items()):
        group.sort(key=lambda r:r["replication"])
        b=mean_interval([r["bias"] for r in group])
        summary=dict(design=key[0],dgp=key[1],track=key[2],learner=key[3],method=key[4],R=len(group),
            mean_theta=float(np.mean([r["theta"] for r in group])),bias=b,
            practical_zero_supported=b["low"]>-margin and b["high"]<margin,
            bias_detected=b["low"]>0 or b["high"]<0,
            mae=float(np.mean([r["abs_error"] for r in group])),
            rmse=float(np.sqrt(np.mean([r["squared_error"] for r in group]))),
            mean_se=float(np.mean([r["se"] for r in group])),
            mc_sd=float(np.std([r["theta"] for r in group],ddof=1)),
            coverage=wilson(sum(r["covers"] for r in group),len(group)),
            nuisance_rmse_g=float(np.mean([r["nuisance_rmse_g"] for r in group])),
            nuisance_rmse_m=float(np.mean([r["nuisance_rmse_m"] for r in group])))
        if "weighted_first_order" in group[0]:
            summary["weighted_components"]={k:mean_interval([r[k] for r in group]) for k in ("weighted_oracle","weighted_first_order","weighted_second_order")}
        summaries.append(summary)
    prefixes=sorted({key[:4] for key in grouped})
    for prefix in prefixes:
        cf=grouped[(*prefix,"crossfit")]
        baselines=["full_no_split","equal_train_reuse","nonorthogonal_crossfit"]
        if (*prefix,"outcome_mismatch_crossfit") in grouped: baselines.append("outcome_mismatch_crossfit")
        for baseline in baselines:
            other=grouped[(*prefix,baseline)]
            assert [r["replication"] for r in other]==[r["replication"] for r in cf]
            contrasts.append(dict(design=prefix[0],dgp=prefix[1],track=prefix[2],learner=prefix[3],baseline=baseline,
                comparator="crossfit",signed_bias_difference=mean_interval([b["bias"]-a["bias"] for a,b in zip(other,cf)]),
                mse_difference=mean_interval([b["squared_error"]-a["squared_error"] for a,b in zip(other,cf)]),
                absolute_mean_bias_difference=abs(np.mean([b["bias"] for b in cf]))-abs(np.mean([a["bias"] for a in other]))))
    (out/"summary.json").write_text(json.dumps(summaries,indent=2,allow_nan=False))
    (out/"paired_contrasts.json").write_text(json.dumps(contrasts,indent=2,allow_nan=False))
    report=["# IV-only repair v3: independent confirmation", "",
        "Original four DGPs; n=1000; two folds; all replications and all methods retained.",
        "Controlled Series OLS treatment nuisance is correctly specified. Mixed RF outcome / Series OLS m is NOT pure RF.",
        "Ordinary RF/XGB/Lasso fit both nuisances using prediction-selected parameters and a training-only rich-series pilot.",
        "All final DML estimates use structural IV-type score. No artificial pressure or oracle training.", "",
        "Practical-zero support means the entire MC95 mean-bias interval is inside the prespecified ±0.01 margin; not exact unbiasedness.",
        "Bias detection (interval excludes zero) and practical equivalence answer different questions; both may be true.",
        "Coverage uses usual asymptotic IV-score standard errors; No-split intervals are descriptive, not guaranteed valid.", "",
        "| Design | DGP | Track | Learner | Method | R | Mean | Bias [MC95] | RMSE | Coverage | Practical zero |",
        "|---|---|---|---|---|---:|---:|---|---:|---:|---|"]
    for s in summaries:
        b=s["bias"]
        report.append(f"| {s['design']} | {s['dgp']} | {s['track']} | {s['learner']} | {s['method']} | {s['R']} | {s['mean_theta']:.6f} | {b['mean']:+.6f} [{b['low']:+.6f},{b['high']:+.6f}] | {s['rmse']:.6f} | {s['coverage']['rate']:.3f} | {s['practical_zero_supported']} |")
    report += ["",f"Replications: {len(ledger)}; estimates: {len(rows)}; failures: {len(failed)}; warnings: {len(warnings)}.",
        "All paired contrasts and weighted decompositions remain in JSON, including adverse outcomes.",
        "No formal outcome is used for learner/parameter selection. Historical runs are unchanged."]
    (out/"RESULTS.md").write_text("\n".join(report)+"\n")
    plot(out,grouped,summaries)
    (out/"ANALYSIS_LEDGER.json").write_text(json.dumps({p.name:digest(p) for p in out.iterdir() if p.is_file() and p.name!="ANALYSIS_LEDGER.json"},indent=2))
    print(json.dumps(dict(replications=len(ledger),estimates=len(rows),summary_cells=len(summaries),failures=len(failed),warnings=len(warnings)),indent=2))


def plot(out,grouped,summaries):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lookup={(s["design"],s["dgp"],s["track"],s["learner"],s["method"]):s for s in summaries}
    def histogram(design,dgp,track,learners,left,filename,heading):
        fig,axes=plt.subplots(len(learners),2,figsize=(12,3.3*len(learners)+1),squeeze=False)
        for i,learner in enumerate(learners):
            values=[np.array([r["bias"]/r["se"] for r in grouped[(design,dgp,track,learner,m)]]) for m in (left,"crossfit")]
            lo=min(-4.,*(float(v.min()) for v in values)); hi=max(4.,*(float(v.max()) for v in values))
            bins=np.linspace(lo-.1,hi+.1,37); x=np.linspace(lo,hi,600)
            for j,method in enumerate((left,"crossfit")):
                ax=axes[i,j]; ax.hist(values[j],bins=bins,density=True,color="steelblue",edgecolor="white",alpha=.8)
                ax.plot(x,stats.norm.pdf(x),color="red",label="N(0,1)"); ax.axvline(0,color="black",linewidth=.8)
                ax.set_xlim(lo-.1,hi+.1); ax.legend(fontsize=8)
                s=lookup[(design,dgp,track,learner,method)]
                ax.set_title(f"{learner} | {method}\nmean={s['mean_theta']:.4f}; bias={s['bias']['mean']:+.4f}; coverage={s['coverage']['rate']:.3f}")
                ax.set_xlabel("(estimate - true effect) / estimated SE"); ax.set_ylabel("Density")
        fig.suptitle(f"{heading}\n{design}, {dgp}, true effect=0.5, n=1000; all replications retained",fontsize=12)
        fig.tight_layout(rect=(0,0,1,.92 if len(learners)==1 else .96)); fig.savefig(out/filename,dpi=160); plt.close(fig)
    for dgp in ("linear","nonlinear"):
        histogram("figure1",dgp,"ordinary",("rf","xgb","lasso"),"nonorthogonal_crossfit",
            f"ordinary_figure1_{dgp}.png","Prediction-selected RF/XGB/Lasso BOTH nuisances; rich-series training pilot")
        histogram("figure2",dgp,"ordinary",("rf","xgb","lasso"),"full_no_split",
            f"ordinary_figure2_{dgp}.png","Ordinary learned nuisances; no injected overfitting")
        histogram("figure2",dgp,"ordinary",("rf","xgb","lasso"),"equal_train_reuse",
            f"ordinary_equal_training_{dgp}.png","Exact same 500-row model bank; reuse vs held-out prediction")
        histogram("figure1",dgp,"controlled",("series_ols",),"outcome_mismatch_crossfit",
            f"controlled_figure1_{dgp}.png","Controlled correctly specified Series OLS; mismatch vs structural IV")
        histogram("figure2",dgp,"controlled",tuple(f"tree_leaf_{l}" for l in (2,5,10,20)),"full_no_split",
            f"controlled_figure2_{dgp}.png","Genuine Decision Tree outcome; correctly specified Series OLS m")
        histogram("figure2",dgp,"controlled",("rf_outcome_series_m",),"full_no_split",
            f"mixed_rf_series_{dgp}.png","Mixed learner sensitivity: RF outcome + Series OLS m, NOT pure RF")
    for track in ("ordinary","controlled"):
        subset=[s for s in summaries if s["track"]==track and s["method"] in ("full_no_split","equal_train_reuse","crossfit")]
        labels=sorted({(s["design"],s["dgp"],s["learner"]) for s in subset})
        fig,ax=plt.subplots(figsize=(11,max(5,len(labels)*.45)))
        for offset,method,color in ((-.18,"full_no_split","gray"),(0,"equal_train_reuse","orange"),(.18,"crossfit","steelblue")):
            cells=[lookup[(*key[:2],track,key[2],method)] for key in labels]
            means=np.array([s["bias"]["mean"] for s in cells]); low=np.array([s["bias"]["low"] for s in cells]); high=np.array([s["bias"]["high"] for s in cells])
            ax.errorbar(means,np.arange(len(labels))+offset,xerr=np.vstack((means-low,high-means)),fmt="o",color=color,label=method,capsize=2)
        ax.axvspan(-.01,.01,color="green",alpha=.08,label="prespecified practical-zero margin")
        ax.axvline(0,color="black",linewidth=.8); ax.set_yticks(np.arange(len(labels)),[" / ".join(k) for k in labels])
        ax.set_xlabel("Signed mean bias, Monte Carlo 95% interval"); ax.set_title(f"{track}: all IV comparisons, n=1000")
        ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out/f"bias_{track}.png",dpi=160); plt.close(fig)


if __name__=="__main__": analyze()
