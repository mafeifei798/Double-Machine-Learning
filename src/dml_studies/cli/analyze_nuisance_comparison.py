#!/usr/bin/env python3
"""Unfiltered, paired analysis with prespecified practical-zero criterion."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy import stats
try:
    from .study_utils import digest,mean_interval,wilson
except ImportError:
    from study_utils import digest,mean_interval,wilson

ROOT=Path(__file__).resolve().parents[3]


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
    (out/"ANALYSIS_LEDGER.json").write_text(json.dumps({p.name:digest(p) for p in out.iterdir() if p.is_file() and p.name!="ANALYSIS_LEDGER.json"},indent=2))
    print(json.dumps(dict(replications=len(ledger),estimates=len(rows),summary_cells=len(summaries),failures=len(failed),warnings=len(warnings)),indent=2))


if __name__=="__main__": analyze()
