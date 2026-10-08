#!/usr/bin/env python3
"""Independent read-only check: no fitted learners and no imported run code."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

from scipy.stats import t


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def near(actual,expected):
    assert math.isfinite(actual) and math.isclose(actual,expected,rel_tol=2e-12,abs_tol=2e-12), (actual,expected)


def verify(root):
    code_root=root.parents[1]
    frozen=read(root/"FROZEN_MANIFEST.json")
    ident=frozen["identity"]; protocol=ident["protocol"]
    assert not ident["development"] and protocol["artificial_pressure"] is False
    assert frozen["identity_sha256"] == hashlib.sha256(json.dumps(ident,sort_keys=True).encode()).hexdigest()
    for rel,value in ident["source_hashes"].items():
        assert sha(code_root/rel)==value,rel
    assert read(root/"STATUS.json")["status"]=="complete"
    ledger=read(root/"RESULT_LEDGER.json")
    universe={(design,dgp,r) for design in ("figure1","figure2") for dgp in ("linear","nonlinear")
              for r in range(protocol["replications"][design])}
    seen=set(); groups=defaultdict(dict)
    methods={"nonorthogonal_crossfit","iv_no_split","iv_crossfit"}
    for rel,digest in ledger.items():
        assert sha(root/rel)==digest
        block=read(root/rel); key=tuple(block["key"])
        assert key in universe and key not in seen
        assert block["identity_sha256"]==frozen["identity_sha256"]
        seen.add(key)
        assert len(block["rows"])==len(protocol["learners"])*len(methods)
        identities=set()
        for row in block["rows"]:
            identity=(row["learner"],row["method"])
            assert identity not in identities; identities.add(identity)
            assert row["finite"] and not row["failure_reason"]
            assert (row["design"],row["dgp"],row["replication"])==key
            assert row["artificial_pressure"] is False and row["treatment_learner"]==row["learner"]
            near(row["theta"]-row["theta0"],row["bias"])
            near(abs(row["bias"]),row["abs_error"])
            near(row["bias"]**2,row["squared_error"])
            assert row["covers"]==(abs(row["bias"])<=1.96*row["se"])
            groups[(*key[:2],*identity)][key[2]]=row
        assert identities=={(l,m) for l in protocol["learners"] for m in methods}
    assert seen==universe
    summaries=read(root/"analysis/summary.json")
    assert len(summaries)==len(groups)==36
    for summary in summaries:
        key=tuple(summary[n] for n in ("design","dgp","learner","method"))
        group=groups[key]; assert set(group)==set(range(protocol["replications"][key[0]]))
        values=list(group.values()); n=len(values)
        errors=[r["theta"]-.5 for r in values]
        mean=statistics.mean(errors); sd=statistics.stdev(errors)
        near(summary["mean_theta"],statistics.mean(r["theta"] for r in values))
        near(summary["bias"]["mean"],mean)
        half=float(t.ppf(.975,n-1))*sd/math.sqrt(n)
        near(summary["bias"]["low"],mean-half); near(summary["bias"]["high"],mean+half)
        near(summary["mc_sd"],sd)
        near(summary["mae"],statistics.mean(abs(x) for x in errors))
        near(summary["rmse"],math.sqrt(statistics.mean(x*x for x in errors)))
        near(summary["mean_se"],statistics.mean(r["se"] for r in values))
        p=sum(r["covers"] for r in values)/n; z=1.959963984540054
        center=(p+z*z/(2*n))/(1+z*z/n)
        width=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
        for name,number in [("rate",p),("low",center-width),("high",center+width)]:
            near(summary["coverage"][name],number)
    contrasts=read(root/"analysis/paired_contrasts.json"); assert len(contrasts)==24
    for contrast in contrasts:
        key=tuple(contrast[n] for n in ("design","dgp","learner"))
        a=groups[(*key,contrast["baseline"])]; b=groups[(*key,"iv_crossfit")]
        assert set(a)==set(b)
        for field,metric in [("signed_bias_difference","bias"),("mae_difference","abs_error"),("mse_difference","squared_error")]:
            values=[b[r][metric]-a[r][metric] for r in sorted(a)]
            mean=statistics.mean(values); half=float(t.ppf(.975,len(values)-1))*statistics.stdev(values)/math.sqrt(len(values))
            for name,value in [("mean",mean),("low",mean-half),("high",mean+half)]: near(contrast[field][name],value)
        for r in a:
            naive=groups[(*key,"nonorthogonal_crossfit")][r]
            assert naive["g_prediction_hash"]==b[r]["g_prediction_hash"]
    for agg in read(root/"analysis/aggregate_bias.json"):
        values={method:statistics.mean(abs(statistics.mean(r["bias"] for r in groups[(agg["design"],dgp,learner,method)].values()))
                                      for dgp in ("linear","nonlinear") for learner in protocol["learners"])
                for method in (agg["baseline"],"iv_crossfit")}
        near(agg["baseline_absolute_mean_bias"],values[agg["baseline"]])
        near(agg["crossfit_absolute_mean_bias"],values["iv_crossfit"])
        near(agg["reduction_percent"],100*(1-values["iv_crossfit"]/values[agg["baseline"]]))
    analysis_ledger=read(root/"analysis/ANALYSIS_LEDGER.json")
    for rel,value in analysis_ledger.items(): assert sha(root/rel)==value
    for design in ("figure1","figure2"):
        for dgp in ("linear","nonlinear"):
            image=root/"analysis"/f"{design}_{dgp}.png"
            assert image.stat().st_size>20000 and image.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    print(json.dumps(dict(status="PASS_INDEPENDENT_FULL_RESULT_VERIFICATION",
                          replications=len(seen),estimates=sum(len(g) for g in groups.values()),
                          summary_cells=len(summaries),paired_contrasts=len(contrasts))))


if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--root",type=Path,required=True)
    verify(parser.parse_args().root.resolve())
