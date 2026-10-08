#!/usr/bin/env python3
"""Independent read-only metric and provenance audit; no model/runner imports."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np
from scipy.stats import t,norm


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def close(a,b):
    if not math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-12): raise AssertionError((a,b))


def verify(root,previous):
    frozen=json.loads((root/"FROZEN_MANIFEST.json").read_text()); identity=frozen["identity"]
    assert hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()==frozen["identity_sha256"]
    for rel,h in identity["source_hashes"].items(): assert sha(root.parents[1]/rel)==h,rel
    status=json.loads((root/"STATUS.json").read_text()); assert status["status"]=="complete"
    p=identity["protocol"]; methods=("equal_train_reuse","equal_train_crossfit")
    expected={(d,g,r) for d in ("figure1","figure2") for g in ("linear","nonlinear")
              for r in range(2 if identity["development"] else p["replications"][d])}
    ledger=json.loads((root/"RESULT_LEDGER.json").read_text()); seen=set(); groups=defaultdict(list); warnings=0
    assert set(ledger)=={str(x.relative_to(root)) for x in root.glob("figure*_*/*rep_*.json")}
    for rel,h in ledger.items():
        assert sha(root/rel)==h; a=json.loads((root/rel).read_text()); key=tuple(a["key"])
        assert key in expected and key not in seen; seen.add(key)
        assert a["identity_sha256"]==frozen["identity_sha256"]
        rows=a["rows"]; warnings+=len(a["warnings"])
        assert len(rows)==6 and {(r["learner"],r["method"]) for r in rows}=={(l,m) for l in p["learners"] for m in methods}
        seed=p["development_seed"] if identity["development"] else p["seed"]
        seed_payload="\0".join(map(str,("fair-iv-v1",seed,key[0],key[1],key[2],"folds"))).encode()
        fold_seed=int.from_bytes(hashlib.sha256(seed_payload).digest()[:4],"little")
        order=np.random.default_rng(fold_seed).permutation(p["n"])
        folds=np.empty(p["n"],dtype=np.int16)
        for j,ix in enumerate(np.array_split(order,2)): folds[ix]=j
        fold_bytes=b"thesis-iv-repair-fold-v1\0"+p["n"].to_bytes(8,"little")+folds.astype("<i2").tobytes()
        def vector_hash(x): return hashlib.sha256(np.asarray(x,dtype="<f8").tobytes()).hexdigest()
        if previous:
            old=json.loads((previous/rel).read_text())["rows"][0]
        for r in rows:
            assert r["finite"],r.get("failure_reason")
            assert (r["design"],r["dgp"],r["replication"])==key
            assert r["protocol_id"]==p["protocol_id"] and r["theta0"]==p["theta0"] and r["n"]==p["n"]
            assert not r["artificial_pressure"] and r["treatment_learner"]==r["learner"]
            assert r["train_n_per_model"]==p["n"]//2 and r["evaluation_n"]==p["n"] and r["fit_calls_per_bank"]==6
            assert r["pilot_mode"]==p["pilot"]
            cf=r["method"]==methods[1]
            assert r["every_row_held_out"]==cf and r["prediction_overlap_n"]==(0 if cf else p["n"])
            assert r["source_half_sha256"]==vector_hash(1-folds if cf else folds)
            assert r["fold_hash"]==hashlib.sha256(fold_bytes).hexdigest()[:16]
            if previous: assert r["data_seed"]==old["data_seed"] and r["fold_hash"]==old["fold_hash"]
            err=r["theta"]-p["theta0"]; close(err,r["bias"]); close(abs(err),r["abs_error"]); close(err*err,r["squared_error"])
            assert r["covers"]==(abs(err)<=1.96*r["se"]) and r["se"]>0 and abs(r["score_mean"])<1e-10
            close(r["score_at_truth"],r["oracle_score"]+r["first_order_score"]+r["second_order_score"])
            close(r["score_at_truth"],r["jacobian"]*err+r["score_mean"])
            assert hashlib.sha256(json.dumps(r["bank_records"],sort_keys=True).encode()).hexdigest()==r["bank_identity"]
            assert len(r["bank_records"])==2 and all(x["train_n"]==p["n"]//2 for x in r["bank_records"])
            for j,record in enumerate(r["bank_records"]):
                assert record["training_half"]==j and record["training_indices_sha256"]==vector_hash(np.flatnonzero(folds==j))
            assert r["pilots"]==[x["pilot"] for x in r["bank_records"]]
            groups[(key[0],key[1],r["learner"],r["method"])].append(r)
        for learner in p["learners"]:
            pair=[r for r in rows if r["learner"]==learner]
            for field in ("bank_identity","bank_records","pilots","fold_hash","data_seed"):
                assert pair[0][field]==pair[1][field]
    assert seen==expected and status["completed"]==status["expected"]==len(expected)
    for rows in groups.values(): rows.sort(key=lambda r:r["replication"])
    summary=json.loads((root/"analysis/summary.json").read_text()); assert len(summary)==24
    def interval(values,reported):
        m=statistics.mean(values); h=t.ppf(.975,len(values)-1)*statistics.stdev(values)/math.sqrt(len(values))
        for v,k in ((m,"mean"),(m-h,"low"),(m+h,"high")): close(v,reported[k])
    for s in summary:
        rows=groups[(s["design"],s["dgp"],s["learner"],s["method"])]; n=len(rows); assert s["R"]==n
        close(statistics.mean(r["theta"] for r in rows),s["mean_theta"])
        close(statistics.stdev(r["theta"] for r in rows),s["mc_sd"])
        close(statistics.mean(r["abs_error"] for r in rows),s["mae"])
        close(math.sqrt(statistics.mean(r["squared_error"] for r in rows)),s["rmse"])
        close(statistics.mean(r["se"] for r in rows),s["mean_se"])
        interval([r["bias"] for r in rows],s["bias"])
        for k,v in s["diagnostics"].items(): interval([r[k] for r in rows],v)
        rate=sum(r["covers"] for r in rows)/n; z=norm.ppf(.975)
        center=(rate+z*z/(2*n))/(1+z*z/n); half=z*math.sqrt(rate*(1-rate)/n+z*z/(4*n*n))/(1+z*z/n)
        close(rate,s["coverage"]["rate"]); close(center-half,s["coverage"]["low"]); close(center+half,s["coverage"]["high"])
    contrasts=json.loads((root/"analysis/paired_contrasts.json").read_text()); assert len(contrasts)==12
    for c in contrasts:
        a,b=(groups[(c["design"],c["dgp"],c["learner"],m)] for m in methods)
        for raw,output in (("bias","signed_bias_difference"),("abs_error","mae_difference"),("squared_error","mse_difference")):
            interval([y[raw]-x[raw] for x,y in zip(a,b)],c[output])
        close(abs(statistics.mean(r["bias"] for r in b))-abs(statistics.mean(r["bias"] for r in a)),c["absolute_mean_bias_difference"])
    aggregates=json.loads((root/"analysis/aggregate_bias.json").read_text()); assert len(aggregates)==2
    # Recompute the complete fixed-seed paired block bootstrap independently.
    rng=np.random.default_rng(p["bootstrap_seed"])
    for d,ag in zip(("figure1","figure2"),aggregates):
        matrices=[]
        for g in ("linear","nonlinear"):
            matrices.append(np.asarray([[[r["bias"] for r in groups[(d,g,l,m)]] for m in methods] for l in p["learners"]]))
        observed=np.mean([np.mean(np.abs(x.mean(axis=-1)),axis=0) for x in matrices],axis=0)
        close(observed[0],ag["reuse_absolute_mean_bias"]); close(observed[1],ag["crossfit_absolute_mean_bias"])
        close(observed[1]-observed[0],ag["difference"])
        draws=[]
        for _ in range(p["bootstrap_reps"]):
            v=[]
            for x in matrices:
                indices=rng.integers(0,x.shape[-1],x.shape[-1]); v.append(np.mean(np.abs(x[:,:,indices].mean(axis=-1)),axis=0))
            value=np.mean(v,axis=0); draws.append(value[1]-value[0])
        for a,b in zip(np.quantile(draws,[.025,.975]),ag["paired_block_bootstrap95"]): close(a,b)
    outputs=json.loads((root/"analysis/ANALYSIS_LEDGER.json").read_text())
    assert set(outputs)=={str(x.relative_to(root)) for x in (root/"analysis").iterdir() if x.is_file() and x.name!="ANALYSIS_LEDGER.json"}
    for rel,h in outputs.items(): assert sha(root/rel)==h
    pngs=list((root/"analysis").glob("*.png")); assert len(pngs)==5
    for png in pngs: assert png.read_bytes()[:8]==b"\x89PNG\r\n\x1a\n" and png.stat().st_size>10000
    return dict(status="PASS",replications=len(seen),rows=sum(len(r) for r in groups.values()),summary_cells=len(summary),
                paired_contrasts=len(contrasts),warnings=warnings,previous_data_and_folds_verified=bool(previous),identity_sha256=frozen["identity_sha256"])


if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--previous-root",type=Path)
    args=parser.parse_args(); print(json.dumps(verify(args.root.resolve(),args.previous_root),indent=2))
