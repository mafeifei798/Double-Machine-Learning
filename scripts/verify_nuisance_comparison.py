#!/usr/bin/env python3
"""Independent read-only ledger, selection, pairing and statistics verifier."""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from scipy.stats import t,norm

ROOT=Path(__file__).resolve().parents[1]


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def check():
    root=ROOT/"output/formal"; frozen=json.loads((root/"FROZEN_MANIFEST.json").read_text())
    identity=frozen["identity"]; p=identity["protocol"]; selection=ROOT/"output/selection"
    assert hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()==frozen["identity_sha256"]
    for path,value in identity["source_hashes"].items(): assert sha(ROOT/path)==value,path
    assert sha(selection/"SELECTED.json")==identity["selection_sha256"]
    assert sha(ROOT/"DOUBLEML_CHECK.json")==identity["doubleml_check_sha256"]
    checks=json.loads((ROOT/"DOUBLEML_CHECK.json").read_text()); assert checks["status"]=="PASS" and len(checks["checks"])==20
    selected=json.loads((selection/"SELECTED.json").read_text()); losses={}; selection_keys=set(); dev_seeds=set()
    for path,value in json.loads((selection/"RESULT_LEDGER.json").read_text()).items():
        assert sha(selection/path)==value
        r=json.loads((selection/path).read_text()); v=r["value"]; key=tuple(r["key"])
        assert key not in selection_keys; selection_keys.add(key); dev_seeds.add(v["data_seed"])
        assert "theta" not in v and "bias" not in v and v["selection_uses"]=="observed_validation_prediction_loss_only"
        label=f"{v['design']}/{v['dgp']}/{v['family']}"; k=(label,v["role"])
        losses[k]=losses.get(k,np.zeros(len(v["squared_errors"])))+np.array(v["squared_errors"])
        index=int(np.argmin(np.array(selected["pooled_validation_sse"][label][v["role"]])))
        assert selected["selected"][label][v["role"]]==v["candidates"][index]
    expected_selection={(d,g,r,l,role) for d in ("figure1","figure2") for g in ("linear","nonlinear")
        for r in range(p["development_replications_per_cell"]) for l in p["ordinary_families"] for role in ("m","g")}
    assert selection_keys==expected_selection
    for (label,role),loss in losses.items(): np.testing.assert_allclose(loss,selected["pooled_validation_sse"][label][role],rtol=0,atol=1e-8)
    grouped=defaultdict(list); seen=set(); warnings=0; formal_seeds=set(); row_n=0
    expected={(d,g,r) for d in ("figure1","figure2") for g in ("linear","nonlinear") for r in range(p["replications"][d])}
    ledger=json.loads((root/"RESULT_LEDGER.json").read_text())
    assert set(ledger)=={file.name for file in root.glob("figure*.json")}
    for path,value in ledger.items():
        assert sha(root/path)==value
        artifact=json.loads((root/path).read_text()); key=tuple(artifact["key"])
        assert key in expected and key not in seen and artifact["identity_sha256"]==frozen["identity_sha256"]
        seen.add(key); warnings+=len(artifact["warnings"])
        controls=["series_ols"] if key[0]=="figure1" else ["series_ols","tree_leaf_2","tree_leaf_5","tree_leaf_10","tree_leaf_20","rf_outcome_series_m"]
        targets={(track,l,m) for track,ls in (("ordinary",p["ordinary_families"]),("controlled",controls)) for l in ls
            for m in (("nonorthogonal_crossfit","full_no_split","equal_train_reuse","crossfit")+
                (("outcome_mismatch_crossfit",) if track=="controlled" and key[0]=="figure1" else ())) }
        rows=artifact["value"]; assert len(rows)==len(targets) and {(r["track"],r["learner"],r["method"]) for r in rows}==targets
        row_n+=len(rows); pairs=defaultdict(dict)
        for r in rows:
            assert (r["design"],r["dgp"],r["replication"])==key and r["finite"] and not r["artificial_pressure"]
            assert r["n"]==1000 and r["theta0"]==.5 and r["se"]>0 and abs(r["score_mean"])<1e-10
            formal_seeds.add(r["data_seed"])
            error=r["theta"]-.5
            assert abs(error-r["bias"])<1e-12 and abs(error**2-r["squared_error"])<1e-12
            assert r["covers"]==(abs(error)<=1.96*r["se"])
            if r["track"]=="ordinary": assert r["selected_params"]==selected["selected"][f"{r['design']}/{r['dgp']}/{r['learner']}"]
            records=r["bank_records"]
            assert hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest()==r["bank_identity"]
            assert all(record["train_n"]==r["train_n_per_model"] for record in records)
            assert r["train_n_per_model"]==(1000 if r["method"]=="full_no_split" else 500)
            assert r["prediction_overlap_n"]==(1000 if r["method"] in ("full_no_split","equal_train_reuse") else 0)
            if r["method"] in ("full_no_split","equal_train_reuse","crossfit"):
                assert abs(sum(r[k] for k in ("weighted_oracle","weighted_first_order","weighted_second_order"))-error)<1e-10
            grouped[(r["design"],r["dgp"],r["track"],r["learner"],r["method"])].append(r)
            pairs[(r["track"],r["learner"])][r["method"]]=r
        for pair in pairs.values():
            a,b=pair["equal_train_reuse"],pair["crossfit"]
            assert a["bank_identity"]==b["bank_identity"] and a["bank_records"]==b["bank_records"]
            assert pair["nonorthogonal_crossfit"]["g_prediction_hash"]==b["g_prediction_hash"]
    assert seen==expected and not formal_seeds.intersection(dev_seeds)
    summaries=json.loads((root/"analysis/summary.json").read_text()); assert len(summaries)==len(grouped)
    for s in summaries:
        key=tuple(s[k] for k in ("design","dgp","track","learner","method")); rows=grouped[key]
        b=np.array([r["theta"]-.5 for r in rows]); n=len(b); avg=float(b.mean()); half=float(t.ppf(.975,n-1)*b.std(ddof=1)/np.sqrt(n))
        for key,value in (("mean",avg),("low",avg-half),("high",avg+half)): assert math.isclose(s["bias"][key],value,abs_tol=1e-12)
        assert s["practical_zero_supported"]==(avg-half>-.01 and avg+half<.01)
        assert s["bias_detected"]==(avg-half>0 or avg+half<0)
        assert abs(s["rmse"]-np.sqrt(np.mean(b*b)))<1e-12 and abs(s["mae"]-np.mean(np.abs(b)))<1e-12
        rate=np.mean([r["covers"] for r in rows]); z=norm.ppf(.975); denominator=1+z*z/n
        center=(rate+z*z/(2*n))/denominator; half=z*np.sqrt(rate*(1-rate)/n+z*z/(4*n*n))/denominator
        for key,value in (("rate",rate),("low",center-half),("high",center+half)): assert abs(s["coverage"][key]-value)<1e-12
        if "weighted_components" in s:
            for key,interval in s["weighted_components"].items():
                values=np.array([r[key] for r in rows]); h=t.ppf(.975,n-1)*values.std(ddof=1)/np.sqrt(n)
                for label,value in (("mean",values.mean()),("low",values.mean()-h),("high",values.mean()+h)): assert abs(interval[label]-value)<1e-12
    contrasts=json.loads((root/"analysis/paired_contrasts.json").read_text())
    for c in contrasts:
        prefix=tuple(c[k] for k in ("design","dgp","track","learner"))
        a=sorted(grouped[(*prefix,c["baseline"])],key=lambda r:r["replication"])
        b=sorted(grouped[(*prefix,"crossfit")],key=lambda r:r["replication"])
        for label,field in (("signed_bias_difference","bias"),("mse_difference","squared_error")):
            diff=np.array([rb[field]-ra[field] for ra,rb in zip(a,b)]); h=t.ppf(.975,len(diff)-1)*diff.std(ddof=1)/np.sqrt(len(diff))
            for name,value in (("mean",diff.mean()),("low",diff.mean()-h),("high",diff.mean()+h)): assert abs(c[label][name]-value)<1e-12
        delta=abs(np.mean([r["bias"] for r in b]))-abs(np.mean([r["bias"] for r in a]))
        assert abs(c["absolute_mean_bias_difference"]-delta)<1e-12
    for path,value in json.loads((root/"analysis/ANALYSIS_LEDGER.json").read_text()).items(): assert sha(root/"analysis"/path)==value
    assert json.loads((root/"STATUS.json").read_text())["status"]=="complete"
    return dict(status="PASS",replications=len(ledger),estimates=row_n,summary_cells=len(summaries),
        selection_tasks=len(selection_keys),doubleml_checks=len(checks["checks"]),paired_contrasts=len(contrasts),
        warnings=warnings,failures=0,formal_identity_sha256=frozen["identity_sha256"],all_cells_retained=True)


if __name__=="__main__": print(json.dumps(check(),indent=2))
