#!/usr/bin/env python3
"""Server-only, hash-locked equal-training follow-up with durable replications."""
from __future__ import annotations
import argparse
import concurrent.futures as futures
import fcntl
import hashlib
import json
import platform
from pathlib import Path
import sys
import time
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from dml_original.equal_training_iv import run_replication
from run_fair_iv import atomic_json, digest


def worker(protocol, design, dgp, rep, development):
    from threadpoolctl import threadpool_limits
    started = time.monotonic()
    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        rows = run_replication(protocol, design, dgp, rep, development=development)
    return dict(rows=rows, elapsed_seconds=time.monotonic()-started,
                warnings=[dict(category=w.category.__name__,message=str(w.message)) for w in captured])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--workers",type=int,default=12)
    parser.add_argument("--development",action="store_true")
    args=parser.parse_args()
    if platform.system() != "Linux" or str(ROOT) != "/data2/lzx/dhy/dml_equal_training_v1_20261002":
        raise RuntimeError("server only: wrong platform/directory")
    if not 1<=args.workers<=12: raise ValueError("worker budget")
    protocol=json.loads((ROOT/"configs/equal_training_iv_v1.json").read_text())
    out=args.out.resolve(); out.mkdir(parents=True,exist_ok=True)
    lock=(out/"RUN.lock").open("a"); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    import numpy,scipy,sklearn,xgboost
    paths=[ROOT/p for p in (
        "configs/equal_training_iv_v1.json","docs/EQUAL_TRAINING_IV_V1_PROTOCOL.md",
        "src/dml_original/equal_training_iv.py","src/dml_original/fair_iv.py",
        "src/dml_original/iv_thesis_repair.py","src/dml_original/__init__.py",
        "src/dml_original/simulation.py","scripts/run_fair_iv.py",
        "scripts/run_equal_training_iv.py","scripts/analyze_equal_training_iv.py",
        "scripts/analyze_fair_iv.py","scripts/verify_equal_training_iv.py",
        "scripts/equal_training_server_job.py","tests/test_equal_training_iv.py",
        "tests/test_fair_iv.py","configs/fair_iv_v1.json",
        "legacy_snapshot/figure1_doubleml_docs_style_20260421.py",
        "legacy_snapshot/figure2_compare_learners_common_20260412.py")]
    identity=dict(protocol=protocol,development=args.development,
                  source_hashes={str(p.relative_to(ROOT)):digest(p) for p in paths},
                  environment=dict(python=platform.python_version(),numpy=numpy.__version__,
                                   scipy=scipy.__version__,sklearn=sklearn.__version__,xgboost=xgboost.__version__))
    sha=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    manifest=out/"FROZEN_MANIFEST.json"
    if manifest.exists():
        prior=json.loads(manifest.read_text())
        if prior["identity"]!=identity or prior["identity_sha256"]!=sha:
            raise RuntimeError("frozen source/config/environment changed")
    else: atomic_json(manifest,dict(identity=identity,identity_sha256=sha,frozen_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())))
    universe=[(d,g,r) for d in ("figure1","figure2") for g in ("linear","nonlinear")
              for r in range(2 if args.development else protocol["replications"][d])]
    jobs=[]; completed={}
    for design,dgp,rep in universe:
        cell=out/f"{design}_{dgp}"; cell.mkdir(exist_ok=True); path=cell/f"rep_{rep:04d}.json"
        if path.exists():
            prior=json.loads(path.read_text())
            if prior["identity_sha256"]!=sha or prior["key"]!=[design,dgp,rep]: raise RuntimeError("bad resume")
            completed[str(path.relative_to(out))]=digest(path)
        else: jobs.append((design,dgp,rep,path))
    started=time.monotonic()
    def progress(status):
        atomic_json(out/"STATUS.json",dict(status=status,completed=len(completed),expected=len(universe),
                    identity_sha256=sha,current_invocation_seconds=time.monotonic()-started))
    progress("running"); print(json.dumps(dict(identity=sha,pending=len(jobs),resumed=len(completed))),flush=True)
    with futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        pending={}; iterator=iter(jobs)
        def submit():
            job=next(iterator,None)
            if job:
                d,g,r,p=job; pending[executor.submit(worker,protocol,d,g,r,args.development)]=job
        for _ in range(2*args.workers): submit()
        while pending:
            done,_=futures.wait(pending,return_when=futures.FIRST_COMPLETED)
            for future in done:
                d,g,r,path=pending.pop(future)
                try: result=future.result()
                except BaseException as exc:
                    atomic_json(out/"ERROR.json",dict(key=[d,g,r],error=repr(exc))); progress("failed"); raise
                result.update(identity_sha256=sha,key=[d,g,r]); atomic_json(path,result)
                completed[str(path.relative_to(out))]=digest(path); progress("running")
                if len(completed)%50==0 or len(completed)==len(universe):
                    print(json.dumps(dict(completed=len(completed),expected=len(universe),seconds=round(time.monotonic()-started,1))),flush=True)
                submit()
    atomic_json(out/"RESULT_LEDGER.json",completed); progress("complete"); print("COMPLETE",flush=True)


if __name__=="__main__": main()
