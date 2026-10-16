#!/usr/bin/env python3
"""Server-only one-shot development selection followed by frozen confirmation."""
from __future__ import annotations
import argparse
import concurrent.futures as futures
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import warnings

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"src"))
from dml_studies import nuisance_comparison as repair
try:
    from .study_utils import atomic_json,digest
    from .execution_guard import require_execution_root
except ImportError:
    from study_utils import atomic_json,digest
    from execution_guard import require_execution_root

FILES=("src/dml_studies/configs/nuisance_comparison.json",
    "src/dml_studies/nuisance_comparison.py","src/dml_studies/doubleml_reference.py",
    "src/dml_studies/iv_score.py","src/dml_studies/__init__.py",
    "src/dml_studies/cli/run_nuisance_comparison.py","src/dml_studies/cli/execution_guard.py",
    "src/dml_studies/cli/analyze_nuisance_comparison.py","src/dml_studies/cli/study_utils.py",
    "tests/conftest.py","tests/test_nuisance_comparison.py","pyproject.toml")


def once(path,value):
    with path.open("x") as stream:
        json.dump(value,stream,indent=2,allow_nan=False); stream.flush(); os.fsync(stream.fileno())


def environment():
    import numpy,scipy,sklearn,xgboost
    return dict(python=platform.python_version(),numpy=numpy.__version__,scipy=scipy.__version__,
        sklearn=sklearn.__version__,xgboost=xgboost.__version__)


def task(mode,protocol,selections,key):
    from threadpoolctl import threadpool_limits
    started=time.monotonic()
    with threadpool_limits(limits=1),warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        value=repair.selection_job(protocol,*key) if mode=="selection" else repair.run_replication(protocol,selections,*key)
    return dict(value=value,elapsed_seconds=time.monotonic()-started,
        warnings=[dict(category=w.category.__name__,message=str(w.message)) for w in captured])


def dispatch(mode,protocol,selections,universe,out,identity,workers):
    out.mkdir(parents=True,exist_ok=True)
    jobs=[(key,out/("_".join(map(str,key))+".json")) for key in universe]
    if any(p.exists() for _,p in jobs): raise RuntimeError("no automatic re-run/resume")
    ledger={}; started=time.monotonic()
    def status(s): atomic_json(out/"STATUS.json",dict(status=s,completed=len(ledger),expected=len(jobs),
        identity_sha256=identity,seconds=time.monotonic()-started))
    status("running")
    with futures.ProcessPoolExecutor(max_workers=workers) as executor:
        pending={}; iterator=iter(jobs)
        def submit():
            item=next(iterator,None)
            if item:
                key,path=item; pending[executor.submit(task,mode,protocol,selections,key)]=(key,path)
        for _ in range(2*workers): submit()
        while pending:
            done,_=futures.wait(pending,return_when=futures.FIRST_COMPLETED)
            for future in done:
                key,path=pending.pop(future)
                try: value=future.result()
                except BaseException as exc:
                    atomic_json(out/"ERROR.json",dict(key=key,error=repr(exc))); status("failed"); raise
                value.update(key=key,identity_sha256=identity); once(path,value)
                ledger[path.name]=digest(path); status("running")
                if len(ledger)%25==0 or len(ledger)==len(jobs):
                    print(mode,len(ledger),"/",len(jobs),round(time.monotonic()-started,1),flush=True)
                submit()
    once(out/"RESULT_LEDGER.json",ledger); status("complete")


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("mode",choices=("launch","run")); args=parser.parse_args()
    require_execution_root(ROOT)
    env=dict(os.environ); env.update(PYTHONPATH=str(ROOT/"src"),MPLCONFIGDIR=str(ROOT/"mpl_cache"),
        OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",CUDA_VISIBLE_DEVICES="",PYTHONDONTWRITEBYTECODE="1")
    if args.mode=="launch":
        command=[sys.executable,str(Path(__file__).resolve()),"run"]
        once(ROOT/"SERVER_LAUNCH_INTENT.json",dict(at=time.time(),command=command,local_simulation=False))
        with (ROOT/"SERVER_JOB.log").open("x") as log:
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        once(ROOT/"SERVER_LAUNCH.json",dict(at=time.time(),pid=process.pid))
        print(json.dumps(dict(status="LAUNCHED_ONCE",pid=process.pid))); return
    os.nice(10); once(ROOT/"SERVER_START.json",dict(at=time.time(),pid=os.getpid(),host=platform.node(),python=sys.executable))
    code=1
    try:
        protocol=json.loads((ROOT/"src/dml_studies/configs/nuisance_comparison.json").read_text())
        source_hashes={p:digest(ROOT/p) for p in FILES}
        identity=dict(protocol=protocol,source_hashes=source_hashes,environment=environment())
        sha=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        once(ROOT/"SOURCE_FROZEN.json",dict(identity=identity,identity_sha256=sha,at=time.time()))
        with (ROOT/"SERVER_PREFLIGHT.log").open("x") as log:
            check=subprocess.run([sys.executable,"-m","pytest","tests/test_nuisance_comparison.py","-q"],env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        once(ROOT/"SERVER_PREFLIGHT.json",dict(returncode=check.returncode,log_sha256=digest(ROOT/"SERVER_PREFLIGHT.log")))
        if check.returncode: raise RuntimeError("preflight failed; selection/formal not launched")
        universe=[(d,g,r,l,role) for d in ("figure1","figure2") for g in ("linear","nonlinear")
            for r in range(protocol["development_replications_per_cell"]) for l in protocol["ordinary_families"] for role in ("m","g")]
        selection=ROOT/"output/selection"
        dispatch("selection",protocol,None,universe,selection,sha,protocol["workers"])
        losses={}; receipt_warnings=[]
        for file in sorted(selection.glob("figure*.json")):
            receipt=json.loads(file.read_text()); value=receipt["value"]
            key=f"{value['design']}/{value['dgp']}/{value['family']}"
            losses.setdefault(key,{})
            role=value["role"]
            if role not in losses[key]: losses[key][role]=[0.]*len(value["squared_errors"])
            losses[key][role]=[a+b for a,b in zip(losses[key][role],value["squared_errors"])]
            receipt_warnings.extend(receipt["warnings"])
        selected={key:{role:repair.candidates(key.rsplit("/",1)[1])[min(range(len(loss)),key=loss.__getitem__)]
            for role,loss in roles.items()} for key,roles in losses.items()}
        once(selection/"SELECTED.json",dict(selected=selected,pooled_validation_sse=losses,
            warnings=receipt_warnings,selection_uses="observed_validation_prediction_loss_only",
            source_identity_sha256=sha,at=time.time()))
        formal=ROOT/"output/formal"; formal.mkdir(parents=True,exist_ok=True)
        formal_identity={**identity,"selection_sha256":digest(selection/"SELECTED.json"),"selected":selected}
        formal_sha=hashlib.sha256(json.dumps(formal_identity,sort_keys=True).encode()).hexdigest()
        once(formal/"FROZEN_MANIFEST.json",dict(identity=formal_identity,identity_sha256=formal_sha,at=time.time()))
        universe=[(d,g,r) for d in ("figure1","figure2") for g in ("linear","nonlinear") for r in range(protocol["replications"][d])]
        dispatch("formal",protocol,selected,universe,formal,formal_sha,protocol["workers"])
        subprocess.run([sys.executable,"src/dml_studies/cli/analyze_nuisance_comparison.py"],env=env,cwd=ROOT,check=True)
        code=0
    finally: once(ROOT/"SERVER_EXIT.json",dict(returncode=code,at=time.time(),pid=os.getpid()))
    print("COMPLETE",flush=True)


if __name__=="__main__": main()
