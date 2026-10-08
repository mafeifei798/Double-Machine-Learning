#!/usr/bin/env python3
"""Source-frozen server-only one-shot paired exploratory extension."""
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
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from dml_original import harder_v5 as study
from run_fair_iv import atomic_json,digest
FILES=('configs/harder_v5.json','docs/HARDER_V5_PROTOCOL.md','src/dml_original/harder_v5.py',
    'src/dml_original/two_claims_v4.py','src/dml_original/fair_iv.py','src/dml_original/iv_thesis_repair.py',
    'src/dml_original/__init__.py','src/dml_original/simulation.py','scripts/run_harder_v5.py',
    'scripts/run_fair_iv.py','scripts/analyze_fair_iv.py','scripts/analyze_harder_v5.py',
    'scripts/verify_harder_v5.py','scripts/check_harder_doubleml.py','tests/test_harder_v5.py')


def once(path,value):
    with path.open('x') as s:
        json.dump(value,s,indent=2,allow_nan=False);s.flush();os.fsync(s.fileno())


def task(p,key):
    from threadpoolctl import threadpool_limits
    start=time.monotonic()
    with threadpool_limits(limits=1),warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always');rows=study.run_replication(p,*key)
    return dict(rows=rows,seconds=time.monotonic()-start,
        warnings=[dict(category=w.category.__name__,message=str(w.message)) for w in captured])


def dispatch(p,out,sha):
    universe=[(d,g,r) for d in ('figure1','figure2') for g in ('linear','nonlinear') for r in range(p['replications'][d])]
    ledger={};start=time.monotonic()
    def status(state):atomic_json(out/'STATUS.json',dict(status=state,completed=len(ledger),expected=len(universe),
        level_replications_completed=2*len(ledger),seconds=time.monotonic()-start,identity_sha256=sha))
    status('running')
    with futures.ProcessPoolExecutor(max_workers=p['workers']) as pool:
        iterator=iter(universe);pending={}
        def submit():
            key=next(iterator,None)
            if key is not None:pending[pool.submit(task,p,key)]=key
        for _ in range(2*p['workers']):submit()
        while pending:
            done,_=futures.wait(pending,return_when=futures.FIRST_COMPLETED)
            for future in done:
                key=pending.pop(future)
                try:value=future.result()
                except BaseException as exc:
                    once(out/'ERROR.json',dict(key=key,error=repr(exc)));status('failed');raise
                value.update(key=key,identity_sha256=sha);path=out/('_'.join(map(str,key))+'.json')
                once(path,value);ledger[path.name]=digest(path);status('running')
                if len(ledger)%50==0:print('PAIRS',len(ledger),'/',len(universe),round(time.monotonic()-start,1),flush=True)
                submit()
    once(out/'RESULT_LEDGER.json',ledger);status('complete')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('launch','run'));args=parser.parse_args()
    if platform.system()!='Linux' or str(ROOT)!='/data2/lzx/dhy/dml_harder_v5_20261002':raise RuntimeError('authorized server only; no local fitting')
    env=dict(os.environ);env.update(PYTHONPATH=str(ROOT/'src'),MPLCONFIGDIR=str(ROOT/'mpl_cache'),
        OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
    if args.mode=='launch':
        command=[sys.executable,str(Path(__file__).resolve()),'run'];once(ROOT/'SERVER_LAUNCH_INTENT.json',dict(command=command,at=time.time()))
        with (ROOT/'SERVER_JOB.log').open('x') as log:
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        once(ROOT/'SERVER_LAUNCH.json',dict(pid=process.pid,at=time.time()));print(json.dumps(dict(status='LAUNCHED_ONCE',pid=process.pid)));return
    os.nice(10);once(ROOT/'SERVER_START.json',dict(host=platform.node(),pid=os.getpid(),python=sys.executable,at=time.time()));code=1
    try:
        import numpy,scipy,sklearn,xgboost,doubleml
        import doubleml.plm.plr as plr
        p=json.loads((ROOT/'configs/harder_v5.json').read_text())
        identity=dict(protocol=p,source_hashes={f:digest(ROOT/f) for f in FILES},
            environment=dict(python=platform.python_version(),numpy=numpy.__version__,scipy=scipy.__version__,
                sklearn=sklearn.__version__,xgboost=xgboost.__version__,doubleml=doubleml.__version__,doubleml_plr_sha256=digest(Path(plr.__file__))))
        sha=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        once(ROOT/'SOURCE_FROZEN.json',dict(identity=identity,identity_sha256=sha,at=time.time()))
        with (ROOT/'SERVER_PREFLIGHT.log').open('x') as log:
            test=subprocess.run([sys.executable,'-m','pytest','tests/test_harder_v5.py','-q'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        once(ROOT/'SERVER_PREFLIGHT.json',dict(returncode=test.returncode,log_sha256=digest(ROOT/'SERVER_PREFLIGHT.log')))
        if test.returncode:raise RuntimeError('software preflight failed; no exploratory experiment launched')
        with (ROOT/'DOUBLEML_CHECK.json').open('x') as s:
            subprocess.run([sys.executable,'scripts/check_harder_doubleml.py'],cwd=ROOT,env=env,stdout=s,check=True)
        out=ROOT/'output/exploratory';out.mkdir(parents=True,exist_ok=False)
        once(out/'FROZEN_MANIFEST.json',dict(identity=identity,identity_sha256=sha,doubleml_check_sha256=digest(ROOT/'DOUBLEML_CHECK.json')))
        dispatch(p,out,sha)
        subprocess.run([sys.executable,'scripts/analyze_harder_v5.py'],cwd=ROOT,env=env,check=True)
        with (ROOT/'INDEPENDENT_VERIFICATION.json').open('x') as s:
            subprocess.run([sys.executable,'scripts/verify_harder_v5.py'],cwd=ROOT,env=env,stdout=s,check=True)
        code=0
    finally:once(ROOT/'SERVER_EXIT.json',dict(returncode=code,pid=os.getpid(),at=time.time()))
    print('COMPLETE_VERIFIED_EXPLORATORY',flush=True)


if __name__=='__main__':main()
