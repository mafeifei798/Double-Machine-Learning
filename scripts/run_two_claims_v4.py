#!/usr/bin/env python3
"""Server-only source-frozen one-shot confirmation; no bias-driven tuning."""
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
from dml_original import two_claims_v4 as study
from run_fair_iv import atomic_json,digest
FILES=('configs/two_claims_v4.json','docs/TWO_CLAIMS_V4_PROTOCOL.md',
    'src/dml_original/two_claims_v4.py','src/dml_original/fair_iv.py',
    'src/dml_original/iv_thesis_repair.py','src/dml_original/__init__.py','src/dml_original/simulation.py',
    'scripts/run_two_claims_v4.py','scripts/run_fair_iv.py','scripts/analyze_fair_iv.py',
    'scripts/analyze_two_claims_v4.py','scripts/verify_two_claims_v4.py',
    'scripts/check_two_claims_doubleml.py','tests/test_two_claims_v4.py')


def once(path,value):
    with path.open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.flush();os.fsync(stream.fileno())


def task(protocol,key):
    from threadpoolctl import threadpool_limits
    started=time.monotonic()
    with threadpool_limits(limits=1),warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always'); rows=study.run_replication(protocol,*key)
    return dict(rows=rows,seconds=time.monotonic()-started,
        warnings=[dict(category=w.category.__name__,message=str(w.message)) for w in captured])


def dispatch(protocol,out,identity):
    jobs=[(d,g,r) for d in ('figure1','figure2') for g in ('linear','nonlinear')
        for r in range(protocol['replications'][d])]
    ledger={};started=time.monotonic()
    def status(state):
        atomic_json(out/'STATUS.json',dict(status=state,completed=len(ledger),expected=len(jobs),
            identity_sha256=identity,seconds=time.monotonic()-started))
    status('running')
    with futures.ProcessPoolExecutor(max_workers=protocol['workers']) as pool:
        iterator=iter(jobs);pending={}
        def submit():
            key=next(iterator,None)
            if key is not None:pending[pool.submit(task,protocol,key)]=key
        for _ in range(2*protocol['workers']):submit()
        while pending:
            done,_=futures.wait(pending,return_when=futures.FIRST_COMPLETED)
            for future in done:
                key=pending.pop(future)
                try:value=future.result()
                except BaseException as exc:
                    once(out/'ERROR.json',dict(key=key,error=repr(exc)));status('failed');raise
                value.update(key=key,identity_sha256=identity)
                path=out/('_'.join(map(str,key))+'.json');once(path,value);ledger[path.name]=digest(path)
                status('running')
                if len(ledger)%100==0:print('FORMAL',len(ledger),'/',len(jobs),round(time.monotonic()-started,1),flush=True)
                submit()
    once(out/'RESULT_LEDGER.json',ledger);status('complete')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('launch','run'));args=parser.parse_args()
    if platform.system()!='Linux' or str(ROOT)!='/data2/lzx/dhy/dml_two_claims_v4_20261002':
        raise RuntimeError('all fitting, simulations and tests must run on the authorized server')
    env=dict(os.environ);env.update(PYTHONPATH=str(ROOT/'src'),MPLCONFIGDIR=str(ROOT/'mpl_cache'),
        OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
    if args.mode=='launch':
        command=[sys.executable,str(Path(__file__).resolve()),'run']
        once(ROOT/'SERVER_LAUNCH_INTENT.json',dict(command=command,at=time.time()))
        with (ROOT/'SERVER_JOB.log').open('x') as log:
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        once(ROOT/'SERVER_LAUNCH.json',dict(pid=process.pid,at=time.time()))
        print(json.dumps(dict(status='LAUNCHED_ONCE',pid=process.pid)));return
    os.nice(10);once(ROOT/'SERVER_START.json',dict(host=platform.node(),pid=os.getpid(),python=sys.executable,at=time.time()))
    code=1
    try:
        import numpy,scipy,sklearn,xgboost,doubleml
        import doubleml.plm.plr as plr
        protocol=json.loads((ROOT/'configs/two_claims_v4.json').read_text())
        identity=dict(protocol=protocol,source_hashes={p:digest(ROOT/p) for p in FILES},
            environment=dict(python=platform.python_version(),numpy=numpy.__version__,scipy=scipy.__version__,
                sklearn=sklearn.__version__,xgboost=xgboost.__version__,doubleml=doubleml.__version__,
                doubleml_plr_sha256=digest(Path(plr.__file__))))
        sha=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        once(ROOT/'SOURCE_FROZEN.json',dict(identity=identity,identity_sha256=sha,at=time.time()))
        with (ROOT/'SERVER_PREFLIGHT.log').open('x') as log:
            check=subprocess.run([sys.executable,'-m','pytest','tests/test_two_claims_v4.py','-q'],
                env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        once(ROOT/'SERVER_PREFLIGHT.json',dict(returncode=check.returncode,log_sha256=digest(ROOT/'SERVER_PREFLIGHT.log')))
        if check.returncode:raise RuntimeError('preflight failed; formal NOT launched')
        with (ROOT/'DOUBLEML_CHECK.json').open('x') as stream:
            subprocess.run([sys.executable,'scripts/check_two_claims_doubleml.py'],env=env,cwd=ROOT,stdout=stream,check=True)
        out=ROOT/'output/formal';out.mkdir(parents=True,exist_ok=False)
        once(out/'FROZEN_MANIFEST.json',dict(identity=identity,identity_sha256=sha,
            doubleml_check_sha256=digest(ROOT/'DOUBLEML_CHECK.json')))
        dispatch(protocol,out,sha)
        subprocess.run([sys.executable,'scripts/analyze_two_claims_v4.py'],env=env,cwd=ROOT,check=True)
        with (ROOT/'INDEPENDENT_VERIFICATION.json').open('x') as stream:
            subprocess.run([sys.executable,'scripts/verify_two_claims_v4.py'],env=env,cwd=ROOT,stdout=stream,check=True)
        code=0
    finally:once(ROOT/'SERVER_EXIT.json',dict(returncode=code,pid=os.getpid(),at=time.time()))
    print('COMPLETE_VERIFIED',flush=True)


if __name__=='__main__':main()
