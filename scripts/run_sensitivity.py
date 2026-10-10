#!/usr/bin/env python3
"""Single-launch server-only fixed sensitivity grid; durable per-job receipts."""
import argparse
import concurrent.futures as futures
import fcntl
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
from dml_studies import sensitivity as s
from run_reference import digest,atomic_json
FILES=('configs/sensitivity.json','configs/moderate_dgp.json','docs/sensitivity_protocol.md',
    'src/dml_studies/sensitivity.py','src/dml_studies/moderate_dgp.py','src/dml_studies/orthogonalization_crossfitting.py',
    'src/dml_studies/doubleml_reference.py','src/dml_studies/iv_score.py','src/dml_studies/__init__.py',
    'src/dml_studies/partialling_out.py','scripts/run_sensitivity.py','scripts/analyze_sensitivity.py',
    'scripts/verify_sensitivity.py','scripts/check_sensitivity_doubleml.py','scripts/check_sensitivity_replay.py',
    'scripts/run_reference.py','scripts/analyze_reference.py','scripts/analyze_moderate_dgp.py','tests/test_sensitivity.py')


def once(path,value):
    with path.open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.flush();os.fsync(stream.fileno())


def universe(p):
    return [(d,g,n,k,preset,rep) for rep in range(p['replications']) for n,k,preset in s.scenarios(p)
        for d in ('figure1','figure2') for g in ('linear','nonlinear')]


def filename(key):
    d,g,n,k,preset,rep=key;return f'{d}_{g}_n{n}_k{k}_{preset}_rep{rep:04d}.json'


def task(p,q,key):
    from threadpoolctl import threadpool_limits
    start=time.monotonic()
    with threadpool_limits(limits=1),warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always');rows=s.run_replication(p,q,*key)
    return dict(rows=rows,seconds=time.monotonic()-start,
        warnings=[dict(category=w.category.__name__,message=str(w.message)) for w in captured])


def dispatch(p,q,out,sha):
    keys=universe(p);ledger={};start=time.monotonic()
    def status(state):atomic_json(out/'STATUS.json',dict(status=state,completed=len(ledger),expected=len(keys),seconds=time.monotonic()-start,identity_sha256=sha))
    status('running')
    with futures.ProcessPoolExecutor(max_workers=p['workers']) as pool:
        it=iter(keys);pending={}
        def submit():
            key=next(it,None)
            if key is not None:pending[pool.submit(task,p,q,key)]=key
        for _ in range(2*p['workers']):submit()
        while pending:
            done,_=futures.wait(pending,return_when=futures.FIRST_COMPLETED)
            for future in done:
                key=pending.pop(future)
                try:value=future.result()
                except BaseException as exc:
                    once(out/'ERROR.json',dict(key=key,error=repr(exc)));status('failed');raise
                value.update(key=key,identity_sha256=sha);name=filename(key);once(out/name,value);ledger[name]=digest(out/name);status('running')
                if len(ledger)%100==0:print('JOBS',len(ledger),'/',len(keys),round(time.monotonic()-start,1),flush=True)
                submit()
    once(out/'RESULT_LEDGER.json',ledger);status('complete')


def main():
    a=argparse.ArgumentParser();a.add_argument('mode',choices=('launch','run'));args=a.parse_args()
    if platform.system()!='Linux' or str(ROOT)!='/data2/lzx/dhy/dml_sensitivity_v6_20261002':raise RuntimeError('authorized server only; local fits prohibited')
    env=dict(os.environ);env.update(PYTHONPATH=str(ROOT/'src'),MPLCONFIGDIR=str(ROOT/'mpl_cache'),
        OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
    if args.mode=='launch':
        command=[sys.executable,str(Path(__file__).resolve()),'run'];once(ROOT/'SERVER_LAUNCH_INTENT.json',dict(command=command,at=time.time()))
        with (ROOT/'SERVER_JOB.log').open('x') as log:
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        once(ROOT/'SERVER_LAUNCH.json',dict(pid=process.pid,at=time.time()));print(json.dumps(dict(status='LAUNCHED_ONCE',pid=process.pid)));return
    lock=(ROOT/'RUN.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);os.nice(10)
    once(ROOT/'SERVER_START.json',dict(host=platform.node(),pid=os.getpid(),python=sys.executable,at=time.time()));code=1
    try:
        import numpy,scipy,sklearn,xgboost,doubleml
        p=json.loads((ROOT/'configs/sensitivity.json').read_text());q=json.loads((ROOT/p['base_config']).read_text())
        identity=dict(protocol=p,base_protocol=q,source_hashes={f:digest(ROOT/f) for f in FILES},
            environment=dict(python=platform.python_version(),numpy=numpy.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,xgboost=xgboost.__version__,doubleml=doubleml.__version__))
        sha=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest();once(ROOT/'SOURCE_FROZEN.json',dict(identity=identity,identity_sha256=sha,at=time.time()))
        with (ROOT/'SERVER_PREFLIGHT.log').open('x') as log:
            test=subprocess.run([sys.executable,'-m','pytest','tests/test_sensitivity.py','-q'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        once(ROOT/'SERVER_PREFLIGHT.json',dict(returncode=test.returncode,log_sha256=digest(ROOT/'SERVER_PREFLIGHT.log')))
        if test.returncode:raise RuntimeError('software preflight failed; no scientific grid launched')
        with (ROOT/'DOUBLEML_CHECK.json').open('x') as stream:subprocess.run([sys.executable,'scripts/check_sensitivity_doubleml.py'],cwd=ROOT,env=env,stdout=stream,check=True)
        out=ROOT/'output/formal';out.mkdir(parents=True,exist_ok=False);once(out/'FROZEN_MANIFEST.json',dict(identity=identity,identity_sha256=sha,doubleml_check_sha256=digest(ROOT/'DOUBLEML_CHECK.json')))
        dispatch(p,q,out,sha)
        subprocess.run([sys.executable,'scripts/analyze_sensitivity.py'],cwd=ROOT,env=env,check=True)
        with (ROOT/'INDEPENDENT_VERIFICATION.json').open('x') as stream:subprocess.run([sys.executable,'scripts/verify_sensitivity.py'],cwd=ROOT,env=env,stdout=stream,check=True)
        with (ROOT/'SOFTWARE_REPLAY_CHECK.json').open('x') as stream:subprocess.run([sys.executable,'scripts/check_sensitivity_replay.py'],cwd=ROOT,env=env,stdout=stream,check=True)
        code=0
    finally:once(ROOT/'SERVER_EXIT.json',dict(returncode=code,pid=os.getpid(),at=time.time()))
    print('COMPLETE_VERIFIED_SENSITIVITY',flush=True)


if __name__=='__main__':main()
