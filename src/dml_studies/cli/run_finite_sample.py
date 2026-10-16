#!/usr/bin/env python3
"""Server-only, single-launch finite-sample simulation. No local model fits."""
from __future__ import annotations
import argparse
import concurrent.futures as cf
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
import warnings

try:
    from .execution_guard import require_execution_root
except ImportError:
    from execution_guard import require_execution_root

ROOT = Path(__file__).resolve().parents[3]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def once(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def atomic(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def guard(p):
    require_execution_root(ROOT)


def task(p, q, dgp, rep):
    # ML imports occur only in server workers, after the controller's root guard.
    sys.path.insert(0, str(ROOT / 'src'))
    from threadpoolctl import threadpool_limits
    from dml_studies.sensitivity import run_replication
    start = time.monotonic()
    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        rows = run_replication(p, q, 'figure1', dgp, p['anchor_n'], 2, 'base', rep)
    return {'rows': rows, 'seconds': time.monotonic() - start,
            'warnings': [{'category': x.category.__name__, 'message': str(x.message)} for x in captured]}


def software_checks(p, q):
    check_p = dict(p, seed=p['preflight_seed'], sample_sizes=[160], anchor_n=160, replications=1)
    jobs = []
    for dgp in ('linear', 'nonlinear'):
        value = task(check_p, q, dgp, 0)
        assert len(value['rows']) == 6
        for family in p['learners']:
            pair = {x['method']: x for x in value['rows'] if x['learner'] == family}
            a, b = pair['naive_ml'], pair['dml_crossfit']
            assert a['finite'] and b['finite']
            assert a['bank_identity'] == b['bank_identity']
            assert a['g_prediction_hash'] == b['g_prediction_hash']
            assert a['m_prediction_hash'] == b['m_prediction_hash']
            assert b['prediction_overlap_n'] == 0 and b['train_n_per_model'] == 80
            assert abs(sum(b[x] for x in ('weighted_oracle', 'weighted_first_order', 'weighted_second_order')) - b['bias']) < 1e-10
        jobs.append({'dgp': dgp, 'rows': len(value['rows']), 'seconds': value['seconds']})
    return {'status': 'PASS', 'software_only': True, 'bias_or_method_win_not_a_gate': True, 'jobs': jobs}


def run(p):
    guard(p)
    os.nice(10)
    lock = (ROOT / 'FINITE_SAMPLE_RUN.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    once(ROOT / 'FINITE_SAMPLE_START.json', {'host': platform.node(), 'pid': os.getpid(),
                                     'python': sys.executable, 'at': time.time(), 'local_model_fits': 0})
    result_code = 1
    start = time.monotonic()
    try:
        import numpy, scipy, sklearn, xgboost
        q = json.loads((ROOT / p['base_config']).read_text())
        files = sorted([*ROOT.joinpath('src').rglob('*.py'), ROOT / p['base_config'],
                        ROOT / 'src/dml_studies/configs/finite_sample.json', ROOT / 'src/dml_studies/cli/execution_guard.py',
                        Path(__file__).resolve()])
        identity = {'protocol': p, 'base_protocol': q,
                    'source_hashes': {str(f.relative_to(ROOT)): sha(f) for f in files},
                    'environment': {'python': platform.python_version(), 'numpy': numpy.__version__,
                                    'scipy': scipy.__version__, 'sklearn': sklearn.__version__, 'xgboost': xgboost.__version__}}
        identity_sha = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        once(ROOT / 'FINITE_SAMPLE_FROZEN.json', {'identity': identity, 'identity_sha256': identity_sha, 'at': time.time()})
        once(ROOT / 'FINITE_SAMPLE_PREFLIGHT.json', software_checks(p, q))
        out = ROOT / 'output/finite_sample'
        out.mkdir(parents=True, exist_ok=False)
        once(out / 'FROZEN_MANIFEST.json', {'identity': identity, 'identity_sha256': identity_sha})
        keys = [(d, r) for r in range(p['replications']) for d in ('linear', 'nonlinear')]
        ledger = {}
        def status(state):
            atomic(ROOT / 'FINITE_SAMPLE_STATUS.json', {'status': state, 'completed_jobs': len(ledger),
                   'expected_jobs': len(keys), 'seconds': time.monotonic() - start, 'identity_sha256': identity_sha})
        status('running')
        with cf.ProcessPoolExecutor(max_workers=p['workers']) as pool:
            iterator = iter(keys)
            pending = {}
            def submit():
                key = next(iterator, None)
                if key is not None:
                    pending[pool.submit(task, p, q, *key)] = key
            for _ in range(2 * p['workers']):
                submit()
            while pending:
                done, _ = cf.wait(pending, return_when=cf.FIRST_COMPLETED)
                for future in done:
                    dgp, rep = pending.pop(future)
                    value = future.result()
                    # Do not drop a learner or a failed estimate.
                    if len(value['rows']) != 6 or not all(x['finite'] for x in value['rows']):
                        once(out / 'FAILED_JOB.json', {'dgp': dgp, 'replication': rep, 'value': value})
                        raise RuntimeError('Numerical failure retained; no complete-case reporting')
                    name = f'figure1_{dgp}_n500_k2_base_rep{rep:04d}.json'
                    value.update(key=['figure1', dgp, 500, 2, 'base', rep], identity_sha256=identity_sha)
                    once(out / name, value)
                    ledger[name] = sha(out / name)
                    status('running')
                    if len(ledger) % 25 == 0:
                        print(json.dumps({'completed_jobs': len(ledger), 'total': len(keys), 'seconds': time.monotonic()-start}), flush=True)
                    submit()
        once(out / 'RESULT_LEDGER.json', ledger)
        status('complete')
        result_code = 0
    except BaseException as error:
        atomic(ROOT / 'FINITE_SAMPLE_ERROR.json', {'error': repr(error), 'traceback': traceback.format_exc()})
        raise
    finally:
        once(ROOT / 'FINITE_SAMPLE_EXIT.json', {'returncode': result_code, 'pid': os.getpid(),
                                        'seconds': time.monotonic()-start, 'at': time.time()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('launch', 'run'))
    args = parser.parse_args()
    p = json.loads((ROOT / 'src/dml_studies/configs/finite_sample.json').read_text())
    guard(p)
    if args.mode == 'run':
        run(p)
        return
    if (ROOT / 'FINITE_SAMPLE_START.json').exists() or (ROOT / 'output/finite_sample').exists():
        raise RuntimeError('Existing launch/results found; inspect receipts instead of relaunching')
    env = dict(os.environ)
    env.update(DML_EXECUTION_ROOT=str(ROOT), PYTHONDONTWRITEBYTECODE='1',
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
    command = [sys.executable, str(Path(__file__).resolve()), 'run']
    once(ROOT / 'FINITE_SAMPLE_LAUNCH_INTENT.json', {'command': command, 'at': time.time()})
    with (ROOT / 'FINITE_SAMPLE_SERVER.log').open('x') as log:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    once(ROOT / 'FINITE_SAMPLE_LAUNCH.json', {'pid': child.pid, 'at': time.time()})
    print(json.dumps({'status': 'LAUNCHED_ONCE_SERVER_ONLY', 'pid': child.pid}))


if __name__ == '__main__':
    main()
