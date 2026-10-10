#!/usr/bin/env python3
"""Run/recover a hash-locked simulation; every replicate is a durable receipt."""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dml_studies.doubleml_reference import run_replication


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temp = path.with_suffix(path.suffix + ".pending")
    with temp.open("w") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def worker(protocol: dict, design: str, dgp: str, rep: int, development: bool) -> dict:
    from threadpoolctl import threadpool_limits
    started = time.monotonic()
    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        rows = run_replication(protocol, design, dgp, rep, development=development)
    return dict(rows=rows, elapsed_seconds=time.monotonic()-started,
                warnings=[dict(category=w.category.__name__, message=str(w.message)) for w in captured])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/doubleml_reference.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--development", action="store_true")
    parser.add_argument("--development-reps", type=int, default=3)
    args = parser.parse_args()
    if args.workers < 1 or args.development_reps < 1:
        parser.error("positive workers/reps required")
    protocol = json.loads(args.config.read_text())
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / "RUN.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    import numpy, scipy, sklearn, xgboost
    source_paths = [
        args.config.resolve(), ROOT / "src/dml_studies/doubleml_reference.py",
        ROOT / "src/dml_studies/iv_score.py",
        ROOT / "src/dml_studies/__init__.py", ROOT / "src/dml_studies/partialling_out.py",
        Path(__file__).resolve(),
        ROOT / "scripts/analyze_reference.py", ROOT / "docs/doubleml_reference_protocol.md",
        ROOT / "tests/test_doubleml_reference.py",
        *sorted((ROOT / "legacy_snapshot").glob("*.py")),
    ]
    identity = dict(protocol=protocol, development=args.development,
                    development_reps=args.development_reps if args.development else None,
                    source_hashes={str(p.relative_to(ROOT)): digest(p) for p in source_paths},
                    environment=dict(python=platform.python_version(), numpy=numpy.__version__,
                                     scipy=scipy.__version__, sklearn=sklearn.__version__, xgboost=xgboost.__version__))
    identity_sha = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    manifest_path = out / "FROZEN_MANIFEST.json"
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text())
        if prior["identity"] != identity or prior["identity_sha256"] != identity_sha:
            raise RuntimeError("source/config/environment differs from frozen identity")
    else:
        atomic_json(manifest_path, dict(identity=identity, identity_sha256=identity_sha,
                                       frozen_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
    universe = [(design, dgp, rep) for design in ("figure1", "figure2")
                for dgp in ("linear", "nonlinear")
                for rep in range(min(args.development_reps, protocol["replications"][design])
                                 if args.development else protocol["replications"][design])]
    completed = {}
    jobs = []
    for design, dgp, rep in universe:
        cell_dir = out / f"{design}_{dgp}"
        cell_dir.mkdir(exist_ok=True)
        path = cell_dir / f"rep_{rep:04d}.json"
        if path.exists():
            artifact = json.loads(path.read_text())
            if artifact["identity_sha256"] != identity_sha or artifact["key"] != [design, dgp, rep]:
                raise RuntimeError(f"incompatible replication {path}")
            completed[str(path.relative_to(out))] = digest(path)
        else:
            jobs.append((design, dgp, rep, path))
    started = time.monotonic()
    def progress(status: str) -> None:
        atomic_json(out / "STATUS.json", dict(status=status, completed=len(completed),
                    expected=len(universe), identity_sha256=identity_sha,
                    updated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    current_invocation_seconds=time.monotonic()-started))
    progress("running")
    print(json.dumps(dict(identity=identity_sha, resumed=len(completed), pending=len(jobs))), flush=True)
    # Submit bounded batches: do not build a large serialized queue of X/Y arrays.
    with futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        pending = {}
        iterator = iter(jobs)
        def submit() -> None:
            job = next(iterator, None)
            if job:
                design, dgp, rep, path = job
                pending[executor.submit(worker, protocol, design, dgp, rep, args.development)] = job
        for _ in range(2*args.workers):
            submit()
        while pending:
            done, _ = futures.wait(pending, return_when=futures.FIRST_COMPLETED)
            for future in done:
                design, dgp, rep, path = pending.pop(future)
                try:
                    result = future.result()
                except BaseException as exc:
                    atomic_json(out / "ERROR.json", dict(key=[design,dgp,rep], error=repr(exc)))
                    progress("failed")
                    raise
                result.update(identity_sha256=identity_sha, key=[design,dgp,rep])
                atomic_json(path, result)
                completed[str(path.relative_to(out))] = digest(path)
                progress("running")
                if len(completed) % 10 == 0 or len(completed) == len(universe):
                    print(json.dumps(dict(completed=len(completed), expected=len(universe),
                                          seconds=round(time.monotonic()-started, 1))), flush=True)
                submit()
    atomic_json(out / "RESULT_LEDGER.json", completed)
    progress("complete")
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
