#!/usr/bin/env python3
"""Software replay of four existing receipts, NOT new statistical evidence."""
import hashlib
import json
from pathlib import Path
import platform
import sys
import warnings
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from dml_original import two_claims_v4 as study


def main():
    assert platform.system()=='Linux' and str(ROOT)=='/data2/lzx/dhy/dml_two_claims_v4_20261002'
    root=ROOT/'output/formal';manifest=json.loads((root/'FROZEN_MANIFEST.json').read_text())
    assert json.loads((root/'STATUS.json').read_text())['status']=='complete'
    for name,sha in manifest['identity']['source_hashes'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha
    p=manifest['identity']['protocol'];checks=[]
    with threadpool_limits(limits=1),warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        for design in ('figure1','figure2'):
            for dgp in ('linear','nonlinear'):
                prior=json.loads((root/f'{design}_{dgp}_0.json').read_text())['rows']
                actual=study.run_replication(p,design,dgp,0)
                assert actual==prior, (design,dgp,'receipt replay differs')
                checks.append(dict(design=design,dgp=dgp,replication=0,rows=len(actual),exact_match=True))
    print(json.dumps(dict(status='PASS',purpose='software verification only; duplicates excluded from Monte Carlo',
        checks=checks,warnings=[dict(category=w.category.__name__,message=str(w.message)) for w in captured],
        checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))


if __name__=='__main__':main()
