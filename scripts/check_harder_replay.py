#!/usr/bin/env python3
"""Exact software replay of existing pair receipts; not new MC evidence."""
import hashlib
import json
from pathlib import Path
import platform
import sys
import warnings
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from dml_original import harder_v5 as study


def main():
    assert platform.system()=='Linux' and str(ROOT)=='/data2/lzx/dhy/dml_harder_v5_20261002'
    root=ROOT/'output/exploratory';manifest=json.loads((root/'FROZEN_MANIFEST.json').read_text())
    assert json.loads((root/'STATUS.json').read_text())['status']=='complete'
    for name,sha in manifest['identity']['source_hashes'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha
    checks=[]
    with threadpool_limits(limits=1),warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        for design in ('figure1','figure2'):
            for dgp in ('linear','nonlinear'):
                prior=json.loads((root/f'{design}_{dgp}_0.json').read_text())['rows']
                replay=study.run_replication(manifest['identity']['protocol'],design,dgp,0)
                assert replay==prior,(design,dgp,'existing paired receipt differs')
                checks.append(dict(design=design,dgp=dgp,replication=0,rows=len(replay),exact_match=True))
    print(json.dumps(dict(status='PASS',purpose='software verification only; duplicates excluded from exploration statistics',
        checks=checks,warnings=[dict(category=w.category.__name__,message=str(w.message)) for w in captured],
        checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))


if __name__=='__main__':main()
