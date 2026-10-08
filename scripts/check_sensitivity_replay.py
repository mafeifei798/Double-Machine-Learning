#!/usr/bin/env python3
"""Replay existing sensitivity receipts on server; never add to MC counts."""
import hashlib
import json
from pathlib import Path
import platform
import sys
import warnings
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from dml_original import sensitivity_v6 as s
from run_sensitivity_v6 import filename


def main():
    assert platform.system()=='Linux' and str(ROOT)=='/data2/lzx/dhy/dml_sensitivity_v6_20261002'
    root=ROOT/'output/formal';manifest=json.loads((root/'FROZEN_MANIFEST.json').read_text());identity=manifest['identity']
    for name,digest in identity['source_hashes'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
    checks=[]
    keys=[('figure1','linear',300,5,'base',0),('figure1','nonlinear',1000,2,'outcome_strong',0),
        ('figure2','linear',1000,2,'treatment_strong',0),('figure2','nonlinear',3000,5,'base',0)]
    with threadpool_limits(limits=1),warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        for key in keys:
            old=json.loads((root/filename(key)).read_text())['rows'];new=s.run_replication(identity['protocol'],identity['base_protocol'],*key)
            assert new==old,(key,'software replay mismatch');checks.append(dict(key=key,rows=len(new),exact_match=True))
    print(json.dumps(dict(status='PASS',purpose='software verification only; replay duplicates excluded from MC',checks=checks,
        warnings=[dict(category=w.category.__name__,message=str(w.message)) for w in captured]),indent=2))


if __name__=='__main__':main()
