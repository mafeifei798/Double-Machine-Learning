"""Shared result I/O and Monte Carlo inference utilities."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from scipy import stats


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def atomic_json(path: Path, value: object) -> None:
    temp = path.with_suffix(path.suffix + ".pending")
    with temp.open("w") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)

def mean_interval(values):
    x = np.asarray(values, dtype=float)
    mean = float(x.mean())
    half = float(stats.t.ppf(.975,len(x)-1)*x.std(ddof=1)/math.sqrt(len(x))) if len(x)>1 else None
    return dict(mean=mean,low=mean-half if half is not None else None,
                high=mean+half if half is not None else None)

def wilson(success,total):
    z = float(stats.norm.ppf(.975)); p=success/total
    center = (p+z*z/(2*total))/(1+z*z/total)
    half = z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/(1+z*z/total)
    return dict(rate=p, low=center-half, high=center+half)

def improvement_p(values):
    x=np.array(values);sd=x.std(ddof=1)
    if sd==0:return 0. if x.mean()<0 else 1.
    return float(stats.t.cdf(x.mean()/(sd/np.sqrt(len(x))),len(x)-1))

def holm(values):
    order=np.argsort(values);adjusted=np.empty(len(values));running=0.
    for rank,index in enumerate(order):
        running=max(running,min(1.,(len(values)-rank)*values[index]));adjusted[index]=running
    return adjusted.tolist()
