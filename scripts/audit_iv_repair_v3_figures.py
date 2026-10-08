#!/usr/bin/env python3
"""Post-run read-only layout audit; never rewrites frozen figures/results."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from analyze_iv_repair_v3 import plot

ROOT=Path(__file__).resolve().parents[1]


def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    root=ROOT/"output/formal"; out=root/"analysis"
    ledger=json.loads((root/"RESULT_LEDGER.json").read_text()); grouped=defaultdict(list)
    for path in ledger:
        for r in json.loads((root/path).read_text())["value"]:
            grouped[tuple(r[k] for k in ("design","dgp","track","learner","method"))].append(r)
    summary=json.loads((out/"summary.json").read_text())
    prior={p.name:digest(p) for p in out.glob("*.png")}
    records=[]
    def inspect(fig,path,**kwargs):
        fig.set_dpi(kwargs.get("dpi",160)); fig.canvas.draw(); renderer=fig.canvas.get_renderer()
        w,h=fig.canvas.get_width_height(); artists=list(fig.texts)
        for ax in fig.axes:
            artists += [ax.title,ax.xaxis.label,ax.yaxis.label]
            xmin,xmax=sorted(ax.get_xlim()); ymin,ymax=sorted(ax.get_ylim())
            artists += [label for loc,label in zip(ax.get_xticks(),ax.get_xticklabels()) if xmin<=loc<=xmax]
            artists += [label for loc,label in zip(ax.get_yticks(),ax.get_yticklabels()) if ymin<=loc<=ymax]
            legend=ax.get_legend()
            if legend: artists += legend.get_texts()
        violations=[]
        for artist in artists:
            if not artist.get_visible() or not artist.get_text(): continue
            bbox=artist.get_window_extent(renderer)
            if bbox.x0<-.5 or bbox.y0<-.5 or bbox.x1>w+.5 or bbox.y1>h+.5:
                violations.append(dict(text=artist.get_text(),bbox=list(bbox.extents)))
        records.append(dict(file=Path(path).name,width=w,height=h,visible_texts=len(artists),out_of_canvas=violations))
    with patch.object(Figure,"savefig",inspect): plot(out,grouped,summary)
    assert {r["file"] for r in records}==set(prior) and len(records)==14
    assert prior=={p.name:digest(p) for p in out.glob("*.png")},"frozen images changed"
    result=dict(status="PASS" if not any(r["out_of_canvas"] for r in records) else "LAYOUT_ISSUES",
        images=len(records),records=records,frozen_image_sha256=prior,formal_images_unchanged=True)
    target=ROOT/"FIGURE_QA.json"
    with target.open("x") as stream: json.dump(result,stream,indent=2)
    print(json.dumps(result,indent=2))


if __name__=="__main__": main()
