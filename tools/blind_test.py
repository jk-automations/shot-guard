"""Blind test: run every check on takes Shot Guard was never tuned on, with specs taken from the prompts the takes
were generated from (fixtures/blind/index.json, private). Writes a private contact sheet per take plus a results file
to label (runs/blind/). Half of the takes are held out: tune on 'dev' only, report 'held-out' separately.
python tools/blind_test.py [--half dev|held-out|all] [--only KEY]"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import shotguard_core as sg  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--half", default="all", choices=["dev", "held-out", "all"])
ap.add_argument("--only")
ap.add_argument("--out", default=os.path.join(ROOT, "runs", "blind"))
a = ap.parse_args()
FX = os.path.join(ROOT, "fixtures")
CLIPS = os.environ.get("SHOTGUARD_CLIPS", os.path.join(FX, "blind", "clips"))   # the index paths are relative to this folder
index = json.load(open(os.path.join(FX, "blind", "index.json")))
os.makedirs(a.out, exist_ok=True)
for k, (key, item) in enumerate(sorted(index.items())):
    half = "dev" if k % 2 == 0 else "held-out"
    if (a.half != "all" and half != a.half) or (a.only and key not in a.only.split(",")):
        continue
    d = item["spec"]
    sp = sg.spec.make(**{x: d[x] for x in ("script", "allowed_text", "name") if x in d},
                      characters=[dict(c, refs=[os.path.join(FX, r) for r in c["refs"]]) for c in d["characters"]],
                      thresholds=d.get("thresholds"))
    t = time.time()
    r = sg.run(os.path.join(CLIPS, item["file"]), sp)
    sg.report.render(r, a.out, key, blur=False)
    row = {"key": key, "half": half, "verdict": r["verdict"], "seconds": round(time.time() - t, 1),
           "flags": [{"check": i["check"], "kind": i["kind"], "severity": i["severity"], "t0": i.get("t0"), "t1": i.get("t1"),
                      "message": i["message"], "truth": None} for i in r["issues"]],
           "transcript": r["checks"].get("dialogue", {}).get("metrics", {}).get("transcript", "")}
    json.dump(row, open(os.path.join(a.out, key + ".flags.json"), "w"), indent=1)
    print(f"{half:8s} {key:12s} {r['verdict']:5s} {row['seconds']:6.1f}s " +
          "; ".join(f"{f['check']}:{f['kind']}@{f['t0'] if f['t0'] is None else round(f['t0'], 1)}" for f in row["flags"]), flush=True)
