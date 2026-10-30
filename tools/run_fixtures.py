"""Run every check on every local fixture (fixtures/, gitignored) and print fixture -> verdict.
python tools/run_fixtures.py [--only NAME] [--checks cuts,text] [--json out.json]"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import shotguard_core as sg  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--only")
ap.add_argument("--json")
ap.add_argument("--checks")
a = ap.parse_args()
FX = os.path.join(ROOT, "fixtures")
index = json.load(open(os.path.join(FX, "fixtures.json")))
out = {}
for name, fx in index.items():
    if a.only and name not in a.only.split(","):
        continue
    t = time.time()
    sp = sg.spec.load(os.path.join(FX, "specs", name + ".json"), base_dir=FX)
    r = sg.run(os.path.join(FX, fx["file"]), sp, a.checks.split(",") if a.checks else None)
    flagged = sorted({i["check"] for i in r["issues"] if i["severity"] == "major"})
    ok = (fx["must_flag"] in flagged) if fx["must_flag"] else not flagged
    out[name] = {"verdict": r["verdict"], "flagged": flagged, "must_flag": fx["must_flag"], "ok": ok,
                 "seconds": round(time.time() - t, 1), "hints": r["hints"],
                 "issues": [f"{i['check']}: {i['message']}" for i in r["issues"]]}
    print(f"{'OK ' if ok else 'BAD'} {name:20s} {r['verdict']:5s} flagged={flagged} want={fx['must_flag']} {out[name]['seconds']}s")
    for i in out[name]["issues"]:
        print("     -", i)
if a.json:
    json.dump(out, open(a.json, "w"), indent=1)
