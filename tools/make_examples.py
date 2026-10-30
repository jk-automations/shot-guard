"""Example contact sheets for docs/examples/ from the local fixtures (fixtures/, gitignored).
Public-safe: names in the private specs are swapped for role names, and every face is blurred on takes that show
real-person characters. Receipts with the real names stay in runs/examples/ (gitignored).
python tools/make_examples.py [--only NAME]"""
import argparse
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import shotguard_core as sg  # noqa: E402

FX = os.path.join(ROOT, "fixtures")
PRIVATE = os.path.join(FX, "public_names.json")   # {"real name": "role name"}, gitignored with fixtures/
ap = argparse.ArgumentParser()
ap.add_argument("--only")
a = ap.parse_args()
names = json.load(open(PRIVATE))
blur_takes = set(names.pop("_blur_takes", []))
pat = re.compile("|".join(re.escape(k) for k in sorted(names, key=len, reverse=True)), re.I)   # transcripts are lower case
lower = {k.lower(): v for k, v in names.items()}


def scrub(o):
    if isinstance(o, str):
        return pat.sub(lambda m: lower[m.group().lower()], o)
    if isinstance(o, dict):
        return {scrub(k) if isinstance(k, str) else k: scrub(v) for k, v in o.items()}
    if isinstance(o, list):
        return [scrub(v) for v in o]
    if isinstance(o, tuple):
        return tuple(scrub(v) for v in o)
    return o


index = json.load(open(os.path.join(FX, "fixtures.json")))
pub, priv = os.path.join(ROOT, "docs", "examples"), os.path.join(ROOT, "runs", "examples")
os.makedirs(pub, exist_ok=True)
rows = []
for name, fx in index.items():
    if a.only and name not in a.only.split(","):
        continue
    sp = sg.spec.load(os.path.join(FX, "specs", name + ".json"), base_dir=FX)
    r = sg.run(os.path.join(FX, fx["file"]), sp)
    sg.report.render(r, priv, name, blur=name in blur_takes)                 # private receipt, real names
    clean = scrub({k: v for k, v in r.items() if k != "spec"})
    clean["spec"] = dict(scrub(sg.spec.public(sp)), characters=scrub(sg.spec.public(sp))["characters"], thresholds=sp["thresholds"])
    clean["spec"]["characters"] = [dict(c, refs=[]) for c in clean["spec"]["characters"]]
    clean["video"] = r["video"]
    _, png, js = sg.report.render(clean, pub, name, blur=name in blur_takes)
    os.remove(js)                                                            # contact sheets only in docs/
    flagged = sorted({i["check"] for i in r["issues"] if i["severity"] == "major"})
    rows.append((name, fx["must_flag"], r["verdict"], flagged))
    print(f"{name:20s} {r['verdict']:5s} flagged={flagged} -> {os.path.relpath(png, ROOT)}")
leaks = [n for n in names for f in os.listdir(pub) if f.endswith(".png") and n.lower() in f.lower()]
print("name leaks in file names:", leaks)
