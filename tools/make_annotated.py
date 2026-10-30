"""Annotated videos, auto-trim and best-of-N from the local fixtures, for the demo (runs/annotated/, gitignored).
Public-safe like tools/make_examples.py: role names instead of real names, and on takes that show real-person
characters every face is blurred and the sound is removed (their voices say real names).
--real keeps real names, faces and voices (the hosts agreed to be shown in the demo): runs/annotated-real/, never committed.
python tools/make_annotated.py [--only NAME] [--real]"""
import argparse
import json
import pickle
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import shotguard_core as sg  # noqa: E402

FX, OUT = os.path.join(ROOT, "fixtures"), os.path.join(ROOT, "runs", "annotated-real" if "--real" in sys.argv else "annotated")
ap = argparse.ArgumentParser()
ap.add_argument("--only")
ap.add_argument("--real", action="store_true")
ap.add_argument("--reuse", action="store_true", help="re-draw from saved check results instead of running the checks again")
a = ap.parse_args()
names = json.load(open(os.path.join(FX, "public_names.json")))
blur_takes = set(names.pop("_blur_takes", [])) if not a.real else set()
pat = re.compile("|".join(re.escape(k) for k in sorted(names, key=len, reverse=True)), re.I)
lower = {k.lower(): v for k, v in names.items()}


def scrub(o):
    if isinstance(o, str):
        return pat.sub(lambda m: lower[m.group().lower()], o)
    if isinstance(o, dict):
        return {k: scrub(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return type(o)(scrub(v) for v in o)
    return o


def public(r):
    if a.real:
        return r
    clean = scrub({k: v for k, v in r.items() if k not in ("spec", "video")})
    clean["spec"], clean["video"] = r["spec"], r["video"]
    return clean


def mute(path):
    tmp = path + ".mute.mp4"
    subprocess.run([sg.video.ffmpeg_bin(), "-nostdin", "-v", "error", "-y", "-i", path, "-an", "-c", "copy", tmp], check=True)
    os.replace(tmp, path)


def annotate(name, path, sp, tag="", note=None):
    cache = os.path.join(OUT, f"{name}{tag}.result.pkl")
    if a.reuse and os.path.exists(cache):
        r = pickle.load(open(cache, "rb"))
    else:
        r = sg.run(path, sp)
        r["spec"] = {k: v for k, v in r["spec"].items() if k != "_cache"}
        pickle.dump(r, open(cache, "wb"))
    if note and r["passed"]:
        r["note"] = note
    out = os.path.join(OUT, f"{name}{tag}.mp4")
    sg.annotate.render(public(r), out, blur=name in blur_takes)
    if name in blur_takes:
        mute(out)
    print(f"{name + tag:24s} {r['verdict']:5s} score {r['score']:5} -> {os.path.relpath(out, ROOT)}", flush=True)
    return r


os.makedirs(OUT, exist_ok=True)
index = json.load(open(os.path.join(FX, "fixtures.json")))
results = {}
for name, fx in index.items():
    if fx["file"].endswith((".jpg", ".png")) or (a.only and name not in a.only.split(",")):
        continue
    sp = sg.spec.load(os.path.join(FX, "specs", name + ".json"), base_dir=FX)
    path = os.path.join(FX, fx["file"])
    r = results[name] = annotate(name, path, sp)
    p = sg.trim.plan(r, sp["thresholds"]["trim_min_s"])
    if p and p.get("seconds"):
        trimmed = sg.trim.cut(path, p["t0"], p["t1"], os.path.join(OUT, f"{name}-trimmed-src.mp4"))
        r2 = annotate(name, trimmed, sp, "-trimmed", note="fixed by auto-trim: no new generation, 1 retry saved")
        print(f"  auto-trim: {p['note']} -> re-checked {r2['verdict']} (cuts check on the piece: {sg.trim.recheck(trimmed, sp)['status']})", flush=True)
        results[name + "-trimmed"] = r2
# The retry demo's next take of the side-swap shot (same scene, seed 1), and best-of-N over every take of that shot
good = os.path.join(ROOT, ".comfy-base", "input", "shotguard_good.mp4")
if os.path.exists(good) and "side_swap" in results:
    sp = sg.spec.load(os.path.join(FX, "specs", "side_swap.json"), base_dir=FX)
    sp["attempt"] = 1
    results["side_swap-attempt1"] = annotate("side_swap", good, sp, "-attempt1")
    takes = [k for k in ("side_swap", "side_swap-trimmed", "side_swap-attempt1") if k in results]
    ranking = sg.best.summary([results[k] for k in takes])
    print("best of N (" + ", ".join(takes) + "):\n" + scrub(ranking))
    json.dump({"takes": takes, "ranking": scrub(ranking)}, open(os.path.join(OUT, "best_of.json"), "w"), indent=1)
# Glitch has no real failing take (none of the 24 real takes froze, stuttered or flickered), so the demo shows a
# good take with one second frozen on purpose, labelled as simulated
if not a.only or "glitch_demo" in a.only.split(","):
    src = os.path.join(FX, "good_W1.mp4")
    frozen = os.path.join(OUT, "glitch_demo-src.mp4")
    subprocess.run([sg.video.ffmpeg_bin(), "-nostdin", "-v", "error", "-y", "-i", src, "-vf", "loop=loop=29:size=1:start=72,trim=duration=8,setpts=PTS-STARTPTS",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-c:a", "copy", frozen], check=True)
    results["glitch_demo"] = annotate("glitch_demo", frozen, sg.spec.load(os.path.join(FX, "specs", "good_W1.json"), base_dir=FX))
summary = {k: {"verdict": v["verdict"], "score": v["score"]} for k, v in results.items()}
json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
