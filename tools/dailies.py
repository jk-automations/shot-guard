"""Dailies from the command line: check every take in a folder, rank them, write one page.
python tools/dailies.py FOLDER [SPEC.json] [--out DIR] [--no-blur] [--no-trim]
A take with its own <name>.json next to it uses that spec; the others use SPEC.json (or an empty spec)."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import shotguard_core as sg  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("folder")
ap.add_argument("spec", nargs="?")
ap.add_argument("--out")
ap.add_argument("--no-blur", action="store_true")
ap.add_argument("--no-trim", action="store_true")
ap.add_argument("--title", help="page title (default: the folder name)")
a = ap.parse_args()
d = sg.batch.run_folder(a.folder, sg.spec.load(a.spec) if a.spec else None, out_dir=a.out, auto_trim=not a.no_trim)
_, png, js = sg.batch.render(d, a.out or os.path.join(a.folder, "shot_guard"), blur=not a.no_blur, title=a.title)
print(sg.batch.summary(d))
print(png)
