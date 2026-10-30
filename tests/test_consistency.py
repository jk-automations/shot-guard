"""Same take, same verdict: running a check twice, in fresh processes, gives the same verdict, score and findings.
(All 8 real fixture takes were also run twice through every check on 2026-10-29: identical; docs/LESSONS.md.)"""
import json
import subprocess
import sys

from conftest import FFMPEG, ROOT, make_clip

RUN = ("import json,sys; sys.path.insert(0, %r); import shotguard_core as sg; "
       "r = sg.run(sys.argv[1], sg.spec.make(), ['cuts', 'glitch', 'audio', 'attention']); "
       "print(json.dumps([r['verdict'], r['score'], sorted((i['kind'], round(i.get('t0') or 0, 3), i['message']) for i in r['issues'])]))")


def _once(path):
    out = subprocess.run([sys.executable, "-c", RUN % str(ROOT), path], capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_two_fresh_runs_agree(tmp_path):
    a, b = make_clip(tmp_path / "a.mp4", 2.0, audio=True), make_clip(tmp_path / "b.mp4", 1.5, pattern="smptebars", audio=True)
    cut = tmp_path / "cut.mp4"
    subprocess.run([FFMPEG, "-nostdin", "-v", "error", "-y", "-i", str(a), "-i", str(b), "-filter_complex",
                    "[0:v][0:a][1:v][1:a]concat=n=2:v=1:a=1[v][a]", "-map", "[v]", "-map", "[a]", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", str(cut)], check=True)
    first, second = _once(str(cut)), _once(str(cut))
    assert first == second and first[0] == "RETRY"   # the cut is found both times, at the same second
