"""Speaker, Glitch, credits meter and Dailies (synthetic clips and mocked measurements, no fixtures needed)."""
import subprocess

import shotguard_core as sg
from conftest import FFMPEG, make_clip


def clip(path, vf, secs=3.0):
    """An ffmpeg test pattern with a filter that adds one fault."""
    subprocess.run([FFMPEG, "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=size=360x640:rate=24:duration={secs}",
                    "-vf", vf, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", str(path)], check=True)
    return str(path)


def kinds(r):
    return {i["kind"]: i["severity"] for i in r["issues"]}


def test_glitch_clean_take_passes(tmp_path):
    r = sg.glitch.check(clip(tmp_path / "c.mp4", "null"), sg.spec.make())
    assert r["status"] == "pass" and r["issues"] == []


def test_glitch_finds_freeze_stutter_and_flicker(tmp_path):
    assert kinds(sg.glitch.check(clip(tmp_path / "f.mp4", "loop=loop=30:size=1:start=24,trim=duration=3"), sg.spec.make())) == {"frozen": "major"}
    assert "stutter" in kinds(sg.glitch.check(clip(tmp_path / "s.mp4", "fps=16,fps=24"), sg.spec.make()))
    assert "flicker" in kinds(sg.glitch.check(clip(tmp_path / "l.mp4", "eq=brightness='0.06*(2*mod(n\\,2)-1)':eval=frame"), sg.spec.make()))


def _speaker_spec():
    return sg.spec.make(script=[{"speaker": "Ana", "text": "the bread is fresh today"}, {"speaker": "Ben", "text": "then I will take two loaves"}],
                        characters=[{"name": "Ana", "refs": ["a.png"]}, {"name": "Ben", "refs": ["b.png"]}])


def _mock(monkeypatch, ana, ben):
    segs = [{"t0": 0.0, "t1": 2.0, "text": "The bread is fresh today."}, {"t0": 2.2, "t1": 4.0, "text": "Then I will take two loaves."}]
    pts = lambda f: [(k / 24, f(k / 24), 0.3, (10, 10, 50, 50)) for k in range(96)]
    monkeypatch.setattr(sg.speaker, "segments", lambda p, s: segs)
    monkeypatch.setattr(sg.speaker, "mouth_motion", lambda p, s: {"Ana": pts(ana), "Ben": pts(ben)})
    monkeypatch.setattr(sg.speaker, "is_image", lambda p: False)


def test_speaker_right_people_pass(monkeypatch):
    _mock(monkeypatch, lambda t: 0.5 if t < 2.1 else 0.2, lambda t: 0.2 if t < 2.1 else 0.5)
    r = sg.speaker.check("x.mp4", _speaker_spec())
    assert r["status"] == "pass" and [L["verdict"] for L in r["metrics"]["lines"]] == ["ok", "ok"]


def test_speaker_wrong_person_fails_with_a_hint(monkeypatch):
    _mock(monkeypatch, lambda t: 0.2, lambda t: 0.5)   # Ben moves his mouth through Ana's line
    sp = _speaker_spec()
    r = sg.speaker.check("x.mp4", sp)
    bad = [i for i in r["issues"] if i["kind"] == "wrong_speaker"]
    assert r["status"] == "fail" and bad[0]["character"] == "Ana" and bad[0]["speaking"] == "Ben" and bad[0]["severity"] == "major"
    g = sg.gate.decide(sp, [r])
    assert g["verdict"] == "RETRY" and "only Ana speaks line 1" in g["prompt_hint"]


def test_speaker_skips_without_two_referenced_characters():
    sp = sg.spec.make(script=[{"speaker": "Ana", "text": "hi"}], characters=[{"name": "Ana", "refs": ["a.png"]}])
    assert sg.speaker.check("x.mp4", sp)["status"] == "skip"


def test_credits_meter_counts_caught_and_saved():
    sp = sg.spec.make(cost_per_take=0.5)
    rs = [sg.gate.decide(sp, [{"check": "cuts", "status": "fail", "issues": [{"kind": "hard_cut", "severity": "major", "message": "cut"}]}]),
          sg.gate.decide(sp, [{"check": "cuts", "status": "pass", "issues": []}])]
    m = sg.credits.meter(rs, trim_fixes=1)
    assert (m["takes"], m["caught"], m["spent"], m["saved"]) == (2, 1, 1.0, 0.5)
    assert "1 fixed by auto-trim: 0.5 credits saved" in m["line"]
    free = sg.credits.meter([dict(rs[0], spec=sg.spec.make())], trim_fixes=1)
    assert "1 take saved" in free["line"] and free["saved"] == 0.0


def test_dailies_ranks_a_folder_and_auto_trims(tmp_path):
    good = make_clip(tmp_path / "a_good.mp4", 3.0)
    p1, p2 = make_clip(tmp_path / "p1.mp4", 3.0), make_clip(tmp_path / "p2.mp4", 0.5, pattern="smptebars")
    subprocess.run([FFMPEG, "-nostdin", "-v", "error", "-y", "-i", str(p1), "-i", str(p2), "-filter_complex", "[0:v][1:v]concat=n=2:v=1[v]",
                    "-map", "[v]", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(tmp_path / "b_cut.mp4")], check=True)
    p1.unlink(); p2.unlink()
    d = sg.batch.run_folder(str(tmp_path), sg.spec.make(), out_dir=str(tmp_path / "out"), which=["cuts", "glitch"], log=lambda s: None)
    assert [x["take"] for x in d["rows"]] == ["a_good.mp4", "b_cut.mp4"]
    assert d["rows"][1]["fixed_by_trim"] and d["rows"][1]["result"]["passed"]
    assert d["meter"]["caught"] == 1 and d["meter"]["retries_saved"] == 1
    _, png, js = sg.batch.render(d, str(tmp_path / "out"), blur=False)
    assert png.endswith(".png") and "auto-trim" in sg.batch.summary(d)


def test_attention_scores_hook_and_flags_a_dead_take(tmp_path):
    busy = sg.attention.check(clip(tmp_path / "b.mp4", "null", secs=6.0), sg.spec.make())
    dead = sg.attention.check(clip(tmp_path / "d.mp4", "trim=end_frame=1,loop=loop=144:size=1:start=0,setpts=N/24/TB", secs=6.0), sg.spec.make())
    assert busy["status"] == dead["status"] == "pass"   # warnings only, never a fail
    assert busy["metrics"]["hook"] > dead["metrics"]["hook"]
    assert {"slow_hook", "dead_stretch"} <= set(kinds(dead))
    g = sg.gate.decide(sg.spec.make(), [dead])
    assert g["verdict"] == "PASS" and "open on movement" in g["prompt_hint"]
