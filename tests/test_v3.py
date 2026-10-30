"""Audio check, annotated video, auto-trim and best-of-N on synthetic clips (no fixtures needed)."""
import subprocess

import cv2
import numpy as np

import shotguard_core as sg
from conftest import FFMPEG, make_clip

SCRIPT = [{"speaker": "A", "text": "hello there"}]


def tone(path, secs=4.0, vol=0.3, gap=None, noise=0.0, end_loud=False):
    """A clip whose sound we control: a tone, optional silence gap, optional noise bed, optional loud ending."""
    sr = 48000
    t = np.arange(int(secs * sr)) / sr
    x = vol * np.sin(2 * np.pi * 220 * t) * (np.sin(2 * np.pi * 1.5 * t) > 0)   # speech-like: sound, pause, sound
    if gap:
        x[int(gap[0] * sr):int(gap[1] * sr)] = 0
    if noise:
        x = x + noise * np.random.default_rng(0).standard_normal(len(x))
    if end_loud:
        x[-int(0.5 * sr):] = vol * np.sin(2 * np.pi * 220 * t[-int(0.5 * sr):])   # still talking on the last frame
    else:
        x[-int(0.3 * sr):] = 0
    wav = str(path) + ".wav"
    import wave
    with wave.open(wav, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())
    v = make_clip(str(path) + ".v.mp4", secs)
    subprocess.run([FFMPEG, "-nostdin", "-v", "error", "-y", "-i", v, "-i", wav, "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                    "-shortest", str(path)], check=True)
    return str(path)


def kinds(r):
    return {i["kind"]: i["severity"] for i in r["issues"]}


def test_audio_clean_take_passes(tmp_path):
    r = sg.audio.check(tone(tmp_path / "a.mp4"), sg.spec.make(script=SCRIPT))
    assert r["status"] == "pass" and r["issues"] == []


def test_audio_finds_clipping_and_dead_air(tmp_path):
    assert kinds(sg.audio.check(tone(tmp_path / "c.mp4", vol=3.0), sg.spec.make(script=SCRIPT)))["clipping"] == "major"
    assert "dead_air" in kinds(sg.audio.check(tone(tmp_path / "g.mp4", gap=(1.0, 3.4)), sg.spec.make(script=SCRIPT)))
    assert kinds(sg.audio.check(tone(tmp_path / "e.mp4", end_loud=True), sg.spec.make(script=SCRIPT))) == {}   # talk or music on the last frame is fine


def test_audio_music_bed_and_silent_track(tmp_path):
    assert "music_over_voice" in kinds(sg.audio.check(tone(tmp_path / "m.mp4", noise=0.2), sg.spec.make(script=SCRIPT)))
    silent = sg.audio.check(tone(tmp_path / "s.mp4", vol=0.0), sg.spec.make(script=SCRIPT))
    assert silent["status"] == "fail" and "silent_track" in kinds(silent)
    assert sg.audio.check(tone(tmp_path / "s2.mp4", vol=0.0), sg.spec.make())["status"] == "pass"   # no script: silence is fine


def test_audio_skips_video_without_sound(tmp_path):
    assert sg.audio.check(str(make_clip(tmp_path / "n.mp4")), sg.spec.make(script=SCRIPT))["status"] == "skip"


def _two_shot(tmp_path, a=3.0, b=1.0):
    p1, p2 = make_clip(tmp_path / "p1.mp4", a), make_clip(tmp_path / "p2.mp4", b, pattern="smptebars")
    out = tmp_path / "cut.mp4"
    subprocess.run([FFMPEG, "-nostdin", "-v", "error", "-y", "-i", p1, "-i", p2, "-filter_complex", "[0:v][1:v]concat=n=2:v=1[v]",
                    "-map", "[v]", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)], check=True)
    return str(out)


def test_trim_keeps_longest_clean_piece(tmp_path):
    clip = _two_shot(tmp_path)
    r = sg.run(clip, sg.spec.make(), which=["cuts"])
    assert r["verdict"] == "RETRY"
    p = sg.trim.plan(r)
    assert p["ok"] and p["t0"] == 0.0 and 2.8 < p["t1"] < 3.0
    out = sg.trim.cut(clip, p["t0"], p["t1"], str(tmp_path / "t.mp4"))
    assert sg.run(out, sg.spec.make(), which=["cuts"])["verdict"] == "PASS"


def test_trim_removes_the_whole_transition_not_just_its_peak():
    # measured on a real take: a 0.5 s transition (5.33 to 5.83 s) peaking at 5.50 s, then two more cuts
    r = {"video": "x.mp4", "issues": [
        {"kind": "hard_cut", "severity": "major", "t0": 5.33, "t1": 5.83, "frame_t": 5.50, "message": "cut"},
        {"kind": "hard_cut", "severity": "major", "t0": 5.92, "t1": 5.92, "frame_t": 5.92, "message": "cut"}]}
    import shotguard_core.trim as tr
    from types import SimpleNamespace
    orig = tr.read_frames
    tr.read_frames = lambda *a, **k: SimpleNamespace(duration=8.0, fps=24.0)
    try:
        p = tr.plan(r)
    finally:
        tr.read_frames = orig
    assert p["ok"] and p["t0"] == 0.0 and p["t1"] < 5.33


def test_trim_refuses_short_pieces_and_clean_takes(tmp_path):
    r = sg.run(_two_shot(tmp_path, 1.5, 1.5), sg.spec.make(), which=["cuts"])
    assert sg.trim.plan(r)["ok"] is False
    assert sg.trim.plan(sg.run(str(make_clip(tmp_path / "ok.mp4", 3.0)), sg.spec.make(), which=["cuts"])) is None


def test_best_of_n_prefers_pass_then_score():
    def res(passed, score):
        return {"passed": passed, "score": score, "verdict": "PASS" if passed else "RETRY", "issues": [], "video": "x"}
    rs = [res(False, 90), res(True, 60), res(True, 80), res(True, 80)]
    ranked, order = sg.best.rank(rs)
    assert order == [2, 3, 1, 0]
    assert sg.best.summary(rs).startswith("1. take 3: PASS")


def test_annotated_video_has_hud_hold_and_sound(tmp_path):
    clip = tone(tmp_path / "a.mp4", secs=3.0)
    r = sg.run(clip, sg.spec.make(script=SCRIPT), which=["cuts", "audio"])
    r["issues"].append({"kind": "hard_cut", "severity": "major", "t0": 1.0, "t1": 1.0, "frame_t": 1.0, "check": "cuts",
                        "message": "hard cut at 1.00 s"})
    out = sg.annotate.render(r, str(tmp_path / "ann.mp4"), blur=False)
    cap = cv2.VideoCapture(out)
    n, w, h = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 26)          # 1.08 s: inside the cut flash
    ok, img = cap.read()
    cap.release()
    assert h > 640 and w == 360                   # picture plus the timeline strip under it
    assert n >= 3.0 * 24 + 30                     # the take plus the verdict hold
    assert ok and img[2:6, 100:260, 2].mean() > 120 and img[2:6, 100:260, 0].mean() < 110   # red border on the cut
    assert sg.video.has_audio(out)
