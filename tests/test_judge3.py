"""Stillness, Places and Captions on synthetic clips (no fixtures, no model downloads, no network beyond a local fake)."""
import json
import os
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

import shotguard_core as sg
from conftest import FFMPEG


def ff(*args):
    subprocess.run([FFMPEG, "-nostdin", "-v", "error", "-y", *args], check=True)


@pytest.fixture(scope="module")
def clips(tmp_path_factory):
    d = tmp_path_factory.mktemp("j3")
    ff("-f", "lavfi", "-i", "testsrc2=size=720x1280,noise=alls=40:allf=0", "-frames:v", "1", str(d / "still.png"))
    pan = "zoompan=z='1+0.0015*on':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s=360x640:fps=24"
    ff("-loop", "1", "-i", str(d / "still.png"), "-vf", pan, "-t", "3", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "14", str(d / "pan.mp4"))
    ff("-loop", "1", "-i", str(d / "still.png"), "-vf", pan + ",lutyuv=y=val/12", "-t", "3", "-c:v", "libx264",
       "-pix_fmt", "yuv420p", "-crf", "14", str(d / "dark.mp4"))
    ff("-f", "lavfi", "-i", "life=s=360x640:mold=10:r=24:ratio=0.3:death_color=#202020:life_color=#e0e0e0,format=yuv420p", "-t", "3",
       "-c:v", "libx264", "-crf", "14", str(d / "moving.mp4"))
    # 3 s of real motion, then 3 s of the panned still: one photo-like stretch at the end
    ff("-i", str(d / "moving.mp4"), "-i", str(d / "pan.mp4"), "-filter_complex", "[0:v][1:v]concat=n=2:v=1[v]", "-map", "[v]",
       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "14", str(d / "mixed.mp4"))
    # two colour shots of 2 s with a hard cut: red then red (same), red then blue (changed)
    for name, b in (("same", "red"), ("changed", "blue")):
        ff("-f", "lavfi", "-i", "color=red:s=360x640:r=24:d=2", "-f", "lavfi", "-i", f"color={b}:s=360x640:r=24:d=2",
           "-filter_complex", "[0:v]noise=alls=20:allf=t[a];[1:v]noise=alls=20:allf=t[b];[a][b]concat=n=2:v=1[v]", "-map", "[v]",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", str(d / f"{name}.mp4"))
    return {k: str(d / f"{k}.mp4") for k in ("pan", "dark", "moving", "mixed", "same", "changed")}


def kinds(r):
    return {i["kind"]: i["severity"] for i in r["issues"]}


# ---------------------------------------------------------------- stillness

def test_panned_photo_warns_fails_against_a_reference_and_real_motion_passes(clips):
    pan = sg.stillness.check(clips["pan"], sg.spec.make())
    assert pan["status"] == "pass" and kinds(pan) == {"photo_like": "minor", "too_many_stills": "minor"}   # warnings alone
    vs_ref = sg.stillness.check(clips["pan"], sg.spec.make(motion_reference={"median_residual": 12.0, "photo_share": 0.05}))
    assert vs_ref["status"] == "fail" and kinds(vs_ref)["below_reference_motion"] == "major"
    assert pan["metrics"]["photo_share"] == 1.0 and "camera move" in pan["issues"][0]["message"]
    moving = sg.stillness.check(clips["moving"], sg.spec.make())
    assert moving["status"] == "pass" and moving["issues"] == [] and moving["metrics"]["median_residual"] > 8


def test_stretch_is_timed(clips):
    r = sg.stillness.check(clips["mixed"], sg.spec.make())
    st = [i for i in r["issues"] if i["kind"] == "photo_like"]
    assert len(st) == 1 and 2.8 <= st[0]["t0"] <= 3.3 and st[0]["t1"] >= 5.7
    assert r["metrics"]["stretches"] == [[round(st[0]["t0"], 2), round(st[0]["t1"], 2)]]


def test_dark_still_is_reported_not_counted(clips):
    r = sg.stillness.check(clips["dark"], sg.spec.make())
    assert "too_dark_to_judge" in kinds(r) and "too_many_stills" not in kinds(r) and r["status"] == "pass"
    assert r["metrics"]["dark_share"] > 0.9 and r["metrics"]["photo_share"] is None


def test_static_ok_span_is_ignored(clips):
    r = sg.stillness.check(clips["mixed"], sg.spec.make(static_ok=[[3.0, 6.0]]))
    assert r["status"] == "pass" and r["issues"] == []


def test_against_a_reference(clips):
    ref = sg.stillness.reference_profile({"path": clips["moving"]}, sg.spec.make()["thresholds"])
    assert ref["photo_share"] == 0.0 and ref["median_residual"] > 8
    r = sg.stillness.check(clips["mixed"], sg.spec.make(motion_reference={"path": clips["moving"]}))
    assert kinds(r)["below_reference_motion"] == "major" and r["metrics"]["reference"]["share_gap"] >= 0.2
    same = sg.stillness.check(clips["moving"], sg.spec.make(motion_reference={"median_residual": 10.0, "photo_share": 0.0}))
    assert same["status"] == "pass" and same["metrics"]["reference"]["source"] == "spec numbers"
    g = sg.gate.decide(sg.spec.make(), [r])
    assert g["verdict"] == "RETRY" and any("not a still image with a zoom or pan" in h for h in g["hints"])


# ---------------------------------------------------------------- places

def colour_embedder(img):
    v = img.reshape(-1, 3).mean(0).astype(float) + 1.0
    return v / np.linalg.norm(v)


def test_places_skips_without_groups(clips):
    assert sg.places.check(clips["same"], sg.spec.make())["status"] == "skip"


def test_places_same_object_passes_and_changed_object_fails(clips):
    sp = sg.spec.make(continuity_groups=[{"name": "food truck", "spans": [[0.2, 1.8], [2.2, 3.8]]}])
    ok = sg.places.check(clips["same"], sp, embedder=colour_embedder)
    assert ok["status"] == "pass" and ok["metrics"]["groups"][0]["pairs"][0]["similarity"] > 0.99
    bad = sg.places.check(clips["changed"], sp, embedder=colour_embedder)
    i = bad["issues"][0]
    assert bad["status"] == "fail" and i["kind"] == "continuity_changed" and i["severity"] == "major"
    assert "food truck" in i["message"] and "0.20-1.80 s" in i["message"]
    assert bad["metrics"]["shots"][1] == pytest.approx(2.0, abs=0.05)   # Shot Guard's cut score found the cut
    assert [m["shot"] for m in bad["metrics"]["groups"][0]["spans"]] == [0, 1]
    g = sg.gate.decide(sp, [bad])
    assert "use one locked reference" in g["hints"][0]


def test_places_frames_stay_clear_of_the_cut(clips):
    sp = sg.spec.make(continuity_groups=[{"name": "x", "spans": [[1.5, 2.5], [0.1, 0.5]]}], thresholds={"places_cut_guard": 0.2})
    r = sg.places.check(clips["changed"], sp, embedder=colour_embedder)
    assert all(abs(t - 2.0) >= 0.2 for t in r["metrics"]["groups"][0]["spans"][0]["frames"])


def test_places_rejects_a_backwards_span():
    with pytest.raises(ValueError):
        sg.spec.make(continuity_groups=[{"name": "x", "spans": [[3, 1], [4, 5]]}])


# ---------------------------------------------------------------- captions

CAPS = [{"text": "It's a truck.", "t0": 0.2, "t1": 1.8}, {"text": "Yep.", "t0": 2.2, "t1": 3.8}]


def fake_caption_judge(bad_on):
    calls = []

    def judge(caption, images):
        calls.append((caption, len(images)))
        return [{"status": "contradicts" if (caption, k) in bad_on else ("no_claim" if caption == "Yep." else "matches"),
                 "observed": "a red wall", "fix": "Show the food truck"} for k in range(len(images))]
    return judge, calls


def test_caption_contradicted_on_two_frames_fails(clips):
    sp = sg.spec.make(captions=CAPS)
    judge, calls = fake_caption_judge({("It's a truck.", 0), ("It's a truck.", 2)})
    r = sg.captions.check(clips["same"], sp, judge)
    assert calls == [("It's a truck.", 3), ("Yep.", 3)] and r["status"] == "fail"
    i = r["issues"][0]
    assert i["kind"] == "caption_mismatch" and "2 of 3 frames" in i["message"] and i["t0"] == 0.2
    assert [c["verdict"] for c in r["metrics"]["captions"]] == ["contradicts", "no_claim"]
    assert sg.gate.decide(sp, [r])["hints"] == ["Show the food truck"]


def test_one_bad_frame_is_not_enough(clips):
    judge, _ = fake_caption_judge({("It's a truck.", 1)})
    assert sg.captions.check(clips["same"], sg.spec.make(captions=CAPS), judge)["status"] == "pass"


def test_captions_skip_without_captions_and_never_pass_unscored(clips, monkeypatch):
    assert sg.captions.check(clips["same"], sg.spec.make())["status"] == "skip"

    def broken(caption, images):
        raise RuntimeError("judge unreachable")
    r = sg.captions.check(clips["same"], sg.spec.make(captions=CAPS), broken)
    assert r["status"] == "skip" and r["metrics"]["scored"] is False and "not scored" in r["note"]
    monkeypatch.setenv("SHOTGUARD_CAPTIONS_OFFLINE", "1")
    off = sg.captions.check(clips["same"], sg.spec.make(captions=CAPS))
    assert off["status"] == "skip" and off["metrics"]["captions"][0]["text"] == "It's a truck."


class _Fake(BaseHTTPRequestHandler):
    seen = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Fake.seen.append((self.path, self.headers.get("X-Priority"), body))
        user = body["messages"][1]["content"]
        n = sum(1 for c in user if c["type"] == "image_url")
        status = "contradicts" if "truck" in user[-1]["text"] else "no_claim"
        content = json.dumps({"frames": [{"id": k + 1, "observed": "a red wall", "status": status, "fix": "Show the truck"} for k in range(n)]})
        out = json.dumps({"choices": [{"message": {"content": "```json\n" + content + "\n```"}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


def test_captions_through_a_fake_openai_endpoint(clips, monkeypatch):
    srv = HTTPServer(("127.0.0.1", 0), _Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        monkeypatch.setenv("SHOTGUARD_JUDGE_URL", f"http://127.0.0.1:{srv.server_port}/v1")
        monkeypatch.setenv("SHOTGUARD_JUDGE_MODEL", "fake-vl")
        monkeypatch.delenv("SHOTGUARD_CAPTIONS_OFFLINE", raising=False)
        r = sg.captions.check(clips["same"], sg.spec.make(captions=CAPS))
    finally:
        srv.shutdown()
    assert r["status"] == "fail" and r["metrics"]["judge"] == "fake-vl"
    path, prio, body = _Fake.seen[0]
    assert path == "/v1/chat/completions" and prio == "batch" and body["model"] == "fake-vl" and body["temperature"] == 0


def test_spec_round_trip_keeps_new_fields(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"motion_reference": {"path": "ref.mp4"}, "static_ok": [[27, 30]],
                             "continuity_groups": [{"name": "truck", "spans": [[1, 2], [3, 4]], "boxes": [None, [0, 0, 0.5, 1]]}],
                             "captions": CAPS}))
    sp = sg.spec.load(str(p))
    assert sp["motion_reference"]["path"] == os.path.join(str(tmp_path), "ref.mp4") and sp["static_ok"] == [[27.0, 30.0]]
    assert sp["continuity_groups"][0]["boxes"][1] == [0.0, 0.0, 0.5, 1.0] and sp["captions"][1]["text"] == "Yep."
    assert sg.spec.public(sp)["motion_reference"]["path"] == "ref.mp4"


@pytest.mark.slow
def test_dino_embedder_loads_offline_if_cached(clips):
    try:
        sg.places.dino()
    except Exception as e:   # not cached on this machine
        pytest.skip(f"DINOv2-small not available offline: {type(e).__name__}")
    sp = sg.spec.make(continuity_groups=[{"name": "wall", "spans": [[0.2, 1.8], [2.2, 3.8]]}])
    r = sg.places.check(clips["same"], sp)
    assert r["metrics"]["embedder"] == "dinov2-small" and r["metrics"]["groups"][0]["pairs"][0]["similarity"] > 0.9


def test_caption_judge_retries_broken_json_once(clips, monkeypatch):
    answers = iter(['{"frames": [{"id": 1, "observed": two hands, "status": "matches"}]}',
                    '{"frames": [{"id": 1, "observed": "a red wall", "status": "contradicts", "fix": "Show the truck"},'
                    ' {"id": 2, "observed": "a red wall", "status": "contradicts"}, {"id": 3, "observed": "a red wall", "status": "matches"}]}'])
    monkeypatch.setattr(sg.continuity, "chat", lambda *a, **k: next(answers))
    monkeypatch.delenv("SHOTGUARD_CAPTIONS_OFFLINE", raising=False)
    r = sg.captions.check(clips["same"], sg.spec.make(captions=CAPS[:1]))
    assert r["status"] == "fail" and r["issues"][0]["fix"] == "Show the truck"


def test_reference_must_be_a_local_video_file(tmp_path):
    for bad in ("http://example.invalid/ad.mp4", "rtsp://cam/stream", "file:///etc/passwd", "/etc/passwd"):
        with pytest.raises(ValueError):
            sg.spec.make(motion_reference={"path": bad})
    with pytest.raises(ValueError):
        sg.spec.make(motion_reference={"photo_share": 0.1})
    with pytest.raises(FileNotFoundError):
        sg.stillness.reference_profile({"path": str(tmp_path / "missing.mp4")}, sg.spec.make()["thresholds"])
