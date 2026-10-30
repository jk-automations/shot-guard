"""Clickable HTML report, share card, live score in the annotated video (synthetic clips, no fixtures needed)."""
import re
import subprocess

import numpy as np
import cv2

import shotguard_core as sg
from conftest import FFMPEG, make_clip


def _cut_take(tmp_path):
    a, b = make_clip(tmp_path / "a.mp4", 2.0), make_clip(tmp_path / "b.mp4", 2.0, pattern="smptebars")
    out = tmp_path / "cut.mp4"
    subprocess.run([FFMPEG, "-nostdin", "-v", "error", "-y", "-i", str(a), "-i", str(b), "-filter_complex", "[0:v][1:v]concat=n=2:v=1[v]",
                    "-map", "[v]", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)], check=True)
    return str(out)


def test_html_report_has_a_seek_button_per_finding_and_no_network(tmp_path):
    path = _cut_take(tmp_path)
    r = sg.run(path, sg.spec.make(), ["cuts", "glitch", "attention"])
    page = open(sg.html_report.render(r, str(tmp_path / "rep"), "take")).read()
    cut = next(i for i in r["issues"] if i["kind"] == "hard_cut")
    assert f'data-t="{cut["frame_t"]:.3f}"' in page and 'src="take.mp4"' in page
    assert (tmp_path / "rep" / "take.mp4").exists() and (tmp_path / "rep" / "take.json").exists()
    assert not re.search(r'(src|href)="https?://', page)   # self-contained: nothing loads from the network


def test_share_card_size_and_dailies_index(tmp_path):
    path = _cut_take(tmp_path)
    r = sg.run(path, sg.spec.make(), ["cuts"])
    img = sg.card.render(r, str(tmp_path / "card.png"), blur=False)
    assert img.size == (1080, 1350)
    d = sg.batch.run_folder(str(tmp_path), sg.spec.make(), out_dir=str(tmp_path / "out"), which=["cuts"], log=lambda s: None)
    idx = open(sg.html_report.index(d, str(tmp_path / "html"))).read()
    assert all(f'href="{x["take"].rsplit(".", 1)[0]}.html"' in idx for x in d["rows"])


def test_live_score_dial_drops_when_the_problem_plays(tmp_path):
    path = _cut_take(tmp_path)
    r = sg.run(path, sg.spec.make(), ["cuts"])
    out = sg.annotate.render(r, str(tmp_path / "ann.mp4"), blur=False)
    cap = cv2.VideoCapture(out)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    u = w / 720.0
    frames = []
    while True:
        ok, im = cap.read()
        if not ok:
            break
        frames.append(im)
    # the dial's arc colour, top right: green before the cut, red after it
    y, x = int((66 - 46) * u), int(w - 66 * u)
    before, after = frames[10][y, x].astype(int), frames[60][y, x].astype(int)
    assert before[1] > before[2] and after[2] > after[1]   # BGR: green wins before, red wins after


def test_retry_forwards_partner_credentials_without_logging_them(monkeypatch, tmp_path):
    sent = {}

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b'{"prompt_id": "p1"}'

    def fake_urlopen(req, timeout=30):
        sent["body"] = req.data.decode()
        return Resp()

    monkeypatch.setattr(sg.retry.urllib.request, "urlopen", fake_urlopen)
    graph = {"1": {"class_type": "KSampler", "inputs": {"seed": 1}}, "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "a cat"}}}
    sp = sg.spec.make(max_retries=2)
    r = sg.gate.decide(sp, [{"check": "cuts", "status": "fail", "issues": [{"kind": "hard_cut", "severity": "major", "message": "cut"}]}])
    out = sg.retry.step(r, graph=graph, run_id="t", ledger_root=str(tmp_path),
                        post=lambda g: sg.retry.post_prompt("http://x", g, extra_data={"api_key_comfy_org": "SECRET-123"}))
    assert out["requeued"] and '"api_key_comfy_org": "SECRET-123"' in sent["body"]
    assert "SECRET-123" not in open(out["ledger"]).read() and "SECRET-123" not in out["status"]


def test_cast_member_without_reference_is_not_a_face_warning(tmp_path):
    sp = sg.spec.make(characters=[{"name": "Crowd", "refs": []}])
    r = sg.characters.check_identity(str(make_clip(tmp_path / "c.mp4", 1.0)), sp)
    assert r["issues"] == []
