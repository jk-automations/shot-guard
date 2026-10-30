"""Report rendering (contact sheet + JSON receipt) and the ComfyUI node wrappers."""
import json

import numpy as np
import pytest
from PIL import Image

import shotguard_core as sg
from conftest import make_clip


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    return str(make_clip(tmp_path_factory.mktemp("c") / "take.mp4", 3.0, pattern="testsrc2"))


def fake_result(path, issues):
    sp = sg.spec.make(name="demo take", allowed_text=["MEGABUN"])
    return sg.gate.decide(sp, [{"check": "cuts", "status": "fail" if issues else "pass", "issues": issues, "metrics": {}},
                               {"check": "dialogue", "status": "skip", "issues": [], "metrics": {}}], path)


def test_report_sheet_and_receipt(clip, tmp_path):
    issues = [{"kind": "hard_cut", "severity": "major", "t0": 1.5, "t1": 1.5, "frame_t": 1.5, "message": "hard cut at 1.50 s"},
              {"kind": "angle_jump", "severity": "major", "t0": 2.0, "t1": 2.4, "frame_t": 2.2, "box": (10, 10, 50, 50), "message": "jump"}]
    img, png, js = sg.report.render(fake_result(clip, issues), str(tmp_path), "r1", blur=True)
    im = Image.open(png)
    assert im.size[0] == 1600 and im.size[1] > 900
    a = np.asarray(im.convert("RGB"))
    assert (np.abs(a[84, 1552].astype(int) - (176, 128, 62)) < 8).all()   # RETRY verdict chip is drawn
    rec = json.load(open(js))
    assert rec["verdict"] == "RETRY" and rec["video"] == "take.mp4" and len(rec["issues"]) == 2
    assert rec["hints"] == ["one continuous shot: no cuts and no change of camera angle"]


def test_report_pass_and_still(tmp_path):
    p = tmp_path / "s.png"
    Image.new("RGB", (400, 700), (180, 160, 140)).save(p)
    img, png, js = sg.report.render(fake_result(str(p), []), str(tmp_path), "r2", blur=False)
    assert json.load(open(js))["verdict"] == "PASS" and Image.open(png).size[0] == 1600


def test_nodes_wire_together(comfy_path, clip, tmp_path, monkeypatch):
    import torch
    import shotguard_nodes as N
    def out_dir(*parts):
        d = tmp_path.joinpath(*parts)
        d.mkdir(parents=True, exist_ok=True)
        return str(d)
    monkeypatch.setattr(N, "_out_dir", out_dir)
    ref = torch.rand(1, 64, 64, 3)
    sp, sp_json = N.ShotGuardSpec().run("demo", "A: hello there\nB: hi", "MEGABUN", 2, 0.0, 0.0, 0,
                                        "Ana", "left", "", "any", "", "any", reference_1=ref)
    assert json.loads(sp_json)["characters"][0] == {"name": "Ana", "refs": ["<image>"], "side": "left", "stay_in_frame": True}
    cuts, summary = N.ShotGuardCuts().run(sp, video_path=clip)
    assert cuts["status"] == "pass" and summary.startswith("cuts: PASS")
    res, verdict, hint, summ, passed = N.ShotGuardGate().run(sp, cuts=cuts)
    assert verdict == "PASS" and passed and hint == ""
    out = N.ShotGuardReport().run(res, "shotguard/report", False)
    assert out["result"][0].shape[0] == 1 and out["result"][0].shape[3] == 3 and json.loads(out["result"][1])["verdict"] == "PASS"
    st = N.ShotGuardRetry().run(res, True, "", "", "http://127.0.0.1:1", "", prompt={}, unique_id="9")
    assert st["result"][0].startswith("PASS: best take is attempt 0")


def test_retry_is_changed_is_json_safe(comfy_path):
    import json, shotguard_nodes as N
    v = N.ShotGuardRetry.IS_CHANGED()
    assert json.loads(json.dumps(v)) == v and v != N.ShotGuardRetry.IS_CHANGED()
