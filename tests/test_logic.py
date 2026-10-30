"""Fast tests with no models: dialogue matching on canned transcripts, the gate and its hints, text judging, spec parsing."""
import pytest

import shotguard_core as sg
from shotguard_core import dialogue, gate, spec, text

SCRIPT = [{"speaker": "A", "text": "Hello there, I'm Sam."}, {"speaker": "B", "text": "Why are you here?"},
          {"speaker": "A", "text": "We have forty ideas."}, {"speaker": "B", "text": "Give me three days."}]


def seg(t0, t1, txt):
    return {"t0": t0, "t1": t1, "text": txt}


def kinds(r):
    return [i["kind"] for i in r["issues"]]


def test_dialogue_clean_take_passes_and_numbers_match():
    r = dialogue.analyze(SCRIPT, [seg(0, 1, "Hello there, I'm Sam."), seg(1, 2, "Why are you here?"),
                                  seg(2, 3, "We have 40 ideas."), seg(3, 4, "Give me three days. (laughs)")])
    assert r["issues"] == [] and all(L["accuracy"] == 1.0 for L in r["lines"])


def test_dialogue_out_of_order_missing_merged_extra():
    r = dialogue.analyze(SCRIPT, [seg(0, 1, "Hello there, I'm Sam."), seg(1, 2, "Give me three days we have forty ideas."),
                                  seg(2, 3, "pizza is great tonight")])
    k = kinds(r)
    assert "missing_line" in k and "out_of_order" in k and "merged_lines" in k and "extra_speech" in k
    assert r["heard_order"] == [1, 4, 3]


def test_dialogue_wrong_words():
    r = dialogue.analyze(SCRIPT[:1], [seg(0, 1, "Hello there Tim")])
    assert kinds(r) == ["wrong_words"]


def test_same_speaker_lines_in_one_breath_are_fine():
    sc = [{"speaker": "A", "text": "One two three."}, {"speaker": "A", "text": "Four five six."}]
    assert dialogue.analyze(sc, [seg(0, 2, "One two three four five six")])["issues"] == []


def result_for(sp, issues):
    return gate.decide(sp, [{"check": "blocking", "status": "fail" if issues else "pass", "issues": issues, "metrics": {}}])


SWAP = {"kind": "side_swap", "severity": "major", "t0": 2.5, "t1": 7.8, "character": "Rosa", "expected": "left", "message": "m"}


def test_gate_pass_retry_fail_by_retries_and_by_cap():
    sp = spec.make(max_retries=2, credit_cap=3.0, cost_per_take=1.0)
    assert result_for(sp, [])["verdict"] == "PASS"
    assert result_for(sp, [SWAP])["verdict"] == "RETRY"
    assert result_for(dict(sp, attempt=2), [SWAP])["verdict"] == "FAIL"          # retries used up
    capped = result_for(dict(sp, attempt=1, credit_cap=2.5), [SWAP])               # 2 spent, a 3rd would pass 2.5
    assert capped["verdict"] == "FAIL" and "credit cap" in capped["why"]


def test_minor_issue_does_not_fail_but_lowers_score():
    sp = spec.make()
    r = result_for(sp, [dict(SWAP, severity="minor")])
    assert r["verdict"] == "PASS" and r["score"] < 100


def test_hints_are_plain_english_and_ordered():
    sp = spec.make(script=SCRIPT, allowed_text=["MEGABUN"])
    issues = [{"check": "text", "kind": "garbled", "severity": "major", "expected": "MEGABUN", "t0": 0, "t1": 0, "message": ""},
              {"check": "cuts", "kind": "hard_cut", "severity": "major", "t0": 1, "t1": 1, "message": ""}]
    checks = [{"check": "dialogue", "status": "fail", "metrics": {},
               "issues": [{"kind": "out_of_order", "severity": "major", "t0": 1, "t1": 2, "lines": [4], "message": ""},
                          {"kind": "merged_lines", "severity": "major", "t0": 1, "t1": 2, "lines": [3, 4], "message": ""}]},
              {"check": "blocking", "status": "fail", "issues": [SWAP], "metrics": {}},
              {"check": "text", "status": "fail", "issues": issues[:1], "metrics": {}},
              {"check": "cuts", "status": "fail", "issues": issues[1:], "metrics": {}}]
    r = gate.decide(sp, checks)
    h = r["hints"]
    assert h[0].startswith("speak the lines in this order, one at a time: 1) \"Hello there, I'm Sam.\"")
    assert "leave a clear pause between \"We have forty ideas.\" and \"Give me three days.\"" in h[1]
    assert "keep Rosa on the left of the frame for the whole take; the camera stays on this side" in h[2]
    assert h[3] == "the sign reads exactly \"MEGABUN\", spelled correctly"
    assert h[4] == "one continuous shot: no cuts and no change of camera angle"
    assert r["prompt_hint"].startswith("Shot Guard fix: speak the lines") and r["prompt_hint"].endswith("Keep everything else identical.")


def test_text_judge():
    allowed = ["MEGABUN", "Menu"]
    assert text.judge("MEGABUN", allowed) is None and text.judge("menu", allowed) is None
    assert text.judge("GABUN", allowed) is None                       # partly hidden sign
    assert text.judge("MEGA-BUKEN", allowed) == ("garbled", "MEGABUN")
    assert text.judge("STARBUCKS", allowed)[0] == "not_allowed"
    assert text.tokens("٠٠ 12 ok MEGA-BUKEN") == ["MEGA-BUKEN"]


def test_spec_parse_and_sides():
    s = spec.parse_script("A: Hi there.\n\n# note\nJust words")
    assert s == [{"speaker": "A", "text": "Hi there."}, {"speaker": "", "text": "Just words"}]
    with pytest.raises(ValueError):
        spec.make(characters=[{"name": "x", "side": "up"}])
    sp = spec.make(thresholds={"line_accuracy": 0.5, "bogus": 1})
    assert sp["thresholds"]["line_accuracy"] == 0.5 and "bogus" not in sp["thresholds"]


def test_cuts_on_synthetic_clips(tmp_path):
    from conftest import make_clip
    import subprocess
    from conftest import FFMPEG
    a = make_clip(tmp_path / "a.mp4", 2.0, pattern="testsrc2")
    assert sg.CHECKS["cuts"](str(a), spec.make())["status"] == "pass"
    b = make_clip(tmp_path / "b.mp4", 2.0, pattern="smptebars")
    cut = tmp_path / "cut.mp4"
    subprocess.run([FFMPEG, "-nostdin", "-v", "error", "-y", "-i", str(a), "-i", str(b), "-filter_complex",
                    "[0:v][1:v]concat=n=2:v=1[v]", "-map", "[v]", str(cut)], check=True)
    r = sg.CHECKS["cuts"](str(cut), spec.make())
    assert r["status"] == "fail" and abs(r["issues"][0]["frame_t"] - 2.0) < 0.1


def test_text_one_letter_misread_is_noise_only_when_the_right_spelling_is_also_read(monkeypatch, tmp_path):
    import cv2
    import numpy as np
    from conftest import make_clip
    clip = str(make_clip(tmp_path / "t.mp4", 3.0))
    reads = iter([[("BAKERY", (0, 0, 90, 90))], [("SAKERY", (0, 0, 90, 90))], [("SAKERY", (0, 0, 90, 90))]])
    monkeypatch.setattr(text, "ocr", lambda img: next(reads))
    no_start = {"allow_start_text": False}
    assert text.check(clip, spec.make(allowed_text=["BAKERY"], thresholds=no_start))["status"] == "pass"
    reads = iter([[("SAKERY", (0, 0, 90, 90))]] * 3)
    r = text.check(clip, spec.make(allowed_text=["BAKERY"], thresholds=no_start))
    assert r["status"] == "fail" and r["issues"][0]["expected"] == "BAKERY"


def test_text_already_on_the_first_frame_is_allowed(monkeypatch, tmp_path):
    from conftest import make_clip
    clip = str(make_clip(tmp_path / "s.mp4", 4.0))
    reads = iter([[("FOR LEASE", (0, 0, 90, 90))]] + [[("FOR LEASE", (0, 0, 90, 90)), ("GLORPTEX", (0, 0, 90, 90))]] * 2)
    monkeypatch.setattr(text, "ocr", lambda img: next(reads))
    r = text.check(clip, spec.make(allowed_text=[]))
    assert [i["token"] for i in r["issues"]] == ["GLORPTEX"]      # the sign from the start frame is fine, new text is not


def test_dialogue_v2_numbers_stretched_words_asr_slips_and_identical_lines():
    sc = [{"speaker": "A", "text": "Fifty thousand views."}, {"speaker": "B", "text": "Ooooooh!"},
          {"speaker": "A", "text": "We do this for the croissants."}, {"speaker": "B", "text": "We do this for the croissants."}]
    r = dialogue.analyze(sc, [seg(0, 1, "50,000 views."), seg(1, 2, "Oh!"),
                              seg(2, 4, "We do this for the crisants. We do this for the croissants.")])
    assert r["issues"] == [], r["issues"]


def test_small_background_text_is_ignored_unless_close_up_mode(monkeypatch, tmp_path):
    from conftest import make_clip
    clip = str(make_clip(tmp_path / "b.mp4", 4.0))        # 640 px tall
    board = [("Calava scribble", (10, 400, 60, 415))]      # 15 px tall: 2% of the frame
    monkeypatch.setattr(text, "ocr", lambda img: board)
    th = {"allow_start_text": False}
    assert text.check(clip, spec.make(allowed_text=["MEGABUN"], thresholds=th))["status"] == "pass"
    assert text.check(clip, spec.make(allowed_text=["MEGABUN"], thresholds=dict(th, min_text_height=0)))["status"] == "fail"
