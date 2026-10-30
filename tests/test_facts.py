"""Facts check (Continuity Lock's vision judge inside Shot Guard). Unit tests use a fake judge; the live test talks to
the configured endpoint (fleet router by default) and skips when it is not reachable."""
import json
import urllib.request

import pytest

from shotguard_core import continuity as cl
from shotguard_core import facts, gate, spec
from conftest import make_clip


def fake_judge(broken_on):
    """Violates every fact on the frames whose index is in broken_on."""
    calls = []

    def judge(img, bible):
        k = len(calls)
        calls.append(k)
        bad = k in broken_on
        iss = [{"fact": f.text, "observed": "it is blue", "severity": f.severity, "fix": "Make it red"} for f in bible.facts] if bad else []
        return {"pass": not bad, "score": 0.0 if bad else 1.0, "issues": iss, "facts": [], "seconds": 0.1}
    return judge, calls


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    return str(make_clip(tmp_path_factory.mktemp("f") / "t.mp4", 2.0, pattern="testsrc2"))


def test_no_facts_skips_without_calling_the_judge(clip):
    judge, calls = fake_judge(set())
    assert facts.check(clip, spec.make(), judge)["status"] == "skip" and calls == []


def test_fact_broken_on_two_frames_fails_and_hints(clip):
    sp = spec.make(facts="the car is glossy red\n[minor] the sky is clear")
    judge, calls = fake_judge({1, 2})
    r = facts.check(clip, sp, judge)
    assert len(calls) == 4 and r["status"] == "fail"
    major = [i for i in r["issues"] if i["severity"] == "major"]
    assert major[0]["fact"] == "the car is glossy red" and "2 of 4 frames" in major[0]["message"]
    g = gate.decide(sp, [r])
    assert g["verdict"] == "RETRY" and g["hints"][0] == "keep this exactly as in the reference: the car is glossy red"


def test_single_frame_glitch_is_not_enough(clip):
    judge, _ = fake_judge({3})
    assert facts.check(clip, spec.make(facts=["the car is glossy red"]), judge)["status"] == "pass"


def test_vendored_judge_parses_messy_model_output():
    out = cl.parse_json_loose('<think>hmm</think>```json\n{"facts":[{"id":1,"status":"violated","observed":"blue",},]}\n```')
    assert out["facts"][0]["status"] == "violated"


def _judge_up():
    try:
        urllib.request.urlopen(cl.llm_defaults()["url"].replace("/v1", "") + "/health", timeout=3)
        return True
    except Exception:
        return False


@pytest.mark.slow
def test_live_judge_on_the_garbled_sign(fixtures):
    if not _judge_up():
        pytest.skip("vision judge endpoint not reachable")
    bad, _, _ = fixtures["garbled_sign"]
    good, _, _ = fixtures["final_sign"]
    sp = spec.make(facts=["the big sign above the shop reads exactly MEGABUN"])
    assert facts.check(bad, sp)["status"] == "fail"
    assert facts.check(good, sp)["status"] == "pass"
