"""Each check flags its known-bad real take and passes the known-good ones. These are the failures a top generator
shipped in our own pilot film; only a human caught them at the time."""
import pytest

import shotguard_core as sg

pytestmark = [pytest.mark.fixtures, pytest.mark.slow]
BAD = {"side_swap": "blocking", "scrambled_dialogue": "dialogue", "garbled_sign": "text", "hidden_cut": "cuts", "end_drift": "identity"}
GOOD = ("good_W1", "good_W4")


@pytest.mark.parametrize("name,check", sorted(BAD.items()))
def test_check_flags_its_bad_take(fixtures, name, check):
    path, sp, must = fixtures[name]
    assert must == check
    r = sg.CHECKS[check](path, sp)
    if check == "identity":   # drift is a warning unless the drop is large (blind test v1: head turns look like drift)
        assert any(i["kind"] == "identity_drift" for i in r["issues"]), r
        return
    assert r["status"] == "fail", r
    assert any(i["severity"] == "major" for i in r["issues"])


@pytest.mark.parametrize("check", list(sg.CHECKS))
@pytest.mark.parametrize("name", GOOD)
def test_check_passes_good_take(fixtures, name, check):
    path, sp, _ = fixtures[name]
    r = sg.CHECKS[check](path, sp)
    assert r["status"] in ("pass", "skip"), [i["message"] for i in r["issues"]]


def test_final_sign_still_passes_text(fixtures):
    path, sp, _ = fixtures["final_sign"]
    assert sg.CHECKS["text"](path, sp)["status"] == "pass"


def test_side_swap_found_where_it_happens(fixtures):
    path, sp, _ = fixtures["side_swap"]
    swaps = [i for i in sg.CHECKS["blocking"](path, sp)["issues"] if i["kind"] == "side_swap"]
    assert swaps and swaps[0]["character"] == "Rosa" and 2.0 <= swaps[0]["t0"] <= 3.0


def test_scrambled_dialogue_order_and_merge(fixtures):
    path, sp, _ = fixtures["scrambled_dialogue"]
    kinds = {i["kind"] for i in sg.CHECKS["dialogue"](path, sp)["issues"]}
    assert {"out_of_order", "merged_lines"} <= kinds


def test_hidden_cut_in_the_known_window(fixtures):
    path, sp, _ = fixtures["hidden_cut"]
    iss = sg.CHECKS["cuts"](path, sp)["issues"]
    assert any(i["t0"] <= 6.4 and i["t1"] >= 5.4 for i in iss), iss


def test_end_drift_found_after_the_cut_on_the_right_face(fixtures):
    path, sp, _ = fixtures["end_drift"]
    r = sg.CHECKS["identity"](path, sp)
    who = sp["characters"][0]["name"]          # the name lives in the private spec, not in the repo
    drift = [i for i in r["issues"] if i["kind"] == "identity_drift"]
    assert drift and all(i["character"] == who for i in drift) and drift[0]["t0"] >= 4.5
    box = drift[0]["box"]                      # the drifting character stands on the left after the cut
    assert (box[0] + box[2] / 2) / 720 < 0.4, box
    c = r["metrics"]["curves"][who]
    assert c["min"] < c["baseline"] - 0.12      # (in the last second this face is cut off by the frame edge, so it is not scored)
    others = [n for n in r["metrics"]["curves"] if n != who]
    assert others and not any(i["character"] in others for i in r["issues"])   # the other face does not drift


def test_garbled_sign_names_the_right_word(fixtures):
    path, sp, _ = fixtures["garbled_sign"]
    iss = sg.CHECKS["text"](path, sp)["issues"]
    assert iss[0]["kind"] == "garbled" and iss[0]["expected"] == "MEGABUN"
