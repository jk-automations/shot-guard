"""Sequence check on real takes: the side-swap take split at its own cut is two shots where the baker jumps sides;
a good take split in half keeps her side."""
import subprocess

import pytest

import shotguard_core as sg
from conftest import FFMPEG

pytestmark = [pytest.mark.fixtures, pytest.mark.slow]


def split(src, at, d):
    a, b = d / "a.mp4", d / "b.mp4"
    subprocess.run([FFMPEG, "-nostdin", "-v", "error", "-y", "-i", src, "-t", str(at), "-an", str(a)], check=True)
    subprocess.run([FFMPEG, "-nostdin", "-v", "error", "-y", "-ss", str(at + 0.1), "-i", src, "-an", str(b)], check=True)
    return [str(a), str(b)]


def test_side_flip_across_a_cut_is_flagged(fixtures, tmp_path):
    path, sp, _ = fixtures["side_swap"]
    r = sg.sequence.check(split(path, 2.35, tmp_path), sp)
    assert r["status"] == "fail" and r["issues"][0]["expected"] == "left", r["metrics"]
    assert sg.gate.decide(sp, [r])["hints"] == ["keep Rosa on the left of the frame, as in the previous shot"]


def test_same_side_across_a_cut_passes(fixtures, tmp_path):
    path, sp, _ = fixtures["good_W1"]
    r = sg.sequence.check(split(path, 4.0, tmp_path), sp)
    assert r["status"] == "pass" and r["metrics"]["pairs"], r["metrics"]
