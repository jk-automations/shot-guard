"""Speaker: is the right person saying the line? For every script line the dialogue check found, measure how much each
named character's mouth moves while the line is heard, and compare the busiest mouth with the script's speaker.
Mouth crops are cut from YuNet's mouth-corner landmarks, so head movement is mostly taken out. Names come from the
same SFace references as the identity check. CPU, no new model. A line is only flagged when the scripted speaker is
on screen, someone else's mouth clearly moves more, and the line is long enough to measure."""
import os

import cv2
import numpy as np

from . import dialogue
from . import faces as fc
from .video import is_image, read_frames


def segments(path, spec):
    """Whisper segments, cached on the spec so the dialogue and speaker checks transcribe once."""
    key = ("_segments", path, os.path.getmtime(path))
    cache = spec.setdefault("_cache", {})
    if key not in cache:
        cache[key] = dialogue.transcribe(path, model=spec["thresholds"].get("whisper_model"))
    return cache[key]


def _crop(gray, cx, cy, w, h):
    x0, y0, x1, y1 = int(cx - w), int(cy - h), int(cx + w), int(cy + h)
    if x0 < 0 or y0 < 0 or x1 > gray.shape[1] or y1 > gray.shape[0] or x1 - x0 < 8 or y1 - y0 < 6:
        return None
    c = cv2.resize(gray[y0:y1, x0:x1], (48, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    return (c - c.mean()) / (c.std() + 1e-3)


def _mouth(gray, lm):
    """Grey mouth crop from YuNet's mouth-corner landmarks, scaled by the eye distance (so it follows the head)."""
    eye = np.hypot(lm[0][0] - lm[1][0], lm[0][1] - lm[1][1]) + 1e-6
    (rx, ry), (lx, ly) = lm[3], lm[4]
    return _crop(gray, (rx + lx) / 2, (ry + ly) / 2 + 0.15 * eye, 0.9 * eye, 0.45 * eye)


def mouth_motion(path, spec, step=1):
    """Named face tracks with mouth motion per sampled frame: {name: [(t, motion, cx, box)]}."""
    key = ("_mouths", path, os.path.getmtime(path))
    cache = spec.setdefault("_cache", {})
    if key in cache:
        return cache[key]
    th = spec["thresholds"]
    F = read_frames(path, every=step / 24.0, max_side=960)
    frames = []
    for t, img in zip(F.times, F.images):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        fs = fc.detect(img)
        for f in fs:
            f["mouth"] = _mouth(gray, f["lm"])
        frames.append({"t": t, "faces": fs, "scale": F.width / img.shape[1]})
    from . import characters
    cuts = characters.people(path, spec)["cuts"]
    gal = {c["name"]: fc.gallery(c.get("refs", [])) for c in spec.get("characters", [])}
    tracks = fc.label_tracks(fc.build_tracks(frames, cuts), gal, min_match=th["id_min_match"])
    out = {}
    for tr in tracks:
        if not tr.get("name"):
            continue
        pts = tr["pts"]
        for a, b in zip(pts, pts[1:]):
            if a["mouth"] is None or b["mouth"] is None or b["t"] - a["t"] > 2.5 * step / 24.0:
                continue
            out.setdefault(tr["name"], []).append((b["t"], float(np.mean(np.abs(b["mouth"] - a["mouth"]))), b["cx"],
                                                   tuple(v * b["scale"] for v in b["box"])))
    for n in out:
        out[n].sort()
    cache[key] = out
    return out


def line_spans(script, segs, th):
    """(line, t0, t1) for every script line the transcript contains, from the dialogue check's word alignment."""
    a = dialogue.analyze(script, segs, line_accuracy=th["line_accuracy"])
    heard = dialogue._heard(segs)
    out = []
    for L in a["lines"]:
        if not L["found"] or L["end"] <= L["start"]:
            continue
        words = heard[L["start"]:L["end"]]
        # word times are spread evenly over a whisper segment, so pad by a word and stop at the segment edges
        t0 = max(segs[words[0]["seg"]]["t0"], words[0]["t"] - 0.3)
        t1 = min(segs[words[-1]["seg"]]["t1"], words[-1]["t"] + 0.3)
        out.append((L, max(0.0, t0), t1))
    return out


def check(path, spec):
    th = spec["thresholds"]
    script = spec.get("script") or []
    names = {c["name"] for c in spec.get("characters", []) if c.get("refs")}
    if is_image(path) or not script or len(names) < 2:
        return {"check": "speaker", "status": "skip", "issues": [], "metrics": {},
                "note": "needs a video, a script and at least two characters with reference images"}
    if not any(L.get("speaker") in names for L in script):
        return {"check": "speaker", "status": "skip", "issues": [], "metrics": {}, "note": "no scripted speaker is a character with references"}
    segs = segments(path, spec)
    if not segs:
        return {"check": "speaker", "status": "skip", "issues": [], "metrics": {}, "note": "nothing heard (the dialogue check reports it)"}
    M = mouth_motion(path, spec)
    issues, lines = [], []
    for L, t0, t1 in line_spans(script, segs, th):
        want = L["speaker"]
        if want not in names or t1 - t0 < th["speaker_min_s"]:
            continue
        score, seen, boxes = {}, {}, {}
        for n, pts in M.items():
            v = [m for t, m, cx, box in pts if t0 <= t <= t1]
            if v:
                score[n] = float(np.mean(v))
                seen[n] = len(v) / max(1.0, (t1 - t0) * 24.0)
                boxes[n] = min(((abs(t - (t0 + t1) / 2), box, t) for t, m, cx, box in pts if t0 <= t <= t1))
        row = {"line": L["index"], "speaker": want, "t0": round(t0, 2), "t1": round(t1, 2),
               "mouth": {n: round(s, 3) for n, s in score.items()}, "on_screen": {n: round(s, 2) for n, s in seen.items()}}
        lines.append(row)
        if seen.get(want, 0.0) < th["speaker_min_seen"]:
            row["verdict"] = "speaker not on screen"
            continue
        top = max((n for n in score if seen[n] >= th["speaker_min_seen"]), key=lambda n: score[n])
        row["verdict"] = "ok" if top == want else "unsure"
        # a head turn also changes the mouth crop: on real takes it gave up to 1.4x, real wrong speakers 1.8x+ (LESSONS)
        ratio = score[top] / (score[want] + 1e-6) if top != want else 0.0
        row["ratio"] = round(ratio, 2)
        if ratio >= th["speaker_ratio"]:
            row["verdict"] = "wrong"
            _, box, bt = boxes[top]
            issues.append({"kind": "wrong_speaker", "severity": "major" if ratio >= th["speaker_ratio_major"] else "minor",
                           "t0": t0, "t1": t1, "frame_t": bt, "box": box, "character": want, "speaking": top, "line": L["index"],
                           "message": f"line {L['index']} belongs to {want}, but {top}'s mouth moves while it is heard "
                                      f"({ratio:.1f}x as much as {want}'s)"})
    return {"check": "speaker", "status": "fail" if any(i["severity"] == "major" for i in issues) else "pass",
            "issues": issues, "metrics": {"lines": lines}}
