"""Glitch: playback faults a viewer sees at once but no content check looks for. A frozen picture inside a moving
shot, stutter (the same frame shown twice again and again, from bad frame interpolation or a dropped-frame
encode), and flicker (brightness pumping up and down from frame to frame). Plain numpy on small grey frames, no model.
Measured on 24 real takes (docs/LESSONS.md): the longest near-still run was 0.46 s in a calm shot, and no take had a
stutter or flicker event, so a freeze of 1 s or more fails and the rest are warnings."""
import cv2
import numpy as np

from .video import is_image, read_frames


def signals(path, max_side=160):
    """Per-frame change (mean abs grey difference to the previous frame) and the brightness series."""
    F = read_frames(path, every=0, max_side=max_side)
    g = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32) for im in F.images]
    d = np.array([np.mean(np.abs(b - a)) for a, b in zip(g, g[1:])])
    lum = np.array([x.mean() for x in g])
    return F, d, lum


def _runs(mask):
    out, start = [], None
    for k, m in enumerate(list(mask) + [False]):
        if m and start is None:
            start = k
        elif not m and start is not None:
            out.append((start, k))
            start = None
    return out


def check(path, spec):
    if is_image(path):
        return {"check": "glitch", "status": "skip", "issues": [], "metrics": {}, "note": "a still has no playback"}
    th = spec["thresholds"]
    F, d, lum = signals(path)
    if len(d) < 8:
        return {"check": "glitch", "status": "skip", "issues": [], "metrics": {}, "note": "too few frames"}
    fps = F.fps or 24.0
    med = float(np.median(d))
    issues = []
    # frozen: frames that barely change at all, compared with how much this shot moves (a calm shot is not a freeze)
    if med >= th["glitch_still_shot"]:
        frozen = d < min(th["glitch_frozen_diff"], 0.25 * med)
        for a, b in _runs(frozen):
            secs = (b - a) / fps
            if secs >= th["glitch_freeze_s"]:
                t0, t1 = (a + 1) / fps, (b + 1) / fps
                issues.append({"kind": "frozen", "severity": "major" if secs >= th["glitch_freeze_major_s"] else "minor",
                               "t0": t0, "t1": t1, "frame_t": t0,
                               "message": f"the picture freezes from {t0:.1f} s to {t1:.1f} s ({secs:.1f} s) in a moving shot"})
    # stutter: one repeated frame between two normal steps, again and again
    rep = [i for i in range(1, len(d) - 1)
           if min(d[i - 1], d[i + 1]) > th["glitch_stutter_motion"] and d[i] < 0.25 * min(d[i - 1], d[i + 1])]
    if len(rep) >= th["glitch_stutter_events"]:
        t0, t1 = (rep[0] + 1) / fps, (rep[-1] + 1) / fps
        issues.append({"kind": "stutter", "severity": "minor", "t0": t0, "t1": t1, "frame_t": t0,
                       "message": f"the motion stutters: {len(rep)} repeated frames between {t0:.1f} s and {t1:.1f} s"})
    # flicker: brightness above and below its local median on alternate frames
    base = np.array([np.median(lum[max(0, i - 3):i + 4]) for i in range(len(lum))])
    hp = lum - base
    big = np.abs(hp) > th["glitch_flicker_luma"]
    flips = [i for i in range(1, len(hp)) if big[i] and big[i - 1] and hp[i] * hp[i - 1] < 0]
    if len(flips) >= th["glitch_flicker_events"]:
        t0, t1 = flips[0] / fps, flips[-1] / fps
        issues.append({"kind": "flicker", "severity": "minor", "t0": t0, "t1": t1, "frame_t": t0,
                       "message": f"the brightness flickers from frame to frame between {t0:.1f} s and {t1:.1f} s ({len(flips)} flips)"})
    return {"check": "glitch", "status": "fail" if any(i["severity"] == "major" for i in issues) else "pass", "issues": issues,
            "metrics": {"motion_median": round(med, 3), "repeated_frames": len(rep), "flicker_flips": len(flips),
                        "brightness_jitter": round(float(np.abs(hp).max()), 2)}}
