"""Hidden cuts: a scene cut or a fast angle jump inside what should be one continuous take.
Per frame: colour-histogram distance x mean pixel change (frame to frame, and across a short window for whip
or morph transitions that spread over several frames). Calibrated on real takes: docs/LESSONS.md."""
import cv2
import numpy as np

from .video import gray_small, hist, is_image, read_frames


def scores(path, window=6):
    F = read_frames(path, every=0)
    hs = [hist(i) for i in F.images]
    gs = [gray_small(i) for i in F.images]

    def score(a, b):
        return cv2.compareHist(hs[a], hs[b], cv2.HISTCMP_BHATTACHARYYA) * float(np.mean(np.abs(gs[b] - gs[a])))
    hard = np.array([score(i - 1, i) for i in range(1, len(hs))])
    win = np.array([score(i, i + window) for i in range(len(hs) - window)]) if len(hs) > window else np.zeros(0)
    return F, hard, win


def _spans(mask, times, pad=0.0):
    out, start = [], None
    for k, m in enumerate(list(mask) + [False]):
        if m and start is None:
            start = k
        elif not m and start is not None:
            out.append((times[start] - pad, times[k - 1] + pad, start, k - 1)); start = None
    return out


def check(path, spec):
    if is_image(path):
        return {"check": "cuts", "status": "skip", "issues": [], "metrics": {}, "note": "a still has no cuts"}
    th = spec["thresholds"]
    F, hard, win = scores(path, th["cut_window_frames"])
    t = F.times
    issues = []
    for t0, t1, a, b in _spans(hard > th["cut_hard"], t[1:]):
        k = a + int(np.argmax(hard[a:b + 1]))
        issues.append({"kind": "hard_cut", "severity": "major", "t0": t0, "t1": t1, "frame_t": t[k + 1],
                       "score": round(float(hard[k]), 2), "message": f"hard cut at {t[k + 1]:.2f} s (score {hard[k]:.2f})"})
    for t0, t1, a, b in _spans(win > th["cut_window"], t[:len(win)]):
        t1 = t1 + th["cut_window_frames"] / F.fps
        if any(i["t0"] <= t1 and t0 <= i["t1"] + th["cut_window_frames"] / F.fps for i in issues):
            continue   # already reported as a hard cut
        k = a + int(np.argmax(win[a:b + 1]))
        # a warning, not a fail: in the blind test (17 unseen takes) every real cut was caught by the hard-cut score,
        # and this window score only added false alarms (fast camera moves, people walking past the lens, flour)
        issues.append({"kind": "angle_jump", "severity": "minor", "t0": t0, "t1": t1, "frame_t": t[k] + 0.5 * th["cut_window_frames"] / F.fps,
                       "score": round(float(win[k]), 2),
                       "message": f"possible angle jump between {t0:.1f} s and {t1:.1f} s, check by eye (score {win[k]:.2f})"})
    return {"check": "cuts", "status": "fail" if any(i["severity"] == "major" for i in issues) else "pass", "issues": issues,
            "metrics": {"max_hard": round(float(hard.max()), 3) if len(hard) else 0.0,
                        "max_window": round(float(win.max()), 3) if len(win) else 0.0, "fps": F.fps, "frames": len(F.images)}}
