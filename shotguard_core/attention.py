"""Attention: will people keep watching? The open, rule-based answer to a virality predictor's hook / attention /
retention scores. Every quarter second gets an attention value from what usually holds a viewer: something moving,
a face on screen (bigger is stronger), someone speaking, and sound. From that: a hook score for the first seconds, an
attention curve, and dead stretches where nothing holds the eye or ear. Signals and weights are plain rules, not a
trained model of viewers or brains, so attention findings are warnings only and never fail a take."""
import numpy as np

from . import audio, faces as fc, glitch
from .video import has_audio, is_image

STEP = 0.25
WEIGHTS = {"motion": 0.3, "face": 0.25, "speech": 0.3, "sound": 0.15}


def curve(path, spec):
    """Per 0.25 s: the four signals (0..1) and their weighted sum. Reuses cached faces and transcript."""
    F, d, _ = glitch.signals(path)
    fps = F.fps or 24.0
    dur = F.duration or len(d) / fps
    n = max(1, int(dur / STEP))
    t = np.arange(n) * STEP
    motion = np.array([np.mean(d[int(a * fps):max(int(a * fps) + 1, int((a + STEP) * fps))] if len(d) else 0.0) for a in t])
    motion = np.clip(motion / spec["thresholds"]["attention_motion_full"], 0, 1)
    _, frames = fc.scan(path, every=STEP)
    face = np.zeros(n)
    for fr in frames:
        k = min(n - 1, int(fr["t"] / STEP))
        sizes = [f["box"][2] * f["box"][3] for f in fr["faces"]]
        if sizes:   # a face filling a fifth of the frame counts as full strength
            img_area = (F.width / fr["scale"]) * (F.height / fr["scale"])
            face[k] = min(1.0, 0.5 + 2.5 * max(sizes) / img_area)
    speech, sound = np.zeros(n), np.zeros(n)
    if has_audio(path):
        from .speaker import segments
        for s in segments(path, spec) or []:
            speech[int(s["t0"] / STEP):int(np.ceil(s["t1"] / STEP))] = 1.0
        db = audio.levels(audio.samples(path))
        per = int(STEP / 0.05)
        lv = np.array([db[k * per:(k + 1) * per].mean() if len(db[k * per:(k + 1) * per]) else -90 for k in range(n)])
        sound = np.clip((lv + 50) / 35, 0, 1)
    e = WEIGHTS["motion"] * motion + WEIGHTS["face"] * face + WEIGHTS["speech"] * speech + WEIGHTS["sound"] * sound
    return {"t": t, "attention": e, "motion": motion, "face": face, "speech": speech, "sound": sound, "duration": dur}


def check(path, spec):
    if is_image(path):
        return {"check": "attention", "status": "skip", "issues": [], "metrics": {}, "note": "a still has no attention curve"}
    th = spec["thresholds"]
    C = curve(path, spec)
    t, e = C["t"], C["attention"]
    k_hook = max(1, int(th["attention_hook_s"] / STEP))
    hook = float(e[:k_hook].mean())
    issues = []
    if hook < th["attention_hook_min"]:
        issues.append({"kind": "slow_hook", "severity": "minor", "t0": 0.0, "t1": float(t[k_hook - 1] + STEP), "frame_t": 0.5,
                       "message": f"slow start: little happens in the first {th['attention_hook_s']:g} s (hook {100 * hook:.0f}/100)"})
    low = e < th["attention_dead"]
    start = None
    for k, m in enumerate(list(low) + [False]):
        if m and start is None:
            start = k
        elif not m and start is not None:
            a, b = max(float(t[start]), th["attention_hook_s"]), float(t[k - 1] + STEP)   # the hook is judged on its own
            if b - a >= th["attention_dead_s"]:
                issues.append({"kind": "dead_stretch", "severity": "minor", "t0": a, "t1": b, "frame_t": (a + b) / 2,
                               "message": f"attention dips from {a:.1f} s to {b:.1f} s: no speech, little movement"})
            start = None
    return {"check": "attention", "status": "pass", "issues": issues,
            "metrics": {"hook": round(100 * hook), "attention": round(100 * float(e.mean())),
                        "low_share": round(float(low.mean()), 2),
                        "curve": {"t": [round(float(x), 2) for x in t], "attention": [round(float(x), 3) for x in e]}}}
