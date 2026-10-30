"""Identity drift and blocking. Both read the same face tracks (faces.scan + build_tracks + label_tracks)."""
import os

import numpy as np

from . import faces as fc
from .video import is_image


def people(path, spec):
    """{name: samples} for every spec character that has reference images. Cached on the spec dict per path."""
    key = ("_people", path, os.path.getmtime(path))
    if key in spec.get("_cache", {}):
        return spec["_cache"][key]
    th = spec["thresholds"]
    F, frames = fc.scan(path, every=th["sample_every"])
    from . import cuts
    cut_times = []
    if not is_image(path):
        Fc, hard, _ = cuts.scores(path, th["cut_window_frames"])
        cut_times = [Fc.times[i + 1] for i in np.where(hard > th["cut_hard"])[0]]
    # only characters with reference images are checked by face (a cast list may name people only to map speakers)
    gal = {c["name"]: fc.gallery(c["refs"]) for c in spec.get("characters", []) if c.get("refs")}
    tracks = fc.label_tracks(fc.build_tracks(frames, cut_times), gal, min_match=th["id_min_match"])
    out = {n: fc.character_samples(tracks, n, g) if len(g) else None for n, g in gal.items()}
    res = {"samples": out, "duration": F.duration or 0.0, "width": F.width, "height": F.height, "cuts": cut_times,
           "gallery_sizes": {n: len(g) for n, g in gal.items()}}
    spec.setdefault("_cache", {})[key] = res
    return res


def _runs(ts, mask, min_dur, step):
    """Spans where mask holds for at least min_dur seconds (samples are `step` apart)."""
    out, start = [], None
    for k, m in enumerate(list(mask) + [False]):
        if m and start is None:
            start = k
        elif not m and start is not None:
            if ts[k - 1] - ts[start] + step >= min_dur - 1e-6:
                out.append((start, k - 1))
            start = None
    return out


def _med3(v):
    return np.array([np.median(v[max(0, i - 1):i + 2]) for i in range(len(v))])


def check_identity(path, spec):
    th = spec["thresholds"]
    P = people(path, spec)
    issues, curves = [], {}
    for name, S in P["samples"].items():
        if S is None:
            issues.append({"kind": "no_reference_face", "severity": "minor", "t0": None, "t1": None, "character": name,
                           "message": f"no face found in {name}'s reference images"})
            continue
        S = [s for s in S if not s["edge"]]
        if not S:
            issues.append({"kind": "not_found", "severity": "major", "t0": None, "t1": None, "character": name,
                           "message": f"{name} never appears (no face matches the references)"})
            continue
        ts = np.array([s["t"] for s in S]); sim = np.array([s["sim"] for s in S]); sm = _med3(sim)
        head = sim[: max(3, int(len(sim) * 0.3))]
        base = float(np.median(head))
        k = int(np.argmin(sm))
        curves[name] = {"t": [round(float(x), 2) for x in ts], "sim": [round(float(x), 3) for x in sim], "baseline": round(base, 3),
                        "min": round(float(sim.min()), 3), "mean": round(float(sim.mean()), 3), "worst_t": round(float(ts[k]), 2)}
        bad = (sm < base - th["id_drift_drop"]) | (sm < th["id_min"])
        for a, b in _runs(ts, bad, th["id_min_dur"], th["sample_every"]):
            w = a + int(np.argmin(sm[a:b + 1]))
            # the absolute floor only counts for a face that matched its reference well at the start (a small,
            # distant face starts low and proves nothing)
            severe = base - sm[w] >= th["id_major_drop"] or (sm[w] < th["id_min"] and base >= 0.4)
            issues.append({"kind": "identity_drift", "severity": "major" if severe else "minor", "t0": float(ts[a]), "t1": float(ts[b]), "frame_t": float(ts[w]),
                           "character": name, "box": S[w]["box"],
                           "message": (f"{name}'s face drifts" if severe else f"{name}'s face may drift (check by eye)")
                           + f" from {ts[a]:.1f} s to {ts[b]:.1f} s (likeness {sm[w]:.2f} vs {base:.2f} at the start)"})
    return {"check": "identity", "status": "fail" if any(i["severity"] == "major" for i in issues) else "pass",
            "issues": issues, "metrics": {"curves": curves, "cuts_used": P["cuts"]}}


def side_of(cx, margin):
    return "left" if cx < 0.5 - margin else "right" if cx > 0.5 + margin else "center"


def check_blocking(path, spec):
    th = spec["thresholds"]
    P = people(path, spec)
    issues, tracks = [], {}
    step = th["sample_every"]
    for c in spec.get("characters", []):
        name, want = c["name"], c.get("side", "any")
        S = P["samples"].get(name)
        if not S:
            continue
        ts = np.array([s["t"] for s in S]); cx = np.array([s["cx"] for s in S])
        tracks[name] = {"t": [round(float(x), 2) for x in ts], "cx": [round(float(x), 3) for x in cx], "side": want}
        if want in ("left", "right"):
            wrong = cx > 0.5 + th["side_margin"] if want == "left" else cx < 0.5 - th["side_margin"]
            for a, b in _runs(ts, wrong, th["side_min_dur"], step):
                other = "right" if want == "left" else "left"
                issues.append({"kind": "side_swap", "severity": "major", "t0": float(ts[a]), "t1": float(ts[b]), "frame_t": float(ts[a]),
                               "character": name, "box": S[a]["box"], "expected": want,
                               "message": f"{name} should stay on the {want} but is on the {other} from {ts[a]:.1f} s to {ts[b]:.1f} s"})
        if c.get("stay_in_frame", True) and len(ts):
            dur = P["duration"]
            gaps = list(zip(ts[:-1], ts[1:])) + [(ts[-1], dur)]
            for g0, g1 in gaps:
                if g1 - g0 - step >= th["leave_min_dur"] - 1e-6:
                    issues.append({"kind": "left_frame", "severity": "major", "t0": float(g0 + step), "t1": float(g1), "frame_t": float(g0),
                                   "character": name, "message": f"{name} is out of frame from {g0 + step:.1f} s to {g1:.1f} s"})
    return {"check": "blocking", "status": "fail" if issues else "pass", "issues": issues, "metrics": {"tracks": tracks}}
