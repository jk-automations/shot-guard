"""Sequence check: consecutive takes of one scene. Each character keeps the same side of the frame across the cut
(the 180-degree rule), so the audience never sees people jump sides between shots. Reuses the face tracks."""
import numpy as np

from . import characters
from .characters import side_of


def _side(samples, t0, t1, margin):
    xs = [s["cx"] for s in samples if t0 <= s["t"] <= t1 and not s["edge"]]
    return (side_of(float(np.median(xs)), margin), float(np.median(xs))) if xs else (None, None)


def check(paths, spec, edge=1.0):
    """paths: takes in edit order. Compares the last `edge` seconds of each take with the first `edge` of the next."""
    th = spec["thresholds"]
    people = [characters.people(p, spec) for p in paths]
    issues, pairs = [], []
    for k in range(len(paths) - 1):
        A, B = people[k], people[k + 1]
        for c in spec.get("characters", []):
            n = c["name"]
            if not A["samples"].get(n) or not B["samples"].get(n):
                continue
            sa, xa = _side(A["samples"][n], A["duration"] - edge, A["duration"], th["side_margin"])
            sb, xb = _side(B["samples"][n], 0.0, edge, th["side_margin"])
            pairs.append({"cut": k + 1, "character": n, "end_of_take": sa, "start_of_next": sb})
            if sa and sb and "center" not in (sa, sb) and sa != sb:
                issues.append({"kind": "side_flip_across_cut", "severity": "major", "t0": 0.0, "t1": edge, "frame_t": 0.0,
                               "take": k + 2, "video": paths[k + 1], "character": n, "expected": sa,
                               "message": f"{n} is on the {sa} at the end of take {k + 1} but on the {sb} at the start of take {k + 2}"})
    return {"check": "sequence", "status": "fail" if issues else "pass", "issues": issues, "metrics": {"pairs": pairs, "takes": len(paths)}}
