"""Places: the same place and the same objects across shots of one cut.

The spec lists `continuity_groups`: things that must look the same in several shots, each with the time spans where it
is on screen, for example {"name": "food truck", "kind": "object", "spans": [[7.3, 9.7], [10.3, 13.2]]}. An optional
"boxes" list gives one crop per span ([x0, y0, x1, y1] as fractions of the frame, or null for the whole frame).

The cut is split into shots with Shot Guard's own cut score (cuts.py), a few frames are taken from the middle of each
span (kept away from the cuts), and each frame is embedded with DINOv2-small (Apache-2.0 weights, CPU), which matches
the same object or place across camera moves and grades. Two spans match when their best frame pair is similar
enough; otherwise the object or place has changed between them (a different truck, a different kitchen).

Thresholds were set on Shot Guard's own pilot-film takes, not on any ad they are used to judge (docs/LESSONS.md)."""
import os

import cv2
import numpy as np

from . import cuts
from .video import is_image

DINO_ID = "facebook/dinov2-small"   # Apache-2.0
_cache = {}


def dino():
    if "dino" not in _cache:
        import torch
        from transformers import AutoImageProcessor, AutoModel
        src = os.environ.get("SHOTGUARD_DINO") or DINO_ID
        model = AutoModel.from_pretrained(src).to("cpu").eval()
        proc = AutoImageProcessor.from_pretrained(src)
        _cache["dino"] = (model, proc, torch)
    return _cache["dino"]


def embed(img_bgr):
    """One L2-normalised DINOv2 CLS vector for a BGR image."""
    from PIL import Image
    model, proc, torch = dino()
    im = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    with torch.no_grad():
        out = model(**proc(images=im, return_tensors="pt"))
    v = out.last_hidden_state[0, 0].numpy().astype(np.float64)
    return v / (np.linalg.norm(v) + 1e-9)


def _crop(img, box):
    if not box:
        return img
    h, w = img.shape[:2]
    x0, y0, x1, y1 = [float(v) for v in box]
    xa, xb, ya, yb = int(max(0, x0) * w), int(min(1, x1) * w), int(max(0, y0) * h), int(min(1, y1) * h)
    return img[ya:yb, xa:xb] if xb - xa >= 16 and yb - ya >= 16 else img


def shots(F, hard, th):
    """Shot boundaries (seconds) from the frame-to-frame cut score."""
    t = F.times
    cut_t = [t[k + 1] for k in range(len(hard)) if hard[k] > th["cut_hard"]]
    return [0.0] + cut_t + [F.duration or (t[-1] if t else 0.0)]


def _shot_of(edges, x):
    for k in range(len(edges) - 1):
        if edges[k] <= x < edges[k + 1]:
            return k
    return len(edges) - 2


def sample_span(F, edges, span, n, guard):
    """Indices of n frames spread over the span, at least `guard` seconds from any cut."""
    t0, t1 = float(span[0]), float(span[1])
    times = np.asarray(F.times)
    ok = (times >= t0) & (times <= t1)
    for e in edges[1:-1]:
        ok &= np.abs(times - e) >= guard
    idx = np.flatnonzero(ok)
    if not len(idx):
        idx = np.flatnonzero((times >= t0) & (times <= t1))
    if not len(idx):
        return []
    return sorted({int(idx[min(len(idx) - 1, int(len(idx) * (i + 0.5) / n))]) for i in range(n)})


def check(path, spec, embedder=None):
    """embedder(bgr) -> unit vector; defaults to DINOv2-small (tests pass a fake)."""
    groups = [g for g in spec.get("continuity_groups") or [] if len(g.get("spans") or []) >= 2]
    if not groups:
        return {"check": "places", "status": "skip", "issues": [], "metrics": {}, "note": "no continuity groups in the spec"}
    if is_image(path):
        return {"check": "places", "status": "skip", "issues": [], "metrics": {}, "note": "a still image has one shot"}
    th = spec["thresholds"]
    embedder = embedder or embed
    F, hard, _ = cuts.scores(path, th["cut_window_frames"], max_side=480)
    edges = shots(F, hard, th)
    issues, rows = [], []
    for g in groups:
        boxes = list(g.get("boxes") or []) + [None] * len(g["spans"])
        vecs, meta = [], []
        for k, span in enumerate(g["spans"]):
            idx = sample_span(F, edges, span, th["places_frames"], th["places_cut_guard"])
            vecs.append([embedder(_crop(F.images[i], boxes[k])) for i in idx])
            meta.append({"span": [float(span[0]), float(span[1])], "frames": [round(F.times[i], 2) for i in idx],
                         "shot": _shot_of(edges, (float(span[0]) + float(span[1])) / 2)})
        pairs = []
        for a in range(len(vecs)):
            for b in range(a + 1, len(vecs)):
                if not vecs[a] or not vecs[b]:
                    pairs.append({"a": a, "b": b, "similarity": None})
                    continue
                sims = np.array([[float(np.dot(x, y)) for y in vecs[b]] for x in vecs[a]])
                i, j = np.unravel_index(int(np.argmax(sims)), sims.shape)
                pairs.append({"a": a, "b": b, "similarity": round(float(sims[i, j]), 3),
                              "frames": [meta[a]["frames"][i], meta[b]["frames"][j]]})
        rows.append({"name": g.get("name", "?"), "kind": g.get("kind", "object"), "spans": meta, "pairs": pairs})
        what = "place" if g.get("kind") == "place" else "object"
        for p in pairs:
            s = p["similarity"]
            A, B = meta[p["a"]]["span"], meta[p["b"]]["span"]
            if s is None:
                issues.append({"kind": "continuity_unmeasured", "severity": "minor", "t0": A[0], "t1": B[1], "group": g.get("name"),
                               "message": f"\"{g.get('name')}\": no frames inside {A[0]:.2f}-{A[1]:.2f} s or {B[0]:.2f}-{B[1]:.2f} s to compare"})
                continue
            if s >= th["places_warn"]:
                continue
            major = s < th["places_fail"]
            issues.append({"kind": "continuity_changed", "severity": "major" if major else "minor", "t0": A[0], "t1": B[1],
                           "frame_t": p["frames"][1], "group": g.get("name"), "what": what, "similarity": s,
                           "spans": [A, B],
                           "message": f"the {what} \"{g.get('name')}\" at {A[0]:.2f}-{A[1]:.2f} s "
                                      f"{'does not match' if major else 'may not match'} the one at {B[0]:.2f}-{B[1]:.2f} s "
                                      f"(similarity {s:.2f}, needs {th['places_warn']:.2f})"})
    return {"check": "places", "status": "fail" if any(i["severity"] == "major" for i in issues) else "pass", "issues": issues,
            "metrics": {"groups": rows, "shots": [round(e, 2) for e in edges], "embedder": "dinov2-small" if embedder is embed else "custom"}}
