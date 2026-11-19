"""Stillness: a photograph with a camera move, and too little motion against a reference ad.

Image-to-video fallbacks often ship a still picture with a 2D zoom or pan on top. Frame to frame it changes (the camera
moves), so Glitch's freeze test does not fire, but nothing inside the picture moves. For each pair of frames about
1/8 s apart this check tracks corner points, fits one zoom/pan/rotate transform (RANSAC), undoes that camera move and
measures what still changes inside the picture (the busiest 5%). A real shot keeps changing (hands, steam, people).

Second part: when the spec names a reference ad (`motion_reference`), the take's motion profile (median residual
motion, share of photo-like frame pairs) is compared with the reference's, measured the same way. Only this comparison
fails a take: without a reference, photo-like stretches and a high photo-like share are warnings, because calm but real
dialogue takes also score high.

Dark frames have little texture, so their residual is small even when something moves: photo-like pairs darker than
`still_dark_luma` are reported as "too dark to judge" and left out of the share. Stretches the spec marks as meant to
be still (`static_ok`, for example an end card) are left out too.

Prototype: a JK judge probe, 2026-11-18; thresholds are that probe's, uncalibrated."""
import cv2
import numpy as np

from .video import is_image, read_frames


def profile(path, th):
    """Per frame pair: time, raw change, residual after undoing the camera move, inlier share, camera move, luma."""
    F = read_frames(path, every=th["still_every"], max_side=240)
    imgs = [cv2.GaussianBlur(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), (7, 7), 0) for im in F.images]
    rows = []
    for k, (a, b) in enumerate(zip(imgs, imgs[1:])):
        raw = float(np.mean(np.abs(b.astype(np.float32) - a.astype(np.float32))))
        row = {"t0": float(F.times[k]), "t1": float(F.times[k + 1]), "raw": raw, "resid": None, "inl": None, "cam": 0.0,
               "luma": float((a.mean() + b.mean()) / 2), "cut": raw >= th["still_cut"]}
        if not row["cut"]:
            pts = cv2.goodFeaturesToTrack(a, 300, 0.01, 6)
            if pts is not None and len(pts) >= 12:
                nxt, st, _ = cv2.calcOpticalFlowPyrLK(a, b, pts, None)
                ok = st.ravel() == 1
                if ok.sum() >= 12:
                    M, inl = cv2.estimateAffinePartial2D(pts[ok], nxt[ok], method=cv2.RANSAC, ransacReprojThreshold=1.0)
                    if M is not None:
                        h, w = a.shape
                        warped = cv2.warpAffine(a, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                        y0, y1, x0, x1 = int(h * .12), int(h * .88), int(w * .12), int(w * .88)
                        diff = np.abs(b[y0:y1, x0:x1].astype(np.float32) - warped[y0:y1, x0:x1].astype(np.float32))
                        row["resid"] = float(np.percentile(diff, 95))   # the busiest 5% of the picture
                        row["inl"] = float(inl.mean())
                        scale = float(np.hypot(M[0, 0], M[0, 1]))
                        row["cam"] = float(abs(scale - 1) * 100 + np.hypot(M[0, 2], M[1, 2]))
        rows.append(row)
    return F, rows


def _in_spans(t0, t1, spans):
    return any(a <= (t0 + t1) / 2 <= b for a, b in spans)


def classify(rows, th, static_ok=()):
    """Label every pair: photo (photo-like, lit), dark (photo-like but too dark to judge), moving, cut,
    unmeasured (too few corners to fit a camera move), allowed (inside a static_ok span)."""
    out = []
    for r in rows:
        if _in_spans(r["t0"], r["t1"], static_ok):
            lab = "allowed"
        elif r["cut"]:
            lab = "cut"
        elif r["resid"] is None:
            lab = "unmeasured"
        elif r["resid"] <= th["still_resid"] and r["inl"] >= th["still_inliers"]:
            lab = "dark" if r["luma"] < th["still_dark_luma"] else "photo"
        else:
            lab = "moving"
        out.append(lab)
    return out


def summary(rows, labels):
    """The motion profile compared against a reference: median residual of the measured lit pairs, photo-like share."""
    lit = [r["resid"] for r, lab in zip(rows, labels) if lab in ("photo", "moving")]
    n = len(lit)
    return {"median_residual": round(float(np.median(lit)), 2) if n else None,
            "photo_share": round(sum(lab == "photo" for lab in labels) / n, 3) if n else None,
            "dark_share": round(sum(lab == "dark" for lab in labels) / max(1, len(labels)), 3),
            "measured_pairs": n, "pairs": len(labels)}


def _runs(labels, rows, want, min_s):
    out, start = [], None
    for k, lab in enumerate(list(labels) + [None]):
        if lab in want and start is None:
            start = k
        elif lab not in want and start is not None:
            t0, t1 = rows[start]["t0"], rows[k - 1]["t1"]
            if t1 - t0 >= min_s - 1e-6:
                out.append((start, k))
            start = None
    return out


def reference_profile(ref, th):
    """`motion_reference` is {"path": video, "static_ok": [[t0, t1]]} or precomputed numbers
    {"median_residual": x, "photo_share": y}."""
    if ref.get("median_residual") is not None and ref.get("photo_share") is not None:
        return {"median_residual": float(ref["median_residual"]), "photo_share": float(ref["photo_share"]),
                "source": "spec numbers"}
    _, rows = profile(ref["path"], th)
    s = summary(rows, classify(rows, th, ref.get("static_ok") or ()))
    s["source"] = "video"
    return s


def check(path, spec):
    if is_image(path):
        return {"check": "stillness", "status": "skip", "issues": [], "metrics": {}, "note": "a still image has no motion to judge"}
    th = spec["thresholds"]
    static_ok = [tuple(s) for s in spec.get("static_ok") or []]
    F, rows = profile(path, th)
    if len(rows) < 4:
        return {"check": "stillness", "status": "skip", "issues": [], "metrics": {}, "note": "too few frames"}
    labels = classify(rows, th, static_ok)
    s = summary(rows, labels)
    issues = []
    for a, b in _runs(labels, rows, {"photo"}, th["still_min_s"]):
        t0, t1 = rows[a]["t0"], rows[b - 1]["t1"]
        cam = float(np.median([r["cam"] for r in rows[a:b]]))
        moved = cam > th["still_cam"]
        what = "a photograph with a camera move" if moved else "a still picture with no movement at all"
        issues.append({"kind": "photo_like" if moved else "no_motion", "severity": "minor", "t0": t0, "t1": t1,
                       "frame_t": (t0 + t1) / 2, "camera_move": round(cam, 2),
                       "residual": round(float(np.median([r["resid"] for r in rows[a:b]])), 2),
                       "message": f"{t0:.2f}-{t1:.2f} s looks like {what}: nothing inside the picture moves"})
    for a, b in _runs(labels, rows, {"dark"}, th["still_min_s"]):
        t0, t1 = rows[a]["t0"], rows[b - 1]["t1"]
        issues.append({"kind": "too_dark_to_judge", "severity": "minor", "t0": t0, "t1": t1, "frame_t": (t0 + t1) / 2,
                       "message": f"{t0:.2f}-{t1:.2f} s is too dark to tell a moving shot from a photograph; check by eye"})
    share = s["photo_share"]
    if share is not None and share >= th["still_share_warn"]:
        # a warning on its own: calm dialogue takes from Shot Guard's own approved pilot film read 32-36% photo-like
        # (docs/LESSONS.md). Only the comparison with a reference ad fails a take.
        issues.append({"kind": "too_many_stills", "severity": "minor", "t0": rows[0]["t0"], "t1": rows[-1]["t1"], "share": share,
                       "message": f"{round(100 * share)}% of the measured frame pairs look like photographs: "
                                  "too many static images? Compare with a reference ad (motion_reference) to judge"})
    ref = None
    if spec.get("motion_reference"):
        ref = reference_profile(spec["motion_reference"], th)
        mine, theirs = s["median_residual"], ref.get("median_residual")
        if mine is not None and theirs:
            ratio = mine / theirs
            gap = (share or 0.0) - (ref.get("photo_share") or 0.0)
            ref.update(ratio=round(ratio, 3), share_gap=round(gap, 3))
            major = ratio < th["still_ref_ratio"] or gap >= th["still_ref_gap"]
            minor = ratio < th["still_ref_ratio_warn"] or gap >= th["still_ref_gap_warn"]
            if major or minor:
                issues.append({"kind": "below_reference_motion", "severity": "major" if major else "minor",
                               "t0": rows[0]["t0"], "t1": rows[-1]["t1"], "ratio": round(ratio, 3), "share_gap": round(gap, 3),
                               "message": f"moves far less than the reference ad: median motion {mine:.1f} against {theirs:.1f} "
                                          f"({round(100 * ratio)}%), photo-like {round(100 * (share or 0))}% against "
                                          f"{round(100 * (ref.get('photo_share') or 0))}%"})
    curve = {"t": [round(r["t0"], 3) for r in rows], "residual": [None if r["resid"] is None else round(r["resid"], 2) for r in rows],
             "label": labels}
    return {"check": "stillness", "status": "fail" if any(i["severity"] == "major" for i in issues) else "pass",
            "issues": issues, "metrics": dict(s, reference=ref, curve=curve, static_ok=static_ok,
                                              stretches=[[round(i["t0"], 2), round(i["t1"], 2)] for i in issues
                                                         if i["kind"] in ("photo_like", "no_motion")])}
