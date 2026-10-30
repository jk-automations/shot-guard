"""Auto-trim: when the only thing wrong with a take is a hidden cut, the longest clean piece between cuts is often
usable as is. Cheaper than a retry: no new generation."""
import subprocess

from .video import ffmpeg_bin, is_image, read_frames

FIXABLE = {"hard_cut", "angle_jump"}


def plan(result, min_seconds=2.0):
    """The longest piece between hard cuts, or None. Says whether trimming fixes every major problem."""
    path = result["video"]
    if is_image(path):
        return None
    # a cut can be a transition several frames long (a dissolve or whip): cut out its whole span, not just its peak
    cuts = sorted((min(i["t0"], i["frame_t"]), max(i["t1"] or i["frame_t"], i["frame_t"]))
                  for i in result["issues"] if i["kind"] == "hard_cut")
    if not cuts:
        return None
    F = read_frames(path, every=1.0, max_side=32)
    dur = F.duration
    pad = 2.0 / (F.fps or 24.0)
    pieces, start = [], 0.0
    for a, b in cuts:
        pieces.append((start, a - pad))
        start = b + pad
    pieces.append((start, dur))
    a, b = max(pieces, key=lambda p: p[1] - p[0])
    if b - a < min_seconds:
        return {"ok": False, "t0": a, "t1": b, "note": f"longest clean piece is {b - a:.1f} s, under {min_seconds:g} s: retry instead"}
    left = [i for i in result["issues"] if i["severity"] == "major" and i["kind"] not in FIXABLE
            and (i.get("t1") if i.get("t1") is not None else i.get("t0", 0.0)) >= a and (i.get("t0") or 0.0) <= b]
    note = f"kept {a:.2f} to {b:.2f} s ({b - a:.1f} of {dur:.1f} s), {len(cuts)} cut(s) removed"
    if left:
        note += "; still has: " + "; ".join(i["message"] for i in left)
    return {"ok": not left, "t0": round(a, 3), "t1": round(b, 3), "seconds": round(b - a, 3), "note": note}


def recheck(path, spec):
    """Cuts check on the trimmed piece: a trim only counts as a fix if the piece itself has no cut left."""
    from . import cuts
    return cuts.check(path, spec)


def cut(path, t0, t1, out_path):
    subprocess.run([ffmpeg_bin(), "-nostdin", "-v", "error", "-y", "-ss", f"{t0:.3f}", "-i", path, "-t", f"{t1 - t0:.3f}",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-c:a", "aac", "-movflags", "+faststart", out_path],
                   check=True)
    return out_path
