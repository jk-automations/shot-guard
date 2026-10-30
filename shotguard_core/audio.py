"""Audio: problems you hear but a frame check never sees. Clipping (distortion), a silent or very quiet track when the
script has lines, dead air in the middle of a take, a music bed as loud as the voice. (A "sound cut off at
the end" rule was tried and dropped: on real takes it fired on music still playing after the last line.) Plain numpy on 50 ms loudness windows, no model. Measured on 24 real takes (docs/LESSONS.md): generated audio
is usually clean, so everything here is a warning except a silent track for a scripted take."""
import subprocess

import numpy as np

from .video import ffmpeg_bin, has_audio, is_image

RATE, WIN = 48000, 2400   # 50 ms windows


def samples(path):
    r = subprocess.run([ffmpeg_bin(), "-nostdin", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(RATE), "-f", "f32le", "-"],
                       capture_output=True)
    return np.frombuffer(r.stdout, np.float32)


def levels(x):
    """Loudness in dBFS per 50 ms window."""
    n = len(x) // WIN
    rms = np.sqrt(np.mean(x[: n * WIN].reshape(n, WIN).astype(np.float64) ** 2, axis=1) + 1e-12)
    return 20 * np.log10(rms)


def _spans(mask, min_len):
    out, start = [], None
    for k, m in enumerate(list(mask) + [False]):
        if m and start is None:
            start = k
        elif not m and start is not None:
            if k - start >= min_len:
                out.append((start, k))
            start = None
    return out


def check(path, spec):
    if is_image(path):
        return {"check": "audio", "status": "skip", "issues": [], "metrics": {}, "note": "a still has no sound"}
    if not has_audio(path):
        return {"check": "audio", "status": "skip", "issues": [], "metrics": {},
                "note": "no audio track" + (" (the dialogue check reports it)" if spec["script"] else "")}
    th = spec["thresholds"]
    x = samples(path)
    if len(x) < 2 * WIN:
        return {"check": "audio", "status": "skip", "issues": [], "metrics": {}, "note": "audio too short"}
    db = levels(x)
    dur = len(x) / RATE
    overall = float(20 * np.log10(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-12))
    clip = float(np.mean(np.abs(x) >= 0.95))   # AAC rounds off flat clipped tops, so count near-full-scale samples
    loud, bed = float(np.percentile(db, 90)), float(np.percentile(db, 20))
    scripted = bool(spec["script"])
    issues = []
    if scripted and overall < th["audio_silent_db"]:
        issues.append({"kind": "silent_track", "severity": "major", "t0": 0.0, "t1": dur,
                       "message": f"the sound track is silent ({overall:.0f} dBFS) but the script has lines"})
    elif overall < th["audio_quiet_db"]:
        issues.append({"kind": "too_quiet", "severity": "minor", "t0": 0.0, "t1": dur,
                       "message": f"the sound is very quiet ({overall:.0f} dBFS overall)"})
    if clip >= th["audio_clip_share"]:
        over = np.abs(x) >= 0.95
        worst = int(np.argmax(np.convolve(over, np.ones(WIN), "valid"))) if len(x) > WIN else 0
        t = worst / RATE
        issues.append({"kind": "clipping", "severity": "major" if clip >= th["audio_clip_major"] else "minor", "t0": t, "t1": t + 0.5,
                       "message": f"the sound clips ({100 * clip:.1f}% of samples near full scale, worst near {t:.1f} s); check by ear for distortion"})
    if scripted and overall >= th["audio_silent_db"]:
        quiet = db < th["audio_gap_db"]
        for a, b in _spans(quiet, int(th["audio_gap_s"] * RATE / WIN)):
            if a == 0 or b >= len(db):
                continue   # leading and trailing quiet are normal
            issues.append({"kind": "dead_air", "severity": "minor", "t0": a * WIN / RATE, "t1": b * WIN / RATE,
                           "message": f"dead air from {a * WIN / RATE:.1f} s to {b * WIN / RATE:.1f} s in a take with dialogue"})
        if loud - bed < th["audio_bed_db"]:
            issues.append({"kind": "music_over_voice", "severity": "minor", "t0": 0.0, "t1": dur,
                           "message": f"the background is nearly as loud as the voice ({loud - bed:.0f} dB apart); check the music does not cover the lines"})
    return {"check": "audio", "status": "fail" if any(i["severity"] == "major" for i in issues) else "pass", "issues": issues,
            "metrics": {"overall_db": round(overall, 1), "clip_share": round(clip, 5), "loud_db": round(loud, 1),
                        "bed_db": round(bed, 1), "seconds": round(dur, 2)}}
