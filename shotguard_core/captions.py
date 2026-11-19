"""Captions: does the picture under each on-screen caption show what the caption says?

Caption spans come from the spec (`captions`: [{"text": ..., "t0": ..., "t1": ...}], for example from an edit
manifest) or, when the spec has none, are read off the take with Florence-2 OCR (text.py): large text that stays on
screen across several OCR frames. For each caption a vision LLM gets a few frames from under it and says, per frame,
whether the picture matches, contradicts or cannot show what the caption claims ("Yep." or "Big feeling." are not
claims about the picture). Like Facts, a caption only breaks when it is contradicted on 2 or more frames (one for a
one-frame span), and the judge is any OpenAI-compatible endpoint that takes images; by default the JK fleet router at
batch priority (continuity.py). With SHOTGUARD_CAPTIONS_OFFLINE=1, or when the judge cannot be reached, the caption
spans are still listed but nothing is scored: the check says "not scored" and never passes a caption it did not see."""
import os
import re
from difflib import SequenceMatcher

import cv2
from PIL import Image

from . import continuity as cl
from .video import frame_at, is_image, read_frames

SYSTEM = ("You check on-screen captions of a video ad against the picture under them. Be literal and judge only what is "
          "visible. Reply with JSON only, no prose.")


def _pil(bgr):
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def messages(caption, images):
    content = []
    for k, im in enumerate(images, 1):
        content += [{"type": "text", "text": f"Frame {k}:"}, {"type": "image_url", "image_url": {"url": cl._b64(im)}}]
    content.append({"type": "text", "text": (
        f"This caption is on screen over all {len(images)} frames: \"{caption}\"\n\n"
        "For each frame decide whether the picture shows what the caption says about the picture.\n"
        'status "matches" = the picture shows it; "contradicts" = the picture clearly shows something else '
        '(the caption names a thing, place, person or action that is not what the frame shows); '
        '"no_claim" = the caption makes no claim a picture can show (dialogue, a feeling, a slogan) or it cannot be seen.\n'
        '"observed" is what the frame actually shows (max 12 words). If it contradicts, "fix" is one short imperative '
        'change to the picture or the caption, else "".\n'
        'Return ONLY: {"frames":[{"id":1,"observed":"...","status":"matches|contradicts|no_claim","fix":"..."}]}')})
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]


def default_judge(th):
    def judge(caption, images):
        parsed, raw = None, ""
        for _ in range(2):   # one retry: a local model sometimes writes broken JSON (an unquoted string)
            raw = cl.chat(messages(caption, images), url=th.get("judge_url"), model=th.get("judge_model"), max_tokens=700)
            parsed = cl.parse_json_loose(raw)
            if parsed and isinstance(parsed.get("frames"), list):
                break
        if not parsed or not isinstance(parsed.get("frames"), list):
            raise RuntimeError(f"caption judge returned unparseable output: {raw[:200]!r}")
        by_id = {}
        for k, it in enumerate(parsed["frames"], 1):
            if isinstance(it, dict):
                try:
                    by_id[int(it.get("id", k))] = it
                except (TypeError, ValueError):
                    by_id[k] = it
        out = []
        for k in range(1, len(images) + 1):
            it = by_id.get(k, {})
            st = str(it.get("status", "no_claim")).lower().strip()
            st = "contradicts" if st.startswith("contra") or st in ("violated", "no", "false") else (
                "matches" if st.startswith("match") or st in ("ok", "yes", "true") else "no_claim")
            out.append({"status": st, "observed": str(it.get("observed", "")), "fix": str(it.get("fix") or "")})
        return out
    return judge


def _norm(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def read_captions(path, th):
    """Caption spans from OCR: lines of large text (taller than captions_min_height of the frame) read on 2+ frames in a row."""
    from . import text
    F = read_frames(path, every=th["captions_ocr_every"], max_side=1024)
    spans = []
    for t, img in zip(F.times, F.images):
        lines = [lab for lab, box in text.ocr(img) if box[3] - box[1] >= th["captions_min_height"] * img.shape[0]
                 and len(re.findall(r"[A-Za-z]", lab)) >= 2]
        line = " ".join(lines).strip()
        if spans and line and SequenceMatcher(None, _norm(spans[-1]["text"]), _norm(line)).ratio() >= 0.75 and t - spans[-1]["t1"] <= th["captions_ocr_every"] + 1e-6:
            spans[-1]["t1"] = t
        elif line:
            spans.append({"text": line, "t0": t, "t1": t, "source": "ocr"})
    return [s for s in spans if s["t1"] > s["t0"]]


def check(path, spec, judge=None):
    """judge(caption, [PIL frames]) -> [{status, observed, fix}] per frame; defaults to the vision LLM (tests pass a fake)."""
    th = spec["thresholds"]
    caps = [dict(c, source="spec") for c in spec.get("captions") or [] if str(c.get("text", "")).strip()]
    if not caps and not spec.get("captions_from_ocr"):
        return {"check": "captions", "status": "skip", "issues": [], "metrics": {}, "note": "no captions in the spec"}
    if is_image(path):
        frames_of = lambda c: [(0.0, _pil(cv2.imread(path)))]
        if not caps:
            return {"check": "captions", "status": "skip", "issues": [], "metrics": {}, "note": "OCR captions need a video"}
    else:
        if not caps:
            caps = read_captions(path, th)

        def frames_of(c):
            t0, t1 = float(c["t0"]), float(c["t1"])
            n = th["captions_frames"]
            out = []
            for i in range(n):
                t = t0 + (t1 - t0) * (i + 0.5) / n
                img = frame_at(path, t)
                if img is not None:
                    out.append((t, _pil(img)))
            return out
    offline = os.environ.get("SHOTGUARD_CAPTIONS_OFFLINE") == "1" and judge is None
    rows = [{"text": c["text"], "t0": float(c["t0"]), "t1": float(c["t1"]), "source": c.get("source", "spec")} for c in caps]
    if offline:
        return {"check": "captions", "status": "skip", "issues": [], "note": "captions read, not scored: offline (no vision judge)",
                "metrics": {"scored": False, "captions": rows}}
    judge = judge or default_judge(th)
    issues, unscored = [], []
    for row, c in zip(rows, caps):
        frames = frames_of(c)
        if not frames:
            row["verdict"] = "no_frames"
            continue
        try:
            verdicts = judge(c["text"], [im for _, im in frames])
        except Exception as e:   # unreachable or broken judge: say so, never pass silently
            row["verdict"] = "not_scored"
            unscored.append(f"{type(e).__name__}: {str(e)[:120]}")
            continue
        row["frames"] = [{"t": round(t, 2), **v} for (t, _), v in zip(frames, verdicts)]
        bad = [f for f in row["frames"] if f["status"] == "contradicts"]
        need = 1 if len(frames) == 1 else th["captions_min_hits"]
        row["verdict"] = "contradicts" if len(bad) >= need else ("no_claim" if all(f["status"] == "no_claim" for f in row["frames"]) else "matches")
        if row["verdict"] == "contradicts":
            issues.append({"kind": "caption_mismatch", "severity": "major", "t0": row["t0"], "t1": row["t1"], "frame_t": bad[0]["t"],
                           "caption": c["text"], "observed": bad[0]["observed"], "fix": bad[0]["fix"],
                           "message": f"caption \"{c['text']}\" ({row['t0']:.2f}-{row['t1']:.2f} s) does not match the picture on "
                                      f"{len(bad)} of {len(frames)} frames: it shows {bad[0]['observed']}"})
    scored = sum(r.get("verdict") in ("contradicts", "matches", "no_claim") for r in rows)
    metrics = {"scored": scored > 0, "scored_captions": scored, "captions": rows,
               "judge": th.get("judge_model") or cl.llm_defaults()["model"]}
    if unscored:
        metrics["judge_errors"] = unscored[:3]
    if scored == 0:
        return {"check": "captions", "status": "skip", "issues": [], "metrics": metrics,
                "note": "captions read, not scored: the vision judge did not answer"}
    if unscored:
        issues.append({"kind": "captions_unscored", "severity": "minor", "t0": rows[0]["t0"], "t1": rows[-1]["t1"],
                       "message": f"{len(unscored)} of {len(rows)} captions were not scored: the vision judge did not answer"})
    return {"check": "captions", "status": "fail" if any(i["severity"] == "major" for i in issues) else "pass",
            "issues": issues, "metrics": metrics}
