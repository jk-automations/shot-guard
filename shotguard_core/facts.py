"""Facts check: locked facts about props, wardrobe, colours and setting ("the car is glossy red", "no logos on the
van"), judged by a vision LLM on a few frames of the take. The judge is Continuity Lock's (vendored in
continuity.py). Optional: it needs an OpenAI-compatible endpoint that accepts images. Everything else in Shot Guard
is deterministic; this is the one place where a model has to look and judge."""
import cv2
from PIL import Image

from . import continuity as cl
from .video import is_image, read_frames


def _pil(bgr):
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def sample(path, n):
    """n frames spread over the take (a still is one frame)."""
    if is_image(path):
        return [(0.0, _pil(cv2.imread(path)))]
    F = read_frames(path, every=0)
    k = len(F.images)
    idx = sorted({min(k - 1, int(k * (i + 0.5) / n)) for i in range(n)})
    return [(F.times[i], _pil(F.images[i])) for i in idx]


def check(path, spec, judge=None):
    """judge(frame, bible) -> Continuity Lock result dict; defaults to cl.check_image (tests pass a fake)."""
    th = spec["thresholds"]
    facts = cl.parse_bible(spec.get("facts") or [])
    if not facts:
        return {"check": "facts", "status": "skip", "issues": [], "metrics": {}, "note": "no locked facts in the spec"}
    refs = [r if isinstance(r, str) else _pil(r) for r in spec.get("fact_refs") or []]
    bible = cl.Bible.make(facts, refs)
    judge = judge or (lambda img, b: cl.check_image(img, b, url=th.get("judge_url"), model=th.get("judge_model")))
    frames = sample(path, th["facts_frames"])
    per_fact = {f.text: [] for f in facts}
    rows = []
    for t, img in frames:
        r = judge(img, bible)
        rows.append({"t": round(t, 2), "score": r["score"], "facts": r["facts"], "seconds": r.get("seconds")})
        for i in r["issues"]:
            per_fact.setdefault(i["fact"], []).append((t, i))
    need = 1 if len(frames) == 1 else th["facts_min_hits"]
    issues = []
    for f in facts:
        hits = per_fact.get(f.text, [])
        if len(hits) < need:
            continue
        t, first = hits[0]
        issues.append({"kind": "fact_violated", "severity": f.severity, "t0": hits[0][0], "t1": hits[-1][0], "frame_t": t,
                       "fact": f.text, "observed": first["observed"], "fix": first["fix"],
                       "message": f"\"{f.text}\" is broken on {len(hits)} of {len(frames)} frames: {first['observed']}"})
    return {"check": "facts", "status": "fail" if any(i["severity"] == "major" for i in issues) else "pass",
            "issues": issues, "metrics": {"frames": rows, "judge": cl.llm_defaults()["model"] if not th.get("judge_model") else th["judge_model"]}}
