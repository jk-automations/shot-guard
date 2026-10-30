"""On-screen text check: Florence-2 OCR (MIT, CPU) on stills or sampled video frames. Every word must be on the
spec's allow-list; near misses of an allowed word are reported as garbled ("MEGA-BUKEN" for "MEGABUN")."""
import os
import re
from difflib import SequenceMatcher

import cv2

from .video import is_image, read_frames

FLORENCE_ID = "microsoft/Florence-2-base"
_cache = {}


def florence_path():
    p = os.environ.get("SHOTGUARD_FLORENCE")
    if p:
        return p
    cands = [os.path.expanduser("~/ComfyUI/models/LLM/Florence-2-base")]
    try:
        import folder_paths
        cands.insert(0, os.path.join(folder_paths.models_dir, "LLM", "Florence-2-base"))
    except ImportError:
        pass
    return next((c for c in cands if os.path.isdir(c)), FLORENCE_ID)   # else download from Hugging Face


def florence():
    if "f2" not in _cache:
        from unittest.mock import patch
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor
        from transformers.dynamic_module_utils import get_imports

        def no_flash(f):   # the remote code lists flash_attn as an import; it is not needed on CPU
            return [i for i in get_imports(f) if i != "flash_attn"]
        with patch("transformers.dynamic_module_utils.get_imports", no_flash):
            path = florence_path()
            model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.float32,
                                                         attn_implementation="eager").to("cpu").eval()
            proc = AutoProcessor.from_pretrained(path, trust_remote_code=True)
        _cache["f2"] = (model, proc, torch)
    return _cache["f2"]


def ocr(img_bgr):
    """[(text, box x0, y0, x1, y1)] in pixels of img_bgr."""
    from PIL import Image
    model, proc, torch = florence()
    im = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    task = "<OCR_WITH_REGION>"
    inp = proc(text=task, images=im, return_tensors="pt")
    with torch.no_grad():   # use_cache=False: the remote code breaks on the new transformers KV cache
        ids = model.generate(input_ids=inp["input_ids"], pixel_values=inp["pixel_values"], max_new_tokens=256,
                             num_beams=3, do_sample=False, use_cache=False)
    raw = proc.batch_decode(ids, skip_special_tokens=False)[0]
    r = proc.post_process_generation(raw, task=task, image_size=im.size)[task]
    out = []
    for label, q in zip(r.get("labels", []), r.get("quad_boxes", [])):
        xs, ys = q[0::2], q[1::2]
        out.append((label.replace("</s>", "").replace("<s>", "").strip(), (min(xs), min(ys), max(xs), max(ys))))
    return out


def norm(s):
    return re.sub(r"[^A-Z0-9]", "", s.upper())


def tokens(text):
    """Word-like pieces with at least 3 Latin letters (OCR junk such as stray digits or symbols is ignored)."""
    return [t for t in re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-]*", text) if len(re.sub(r"[^A-Za-z]", "", t)) >= 3]


def judge(token, allowed):
    """None if the token is allowed; else (kind, closest allowed word)."""
    n = norm(token)
    allowed_n = [norm(a) for a in allowed if norm(a)]
    if n in allowed_n:
        return None
    if any(len(n) >= 3 and n in a for a in allowed_n):   # partly hidden sign: "BUN" of "MEGABUN"
        return None
    best = max(allowed_n, key=lambda a: SequenceMatcher(None, n, a).ratio(), default=None)
    if best and SequenceMatcher(None, n, best).ratio() >= 0.5:
        return ("garbled", best)
    return ("not_allowed", best)


def edits(a, b):
    """Levenshtein distance."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def check(path, spec):
    th = spec["thresholds"]
    allowed = [a for a in spec.get("allowed_text", []) if a.strip()]
    allowed_words = allowed + [w for a in allowed for w in tokens(a)]
    if is_image(path):
        times, images = [0.0], [cv2.imread(path)]
    else:
        F = read_frames(path, every=th["ocr_every"])
        times, images = F.times, F.images
    if not is_image(path) and th.get("allow_start_text") and images:
        allowed_words = allowed_words + [tok for text, _ in ocr(images[0]) for tok in tokens(text)]
    seen = {}   # normalized token -> list of (t, raw, box)
    reads = []
    for t, img in zip(times, images):
        found = ocr(img)
        reads.append({"t": t, "text": [f[0] for f in found]})
        for text, box in found:
            if box[3] - box[1] < th["min_text_height"] * img.shape[0]:
                continue   # small background text (founder call 10-28: only matters for close-up ads)
            for tok in tokens(text):
                seen.setdefault(norm(tok), []).append((t, tok, box, text))
    min_hits = 1 if len(images) == 1 else th["ocr_min_hits"]
    issues, lines_done = [], set()
    for n, hits in seen.items():
        verdict = judge(hits[0][1], allowed_words)
        if verdict is None or len(hits) < min_hits:
            continue
        kind, best = verdict
        if kind == "garbled" and best in seen and edits(n, best) <= 1:
            continue   # a one-letter misread of a sign that is also read correctly in this take: OCR noise
        t, tok, box, text = hits[0]
        if (kind, text) in lines_done:
            continue
        lines_done.add((kind, text))
        expected = next((a for a in allowed if norm(a) == best), best)
        msg = (f"sign reads \"{text}\", expected \"{expected}\"" if kind == "garbled"
               else f"unexpected text \"{text}\" (not on the allowed list)")
        issues.append({"kind": kind, "severity": "major", "t0": hits[0][0], "t1": hits[-1][0], "frame_t": t, "box": box,
                       "token": tok, "expected": expected if kind == "garbled" else None, "message": msg})
    return {"check": "text", "status": "fail" if issues else "pass", "issues": issues,
            "metrics": {"frames_read": len(images), "reads": reads, "allowed": allowed}}
