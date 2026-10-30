"""Report: one contact sheet PNG (verdict, timeline with flagged spans, one card per issue with the frame, a box
and the timecode, fix hints) plus a JSON receipt. Pillow only: no ffmpeg drawtext or libass needed."""
import json
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from . import spec as spec_mod
from .video import frame_at, is_image, read_frames

INK, PAPER, MUTED, LINE = (38, 34, 31), (246, 242, 236), (120, 112, 104), (222, 214, 204)
VERDICT = {"PASS": (86, 128, 84), "RETRY": (176, 128, 62), "FAIL": (170, 82, 62)}
CHECK_COLOR = {"dialogue": (92, 112, 150), "identity": (150, 92, 120), "blocking": (176, 128, 62),
               "text": (86, 128, 120), "cuts": (170, 82, 62), "audio": (70, 120, 150), "facts": (120, 110, 70), "sequence": (110, 96, 140),
               "speaker": (60, 110, 160), "glitch": (150, 110, 60), "attention": (130, 100, 150)}
W, PAD = 1600, 40


def font(size, bold=False):
    for name in (("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"), "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
                 if bold else "/System/Library/Fonts/Supplemental/Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def _wrap(draw, text, f, width):
    lines, cur = [], ""
    for w in text.split():
        t = (cur + " " + w).strip()
        if draw.textlength(t, font=f) <= width:
            cur = t
        else:
            lines.append(cur)
            cur = w
    return lines + ([cur] if cur else [])


def _blur_faces(img_bgr):
    from . import faces as fc
    out = img_bgr.copy()
    for f in fc.detect(img_bgr, min_score=0.6, min_rel=0.015):
        x, y, w, h = (int(v) for v in f["box"])
        x0, y0 = max(0, x - w // 3), max(0, y - h // 2)
        x1, y1 = min(out.shape[1], x + w + w // 3), min(out.shape[0], y + h + h // 3)
        if x1 > x0 and y1 > y0:
            k = max(31, (max(w, h) // 2) | 1)
            out[y0:y1, x0:x1] = cv2.GaussianBlur(out[y0:y1, x0:x1], (k, k), 0)
    return out


def _thumb(path, t, height, blur, box=None, color=None):
    img = frame_at(path, t if t is not None else 0.0)
    if img is None:
        img = np.full((height, height * 9 // 16, 3), 200, np.uint8)
    if blur:
        img = _blur_faces(img)
    H, Wd = img.shape[:2]
    s = height / H
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).resize((max(1, int(Wd * s)), height), Image.LANCZOS)
    if box is not None:   # box is (x, y, w, h) in frame pixels
        d = ImageDraw.Draw(pil)
        x0, y0, x1, y1 = box[0] * s, box[1] * s, (box[0] + box[2]) * s, (box[1] + box[3]) * s
        d.rectangle([x0 - 4, y0 - 4, x1 + 4, y1 + 4], outline=color or (210, 70, 50), width=4)
    return pil


def render(result, out_dir, prefix="shotguard", blur=True):
    """Write <prefix>.png and <prefix>.json into out_dir. Returns (PIL image, png path, json path)."""
    os.makedirs(out_dir, exist_ok=True)
    path, sp = result["video"], result["spec"]
    issues = [i for i in result["issues"] if i.get("t0") is not None or i.get("frame_t") is not None][:8]
    other = [i for i in result["issues"] if i not in issues]
    still = is_image(path)
    if still:
        dur, times = 0.0, [0.0]
    else:
        F = read_frames(path, every=0, max_side=64)
        dur = F.duration or (F.times[-1] if F.times else 0.0)
        times = [dur * (k + 0.5) / 8 for k in range(8)]
    f_big, f_mid, f_small, f_bold = font(44, True), font(24), font(20), font(22, True)
    probe = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    hint_lines = [ln for k, h in enumerate(result["hints"]) for ln in _wrap(probe, f"{k + 1}. {h}", f_mid, W - 2 * PAD)]
    card_w, cols = (W - 2 * PAD - 3 * 24) // 4, 4
    card_h = 520
    rows = (len(issues) + cols - 1) // cols
    strip_h = 300
    Ht = 190 + 50 + 60 + strip_h + 90 + rows * (card_h + 24) + (40 + 32 * len(other) if other else 0) + 60 + 34 * len(hint_lines) + PAD
    sheet = Image.new("RGB", (W, Ht), PAPER)
    d = ImageDraw.Draw(sheet)
    # header
    d.rectangle([0, 0, W, 170], fill=INK)
    d.text((PAD, 28), "SHOT GUARD", font=f_bold, fill=(200, 190, 178))
    d.text((PAD, 62), sp.get("name", "take"), font=f_big, fill=PAPER)
    spent = f"  ·  spent {result['spent']:g}" + (f" of {sp['credit_cap']:g}" if sp.get("credit_cap") else "") if sp.get("cost_per_take") else ""
    att = result["checks"].get("attention", {}).get("metrics", {})
    spent += f"  ·  hook {att['hook']}  ·  attention {att['attention']}" if att.get("hook") is not None else ""
    d.text((PAD, 122), f"{os.path.basename(path)}  ·  attempt {result['attempt']}  ·  score {result['score']}{spent}  ·  {result['why']}",
           font=f_small, fill=(200, 190, 178))
    v = result["verdict"]
    vw = d.textlength(v, font=f_big) + 56
    d.rounded_rectangle([W - PAD - vw, 50, W - PAD, 118], radius=12, fill=VERDICT[v])
    d.text((W - PAD - vw + 28, 58), v, font=f_big, fill=PAPER)
    # check summary
    y = 190
    x = PAD
    for name in ("dialogue", "speaker", "identity", "blocking", "text", "cuts", "glitch", "audio", "facts", "sequence", "attention"):
        c = result["checks"].get(name)
        st = "not run" if c is None else c["status"].upper()
        if st == "PASS" and c.get("issues"):
            st = "WARN"   # passed, with minor findings worth a look
        col = (MUTED if c is None or st == "SKIP" else VERDICT["PASS"] if st == "PASS" else VERDICT["RETRY"] if st == "WARN"
               else VERDICT["FAIL"])
        label = f"{name}  {st}"
        tw = d.textlength(label, font=f_small) + 28
        if x + tw > W - PAD:
            x, y = PAD, y + 50
        d.rounded_rectangle([x, y, x + tw, y + 36], radius=8, outline=col, width=2)
        d.text((x + 14, y + 7), label, font=f_small, fill=col)
        x += tw + 14
    # timeline strip
    y += 60
    thumbs = [_thumb(path, t, strip_h - 50, blur) for t in times]
    gap = 10
    tot = sum(t.width for t in thumbs) + gap * (len(thumbs) - 1)
    s = min(1.0, (W - 2 * PAD) / tot)
    x = PAD + max(0, (W - 2 * PAD - int(tot * s)) // 2)
    for t, th in zip(times, thumbs):
        if s < 1:
            th = th.resize((int(th.width * s), int(th.height * s)))
        sheet.paste(th, (x, y))
        d.text((x, y + th.height + 6), f"{t:.1f} s" if not still else "still", font=f_small, fill=MUTED)
        x += th.width + gap
    y += strip_h
    if not still and dur > 0:
        x0, x1 = PAD, W - PAD
        d.rectangle([x0, y + 20, x1, y + 28], fill=LINE)
        for i in result["issues"]:
            if i.get("t0") is None:
                continue
            a = x0 + (x1 - x0) * max(0.0, i["t0"]) / dur
            b = x0 + (x1 - x0) * min(dur, max(i["t1"] or i["t0"], i["t0"] + 0.15)) / dur
            d.rectangle([a, y + 14, max(b, a + 6), y + 34], fill=CHECK_COLOR.get(i["check"], MUTED))
        for k in range(int(dur) + 1):
            xx = x0 + (x1 - x0) * k / dur
            d.text((xx - 6, y + 40), f"{k}", font=f_small, fill=MUTED)
    y += 90
    # issue cards
    for k, i in enumerate(issues):
        cx, cy = PAD + (k % cols) * (card_w + 24), y + (k // cols) * (card_h + 24)
        col = CHECK_COLOR.get(i["check"], MUTED)
        d.rounded_rectangle([cx, cy, cx + card_w, cy + card_h], radius=12, fill=(255, 252, 248), outline=LINE, width=2)
        t = i.get("frame_t", i.get("t0"))
        box = i.get("box")
        if box is not None and i["check"] == "text":
            box = (box[0], box[1], box[2] - box[0], box[3] - box[1])
        th = _thumb(i.get("video") or path, t, 330, blur, box, col)
        if th.width > card_w - 24:
            th = th.resize((card_w - 24, int(th.height * (card_w - 24) / th.width)))
        sheet.paste(th, (cx + (card_w - th.width) // 2, cy + 12))
        ty = cy + 12 + 330 + 12
        tc = "" if t is None or still else (f"{i['t0']:.1f}–{i['t1']:.1f} s" if i.get("t1") not in (None, i["t0"]) else f"{t:.1f} s")
        d.text((cx + 14, ty), f"{i['check'].upper()} · {i['kind'].replace('_', ' ')}", font=f_bold, fill=col)
        d.text((cx + 14, ty + 30), tc, font=f_small, fill=MUTED)
        for n, ln in enumerate(_wrap(d, i["message"], f_small, card_w - 28)[:4]):
            d.text((cx + 14, ty + 58 + n * 24), ln, font=f_small, fill=INK)
    y += rows * (card_h + 24)
    if other:
        d.text((PAD, y + 4), "Other findings", font=f_bold, fill=INK)
        for n, i in enumerate(other):
            d.text((PAD, y + 36 + n * 32), f"{i['check']}: {i['message']}", font=f_small, fill=INK)
        y += 40 + 32 * len(other)
    d.text((PAD, y + 14), "Fix hints for the next take" if result["hints"] else "No fixes needed", font=f_bold, fill=INK)
    for n, ln in enumerate(hint_lines):
        d.text((PAD, y + 52 + n * 34), ln, font=f_mid, fill=INK)
    png = os.path.join(out_dir, prefix + ".png")
    sheet.save(png, optimize=True)
    js = os.path.join(out_dir, prefix + ".json")
    with open(js, "w") as f:
        json.dump(receipt(result), f, indent=1, default=_jsonable)
    return sheet, png, js


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def receipt(result):
    r = {k: v for k, v in result.items() if k not in ("spec",)}
    r["spec"] = spec_mod.public(result["spec"])
    r["video"] = os.path.basename(result["video"] or "")
    r["tool"] = "shot-guard 0.1.0"
    return json.loads(json.dumps(r, default=_jsonable))
