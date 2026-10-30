"""Share card: one 1080x1350 image per take for a post or a chat. The frame at the worst moment with its box, the
verdict, a score dial, the hook score and the top finding in one sentence."""
import os

from PIL import Image, ImageDraw

from .report import CHECK_COLOR, INK, MUTED, PAPER, VERDICT, _thumb, _wrap, font
from .video import is_image

W, H = 1080, 1350


def _height(path):
    import cv2
    if is_image(path):
        return cv2.imread(path).shape[0]
    cap = cv2.VideoCapture(path)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return h or 1


def render(result, out_path, blur=True):
    r = result
    bad = next((i for i in r["issues"] if i["severity"] == "major" and (i.get("frame_t") is not None or i.get("t0") is not None)),
               next((i for i in r["issues"] if i.get("frame_t") is not None or i.get("t0") is not None), None))
    t = (bad.get("frame_t") if bad.get("frame_t") is not None else bad["t0"]) if bad else 1.0
    box = bad.get("box") if bad else None
    if box is not None and bad["check"] == "text":
        box = (box[0], box[1], box[2] - box[0], box[3] - box[1])
    col = CHECK_COLOR.get(bad["check"], VERDICT["FAIL"]) if bad else VERDICT["PASS"]
    card = Image.new("RGB", (W, H), INK)
    shot = _thumb(r["video"], t, 860, blur, box, col)
    s = max(W / shot.width, 860 / shot.height)   # fill the top, centred crop
    shot = shot.resize((int(shot.width * s), int(shot.height * s)))
    cx = (shot.width - W) // 2
    orig_h = _height(r["video"])
    cy = (shot.height - 860) // 3 if not box else max(0, min(shot.height - 860, int((box[1] + box[3] / 2) * 860 / orig_h * s) - 430))
    card.paste(shot.crop((cx, cy, cx + W, cy + 860)), (0, 0))
    d = ImageDraw.Draw(card)
    d.rounded_rectangle([36, 36, 36 + d.textlength("SHOT GUARD", font=font(26, True)) + 36, 84], radius=10, fill=INK)
    d.text((54, 45), "SHOT GUARD", font=font(26, True), fill=(200, 190, 178))
    # verdict + dial
    v = r["verdict"]
    fb = font(64, True)
    d.rounded_rectangle([48, 900, 48 + d.textlength(v, font=fb) + 56, 990], radius=16, fill=VERDICT[v])
    d.text((76, 908), v, font=fb, fill=PAPER)
    cxd, cyd, rd = W - 130, 960, 82
    d.arc([cxd - rd, cyd - rd, cxd + rd, cyd + rd], 0, 360, fill=(70, 64, 58), width=14)
    d.arc([cxd - rd, cyd - rd, cxd + rd, cyd + rd], -90, -90 + 3.6 * float(r["score"]), fill=VERDICT[v], width=14)
    num = f"{float(r['score']):.0f}"
    d.text((cxd - d.textlength(num, font=font(56, True)) / 2, cyd - 38), num, font=font(56, True), fill=PAPER)
    d.text((cxd - d.textlength("SCORE", font=font(20, True)) / 2, cyd + 22), "SCORE", font=font(20, True), fill=(200, 190, 178))
    am = r["checks"].get("attention", {}).get("metrics", {})
    if am.get("hook") is not None:
        d.text((48, 1008), f"hook {am['hook']}/100  ·  attention {am['attention']}/100", font=font(30, True), fill=(200, 190, 178))
    # the finding
    head = (f"{bad['check'].upper()} · {bad['kind'].replace('_', ' ')}" if bad else "ALL CHECKS PASSED")
    d.text((48, 1070), head, font=font(28, True), fill=col)
    msg = bad["message"] if bad else (r["hints"][0] if r["hints"] else "Script, cast, shot and sound all match.")
    for k, ln in enumerate(_wrap(d, msg, font(40, True), W - 96)[:3]):
        d.text((48, 1112 + k * 52), ln, font=font(40, True), fill=PAPER)
    d.text((48, H - 52), f"{r['spec'].get('name', 'take')}  ·  checked by Shot Guard, open source, CPU only", font=font(24), fill=MUTED)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    card.save(out_path, optimize=True)
    return card
