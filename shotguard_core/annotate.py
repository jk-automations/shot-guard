"""Annotated take: the video itself with Shot Guard's findings drawn on at the second they happen. A playhead runs
along a timeline under the picture, each check's chip lights up when its first problem plays, boxes mark the person or
sign, a cut flashes red, and the last second holds on the verdict. The original sound is kept. Pillow + ffmpeg pipe."""
import os
import subprocess
import tempfile

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .report import CHECK_COLOR, INK, LINE, MUTED, PAPER, VERDICT, _blur_faces, _wrap, font
from .video import ffmpeg_bin, has_audio, is_image

CHECKS = ("dialogue", "speaker", "identity", "blocking", "text", "cuts", "glitch", "audio", "facts", "attention")
HOLD = 1.5   # seconds the verdict card holds after the take


def _box_xyxy(i):
    b = i.get("box")
    if b is None:
        return None
    b = [float(v) for v in b]
    return b if i["check"] == "text" else [b[0], b[1], b[0] + b[2], b[1] + b[3]]


def _shown(i):
    """Seconds an issue's message stays on screen."""
    t0 = i.get("t0") if i.get("t0") is not None else i.get("frame_t", 0.0)
    t1 = i.get("t1") if i.get("t1") is not None else t0
    return t0, max(t1, t0 + 1.6)


def render(result, out_path, blur=True, max_height=1280):
    path = result["video"]
    if is_image(path):
        raise ValueError("Shot Guard Annotate: needs a video take, not a still")
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w0, h0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    s = min(1.0, max_height / h0)
    vw, vh = (int(w0 * s) // 2) * 2, (int(h0 * s) // 2) * 2
    u = vw / 720.0                           # layout unit: sizes scale with the picture width
    hud = int(250 * u) // 2 * 2
    W, H = vw, vh + hud
    dur = n / fps if n else 0.0
    issues = sorted([i for i in result["issues"] if i.get("t0") is not None or i.get("frame_t") is not None],
                    key=lambda i: _shown(i)[0])
    f_chip, f_msg, f_tag, f_big = font(int(16 * u), True), font(int(24 * u), True), font(int(18 * u), True), font(int(64 * u), True)
    raw = tempfile.mktemp(suffix=".mp4")
    enc = subprocess.Popen([ffmpeg_bin(), "-nostdin", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                            "-r", f"{fps:.4f}", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", raw],
                           stdin=subprocess.PIPE)
    last, k = None, 0

    def frame(img_rgb, t, final=False):
        canvas = Image.new("RGB", (W, H), INK)
        canvas.paste(img_rgb, (0, 0))
        d = ImageDraw.Draw(canvas)
        live = [i for i in issues if _shown(i)[0] <= t <= _shown(i)[1]] if not final else []
        for i in live:   # boxes near the frame they were measured on
            b, ft = _box_xyxy(i), i.get("frame_t", i.get("t0"))
            if b is not None and ft is not None and abs(t - ft) <= 0.6:
                col = CHECK_COLOR.get(i["check"], (210, 70, 50))
                d.rectangle([b[0] * s - 6, b[1] * s - 6, b[2] * s + 6, b[3] * s + 6], outline=col, width=max(3, int(5 * u)))
        for i in issues:   # cuts flash the whole picture
            if i["kind"] == "hard_cut" and not final and 0 <= t - i["frame_t"] <= 0.3:
                bw = max(6, int(14 * u))
                d.rectangle([bw // 2, bw // 2, vw - bw // 2, vh - bw // 2], outline=VERDICT["FAIL"], width=bw)
                tag = f"CUT  {i['frame_t']:.2f} s"
                tw = d.textlength(tag, font=f_msg)
                d.rounded_rectangle([vw / 2 - tw / 2 - 18 * u, 40 * u, vw / 2 + tw / 2 + 18 * u, 84 * u], radius=int(10 * u), fill=VERDICT["FAIL"])
                d.text((vw / 2 - tw / 2, 48 * u), tag, font=f_msg, fill=PAPER)
        # brand tag, top left
        d.rounded_rectangle([14 * u, 14 * u, 14 * u + d.textlength("SHOT GUARD", font=f_tag) + 24 * u, 46 * u], radius=int(8 * u), fill=INK)
        d.text((26 * u, 20 * u), "SHOT GUARD", font=f_tag, fill=(200, 190, 178))
        # chips: grey until the playhead reaches a check's first problem; two rows when there are many checks
        y, x = vh + 12 * u, 18 * u
        for c in CHECKS:
            r = result["checks"].get(c)
            if r is None or r.get("status") == "skip":
                continue
            seen = [i for i in issues if i["check"] == c and (final or _shown(i)[0] <= t)]
            major = any(i["severity"] == "major" for i in seen)
            col = VERDICT["FAIL"] if major else VERDICT["RETRY"] if seen else (VERDICT["PASS"] if final else MUTED)
            label = c + ("  FAIL" if major else "  CHECK" if seen else "  OK" if final else "")
            tw = d.textlength(label, font=f_chip) + 18 * u
            if x + tw > W - 12 * u:
                if y > vh + 20 * u:
                    break
                x, y = 18 * u, y + 34 * u
            d.rounded_rectangle([x, y, x + tw, y + 28 * u], radius=int(7 * u), fill=col if seen or final else None, outline=col, width=2)
            d.text((x + 9 * u, y + 5 * u), label, font=f_chip, fill=PAPER if seen or final else (200, 190, 178))
            x += tw + 6 * u
        # timeline: spans appear as the playhead passes them
        ty, x0, x1 = vh + 104 * u, 18 * u, W - 18 * u
        d.rectangle([x0, ty, x1, ty + 10 * u], fill=(70, 64, 58))
        if dur > 0:
            for i in issues:
                a, b = i.get("t0", i.get("frame_t")), i.get("t1") or i.get("t0", i.get("frame_t"))
                if a is None or (a > t and not final):
                    continue
                b = min(b if final else min(b, t), dur)
                xa, xb = x0 + (x1 - x0) * a / dur, x0 + (x1 - x0) * max(b, a + 0.12) / dur
                d.rectangle([xa, ty - 4 * u, xb, ty + 14 * u], fill=CHECK_COLOR.get(i["check"], MUTED))
            att = result["checks"].get("attention", {}).get("metrics", {}).get("curve")
            if att:   # the attention curve rides just above the timeline, drawn up to the playhead
                pts = [(x0 + (x1 - x0) * tt / dur, ty - 6 * u - 26 * u * v) for tt, v in zip(att["t"], att["attention"])
                       if final or tt <= t]
                if len(pts) > 1:
                    d.line(pts, fill=CHECK_COLOR["attention"], width=max(2, int(3 * u)))
            ph = x0 + (x1 - x0) * min(t, dur) / dur
            d.rectangle([ph - 2 * u, ty - 12 * u, ph + 2 * u, ty + 22 * u], fill=PAPER)
            d.text((x1 - 70 * u, ty + 18 * u), f"{min(t, dur):4.1f} s", font=f_chip, fill=(200, 190, 178))
        # newest message
        msg = live[-1] if live else None
        if final:
            v = result["verdict"]
            ov = Image.new("RGBA", (vw, vh), (30, 27, 24, 150))
            canvas.paste(ov, (0, 0), ov)
            d = ImageDraw.Draw(canvas)
            tw = d.textlength(v, font=f_big)
            d.rounded_rectangle([vw / 2 - tw / 2 - 36 * u, vh / 2 - 60 * u, vw / 2 + tw / 2 + 36 * u, vh / 2 + 40 * u], radius=int(16 * u), fill=VERDICT[v])
            d.text((vw / 2 - tw / 2, vh / 2 - 52 * u), v, font=f_big, fill=PAPER)
            sub = f"score {result['score']}  ·  {result['why']}"
            d.text((vw / 2 - d.textlength(sub, font=f_chip) / 2, vh / 2 + 56 * u), sub, font=f_chip, fill=PAPER)
            am = result["checks"].get("attention", {}).get("metrics", {})
            if am.get("hook") is not None:
                at = f"hook {am['hook']}/100  ·  attention {am['attention']}/100"
                d.text((vw / 2 - d.textlength(at, font=f_chip) / 2, vh / 2 + 120 * u), at, font=f_chip, fill=PAPER)
            if result.get("note"):   # e.g. "fixed by auto-trim: 1 retry saved"
                d.text((vw / 2 - d.textlength(result["note"], font=f_chip) / 2, vh / 2 + 88 * u), result["note"], font=f_chip, fill=PAPER)
            line = (result["hints"][0] if result["hints"] else "no fixes needed")
            for m, ln in enumerate(_wrap(d, ("next take: " if result["hints"] else "") + line, f_msg, W - 40 * u)[:2]):
                d.text((18 * u, ty + 46 * u + m * 30 * u), ln, font=f_msg, fill=PAPER)
        elif msg:
            col = CHECK_COLOR.get(msg["check"], MUTED)
            d.text((18 * u, ty + 40 * u), f"{msg['check'].upper()} · {msg['kind'].replace('_', ' ')}", font=f_chip, fill=col)
            for m, ln in enumerate(_wrap(d, msg["message"], f_msg, W - 40 * u)[:2]):
                d.text((18 * u, ty + 64 * u + m * 30 * u), ln, font=f_msg, fill=PAPER)
        else:
            d.text((18 * u, ty + 40 * u), "checking...", font=f_chip, fill=MUTED)
        return canvas

    try:
        while True:
            ok, img = cap.read()
            if not ok:
                break
            if blur:
                img = _blur_faces(img)
            if s < 1.0:
                img = cv2.resize(img, (vw, vh), interpolation=cv2.INTER_AREA)
            else:
                img = img[:vh, :vw]
            last = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            enc.stdin.write(frame(last, k / fps).tobytes())
            k += 1
        if last is not None:
            hold = frame(last, k / fps, final=True).tobytes()
            for _ in range(int(HOLD * fps)):
                enc.stdin.write(hold)
    finally:
        cap.release()
        enc.stdin.close()
        enc.wait()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    cmd = [ffmpeg_bin(), "-nostdin", "-v", "error", "-y", "-i", raw]
    cmd += (["-i", path, "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac"] if has_audio(path) else ["-c", "copy"])
    subprocess.run(cmd + ["-movflags", "+faststart", out_path], check=True)
    os.remove(raw)
    return out_path
