"""Render the Shot Guard demo video: the annotated takes playing (tools/make_annotated.py), a running catch counter,
a split-screen fix loop, auto-trim and best-of-N, plus a local voiceover (Kokoro-82M, Apache-2.0, CPU).
python3 tools/make_demo.py [--real]  ->  runs/demo/demo-16x9.mp4 and demo-9x16.mp4 (--real: runs/demo-real/).
Default inputs are public-safe: annotated takes with role names, faces blurred and sound removed on real-person takes.
--real uses runs/annotated-real (the hosts' real names, faces and voices, shown with their OK); never committed.
Narration is captioned in the classic style: a few words at a time in a band under the picture."""
import json
import os
import subprocess
import sys
import wave

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL = "--real" in sys.argv
EX, IMG = os.path.join(ROOT, "docs", "examples"), os.path.join(ROOT, "docs", "img")
ANN = os.path.join(ROOT, "runs", "annotated-real" if REAL else "annotated")
DAILIES = os.path.join(ROOT, "runs", "dailies-real" if REAL else "dailies", "dailies.png")
OUT = os.path.join(ROOT, "runs", "demo-real" if REAL else "demo")
VOICE = os.path.join(OUT, "voice")
FPS, SR = 30, 48000
BG, INK, MUTED, RED, GOLD, GREEN = (30, 27, 24), (240, 235, 227), (170, 160, 148), (196, 92, 70), (190, 140, 70), (110, 160, 105)
FONTS = "/System/Library/Fonts/Supplemental/"
FFMPEG = "/opt/homebrew/bin/ffmpeg"
KOKORO = os.environ.get("SHOTGUARD_KOKORO_PY", "/Users/jefe/venvs/kokoro-tts/bin/python")


def font(size, bold=True):
    return ImageFont.truetype(FONTS + ("Arial Bold.ttf" if bold else "Arial.ttf"), size)


class Clip:
    """A video read by time, so 24 fps takes play at the right speed in a 30 fps edit. Holds the last frame."""
    def __init__(self, name, start=0.0, end=None, speed=1.0):
        self.path = os.path.join(ANN, name + ".mp4")
        cap = cv2.VideoCapture(self.path)
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        self.frames = []
        while True:
            ok, img = cap.read()
            if not ok:
                break
            self.frames.append(img)
        cap.release()
        self.start, self.speed = start, speed
        self.end = end if end is not None else len(self.frames) / self.fps
        self.length = (self.end - self.start) / speed

    def at(self, t):
        k = min(len(self.frames) - 1, int((self.start + min(t, self.length) * self.speed) * self.fps))
        return Image.fromarray(cv2.cvtColor(self.frames[k], cv2.COLOR_BGR2RGB))


def still(path, box=None):
    im = Image.open(path).convert("RGB")
    return im.crop(box) if box else im


def fit(im, w, h):
    s = min(w / im.width, h / im.height)
    return im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.BILINEAR)


def wrap(d, text, f, width):
    lines, cur = [], ""
    for w in text.split():
        t = (cur + " " + w).strip()
        if d.textlength(t, font=f) <= width or not cur:
            cur = t
        else:
            lines.append(cur); cur = w
    return lines + ([cur] if cur else [])


def text_block(d, x, y, width, eyebrow, head, body, accent, scale=1.0):
    fe, fh, fb = font(int(26 * scale)), font(int(60 * scale)), font(int(32 * scale), bold=False)
    if eyebrow:
        d.text((x, y), eyebrow, font=fe, fill=accent); y += int(50 * scale)
    for ln in wrap(d, head, fh, width):
        d.text((x, y), ln, font=fh, fill=INK); y += int(72 * scale)
    y += int(12 * scale)
    for ln in wrap(d, body, fb, width):
        d.text((x, y), ln, font=fb, fill=MUTED); y += int(44 * scale)
    return y


# ---- the edit. Each scene: voice line, minimum seconds, and a draw(frame, t, W, H) function ----

def _who():
    """The speaker catch in words. With --real the names come from the private check result, never from this file."""
    try:
        import pickle
        r = pickle.load(open(os.path.join(ANN, "scrambled_dialogue.result.pkl"), "rb"))
        i = next(i for i in r["issues"] if i["kind"] == "wrong_speaker")
        return f"{i['speaking']} is the one saying {i['character']}'s line." if REAL else "the wrong host is saying the line."
    except (OSError, StopIteration):
        return "the wrong host is saying the line."


WHO = _who()
CATCHES = [   # annotated take, clip window, headline, voice
    ("side_swap", (0.0, 9.5), "Side swap after a hidden cut", "The baker jumps to the wrong side of the frame, right after a hidden cut."),
    ("scrambled_dialogue", (0.0, 9.5), "Wrong lines, wrong mouth",
     "Here the lines come out of order, and " + WHO),
    (None, None, "A misspelled sign", "The sign should say Megabun. The model wrote Mega Buken."),
    ("hidden_cut", (2.5, 9.5), "A cut inside one shot", "This should be one continuous shot. There is a cut at five and a half seconds."),
    ("end_drift", (2.5, 9.5), "Cut, then a different face", "Another hidden cut, and after it the face no longer matches."),
    ("glitch_demo", (1.5, 9.5), "A frozen second (simulated)", "And a frozen second mid take. We froze this one on purpose, to show it."),
]
N_CATCH = len(CATCHES)


def counter(d, x, y, n, scale=1.0):
    f = font(int(30 * scale))
    d.text((x, y), "CAUGHT", font=f, fill=MUTED)
    for k in range(N_CATCH):
        cx = x + int((150 + k * 46) * scale)
        r = int(16 * scale)
        d.ellipse([cx - r, y + int(17 * scale) - r, cx + r, y + int(17 * scale) + r], fill=RED if k < n else None,
                  outline=RED if k < n else (90, 82, 74), width=3)


def scene_catch(k, name, window, head, voice):
    clip = Clip(name, *window) if name else None
    sign = still(os.path.join(EX, "garbled_sign.png"), (129, 652, 313, 982)) if not name else None

    def draw(fr, t, W, H):
        d = ImageDraw.Draw(fr)
        wide = W > H
        if clip:
            im = clip.at(t)
        else:   # the still: slow push-in on the boxed sign
            z = 1.0 + 0.25 * min(1.0, t / 4.0)
            w0, h0 = sign.width / z, sign.height / z
            cx, cy = sign.width / 2, sign.height * 0.32
            im = sign.crop((max(0, cx - w0 / 2), max(0, cy - h0 / 2), min(sign.width, cx + w0 / 2), min(sign.height, cy + h0 / 2)))
        if wide:
            pic = fit(im, int(W * 0.40), H - 60)
            fr.paste(pic, (int(W * 0.06), (H - pic.height) // 2))
            x = int(W * 0.52)
            counter(d, x, int(H * 0.16), k + 1 if t > 0.6 else k)
            text_block(d, x, int(H * 0.34), int(W * 0.42), f"CATCH {k + 1} OF {N_CATCH}", head, "", RED, 1.1)
        else:
            pic = fit(im, W - 60, int(H * 0.72))
            fr.paste(pic, ((W - pic.width) // 2, 40))
            y = 40 + pic.height + 40
            counter(d, 60, y, k + 1 if t > 0.6 else k)
            text_block(d, 60, y + 70, W - 120, f"CATCH {k + 1} OF {N_CATCH}", head, "", RED, 0.95)
    return {"voice": voice, "min": clip.length if clip else 5.0, "draw": draw, "audio": clip.path if clip else None,
            "tick": 0.6}


def scene_cold_open():
    bits = [("side_swap", 2.3), ("scrambled_dialogue", 4.6), ("hidden_cut", 5.45), ("end_drift", 4.65)]
    clips = [Clip(n, s, s + 1.1) for n, s in bits]

    def draw(fr, t, W, H):
        d = ImageDraw.Draw(fr)
        c = clips[min(len(clips) - 1, int(t / 1.1))]
        pic = fit(c.at(t % 1.1), W - 80 if W < H else int(W * 0.42), H - 80 if W > H else int(H * 0.70))
        fr.paste(pic, (int(W * 0.06) if W > H else (W - pic.width) // 2, (H - pic.height) // 2 if W > H else 40))
        if W > H:
            text_block(d, int(W * 0.54), int(H * 0.36), int(W * 0.40), "AI VIDEO", "Every one of these shipped.", "", RED, 1.15)
        else:
            text_block(d, 60, int(H * 0.70) + 90, W - 120, "AI VIDEO", "Every one of these shipped.", "", RED, 1.0)
    return {"voice": "AI video models make mistakes you only find by watching every single take.", "min": 4.4, "draw": draw}


def scene_text(eyebrow, head, body, voice, accent=GOLD, image=None, min_s=3.0):
    def draw(fr, t, W, H):
        d = ImageDraw.Draw(fr)
        if image is None:
            text_block(d, int(W * 0.10), int(H * (0.32 if W > H else 0.36)), int(W * 0.80), eyebrow, head, body, accent, 1.3 if W > H else 1.15)
            return
        z = 1.0 + 0.04 * t / max(min_s, 1)
        if W > H:
            pic = fit(image, int(W * 0.60 * z), int((H - 120) * z))
            fr.paste(pic, (50, (H - pic.height) // 2))
            text_block(d, int(W * 0.64), int(H * 0.30), int(W * 0.32), eyebrow, head, body, accent, 1.0)
        else:
            pic = fit(image, int((W - 60) * z), int(H * 0.55 * z))
            fr.paste(pic, ((W - pic.width) // 2, 80))
            text_block(d, 60, 80 + pic.height + 60, W - 120, eyebrow, head, body, accent, 1.0)
    return {"voice": voice, "min": min_s, "draw": draw}


def scene_pair(left, right, labels, eyebrow, head, voice, colors=(GOLD, GREEN), window_l=None, window_r=None):
    a, b = Clip(left, *(window_l or (0.0, None))), Clip(right, *(window_r or (0.0, None)))

    def draw(fr, t, W, H):
        d = ImageDraw.Draw(fr)
        f = font(30)
        if W > H:
            ph = H - 200
            for k, (c, lab, col) in enumerate(zip((a, b), labels, colors)):
                pic = fit(c.at(t), int(W * 0.30), ph)
                x = int(W * 0.04) + k * int(W * 0.31)
                fr.paste(pic, (x, 140))
                d.text((x, 92), lab, font=f, fill=col)
            text_block(d, int(W * 0.68), int(H * 0.30), int(W * 0.29), eyebrow, head, "", colors[1], 0.95)
        else:
            ph = int(H * 0.62)
            for k, (c, lab, col) in enumerate(zip((a, b), labels, colors)):
                pic = fit(c.at(t), W // 2 - 40, ph)
                x = 20 + k * (W // 2)
                fr.paste(pic, (x, 120))
                d.text((x, 70), lab, font=font(26), fill=col)
            text_block(d, 60, 120 + ph + 60, W - 120, eyebrow, head, "", colors[1], 0.95)
    return {"voice": voice, "min": max(a.length, b.length), "draw": draw, "audio": b.path}


def scene_best_of():
    data = json.load(open(os.path.join(ANN, "best_of.json")))
    lines = data["ranking"].splitlines()

    def draw(fr, t, W, H):
        d = ImageDraw.Draw(fr)
        x, y = int(W * 0.10), int(H * (0.22 if W > H else 0.26))
        d.text((x, y), "BEST OF N", font=font(28), fill=GOLD)
        d.text((x, y + 50), "Three takes of one shot, ranked", font=font(56 if W > H else 50), fill=INK)
        for k, ln in enumerate(lines):
            if t < 0.8 + 0.7 * k:
                continue
            yy = y + 170 + k * 90
            col = GREEN if k == 0 else MUTED
            d.rounded_rectangle([x, yy, x + 60, yy + 60], radius=12, fill=col)
            d.text((x + 19, yy + 10), str(k + 1), font=font(36), fill=BG)
            for m, part in enumerate(wrap(d, ln.split(". ", 1)[1], font(30, bold=False), W - x - 160)[:2]):
                d.text((x + 90, yy + 2 + m * 36), part, font=font(30, bold=k == 0 and m == 0), fill=INK if k == 0 else MUTED)
    return {"voice": "Generate a few takes, and Shot Guard ranks them and keeps the best one.", "min": 4.5, "draw": draw}


def scene_attention():
    clip = Clip("side_swap-attempt1")

    def draw(fr, t, W, H):
        d = ImageDraw.Draw(fr)
        if W > H:
            pic = fit(clip.at(t), int(W * 0.40), H - 60)
            fr.paste(pic, (int(W * 0.06), (H - pic.height) // 2))
            text_block(d, int(W * 0.52), int(H * 0.28), int(W * 0.42), "WILL THEY KEEP WATCHING?", "Hook score and an attention curve",
                       "Movement, faces, speech and sound, every quarter second. Rules, not a black box.", GOLD, 1.0)
        else:
            pic = fit(clip.at(t), W - 60, int(H * 0.70))
            fr.paste(pic, ((W - pic.width) // 2, 40))
            text_block(d, 60, 40 + pic.height + 50, W - 120, "WILL THEY KEEP WATCHING?", "Hook score and an attention curve", "", GOLD, 0.95)
    return {"voice": "It also scores the hook and draws an attention curve, like a virality predictor. Except this one is open, "
                     "and runs on your own machine.", "min": clip.length, "draw": draw, "audio": clip.path}


def scene_dailies():
    page = still(DAILIES)

    def draw(fr, t, W, H):
        d = ImageDraw.Draw(fr)
        if W > H:
            pw = int(W * 0.56)
            pic = page.resize((pw, int(page.height * pw / page.width)))
            vis = H - 80
            off = int(max(0, pic.height - vis) * min(1.0, max(0.0, (t - 1.0) / 5.0)))   # slow scroll down the ranking
            fr.paste(pic.crop((0, off, pw, off + min(vis, pic.height))), (40, 40))
            text_block(d, int(W * 0.62), int(H * 0.28), int(W * 0.34), "DAILIES", "A folder of takes in. One ranked page out.",
                       "Best take first. Hidden cuts trimmed. Credits saved, counted.", GOLD, 0.95)
        else:
            pw = W - 40
            pic = page.resize((pw, int(page.height * pw / page.width)))
            vis = int(H * 0.58)
            off = int(max(0, pic.height - vis) * min(1.0, max(0.0, (t - 1.0) / 5.0)))
            fr.paste(pic.crop((0, off, pw, off + min(vis, pic.height))), (20, 60))
            text_block(d, 60, 60 + vis + 60, W - 120, "DAILIES", "A folder of takes in. One ranked page out.", "", GOLD, 1.0)
    return {"voice": "Or point it at a whole folder of takes. You get one page: best take first, cuts trimmed, and the retries it saved.",
            "min": 7.0, "draw": draw}


def scene_plug():
    shots = [still(os.path.join(IMG, f)) for f in ("plug-in-seedance-2-5.png", "plug-in-tiny-world.png", "plug-in-underdog.png")]
    labels = ["After the Seedance 2.5 template", "After an LTX graph", "After a Veo graph"]

    def draw(fr, t, W, H):
        k = min(2, int(t / 2.0))
        d = ImageDraw.Draw(fr)
        if W > H:
            pic = fit(shots[k], int(W * 0.64), H - 120)
            fr.paste(pic, (40, (H - pic.height) // 2))
            text_block(d, int(W * 0.68), int(H * 0.32), int(W * 0.29), "PLUGS IN ANYWHERE", labels[k], "CPU only. Open licences.", GOLD, 0.95)
        else:
            pic = fit(shots[k], W - 40, int(H * 0.5))
            fr.paste(pic, (20, 160))
            text_block(d, 60, 160 + pic.height + 80, W - 120, "PLUGS IN ANYWHERE", labels[k], "CPU only. Open licences.", GOLD, 1.0)
    return {"voice": "Plug it in after any video model. Seedance, LTX, Veo, or your own.", "min": 6.0, "draw": draw}


def scenes():
    return [
        scene_cold_open(),
        scene_text("SHOT GUARD", "A ComfyUI node that watches every take for you.",
                   "Dialogue · speaker · faces · blocking · signs · cuts · glitches · audio · facts · attention",
                   "Shot Guard is a ComfyUI node pack that watches every take for you.", min_s=3.2),
        *[scene_catch(k, *c) for k, c in enumerate(CATCHES)],
        scene_pair("side_swap", "side_swap-attempt1", ("ATTEMPT 0 · RETRY", "ATTEMPT 1 · PASS"), "THE FIX LOOP",
                   "Fix hint in, new seed, best take kept",
                   "Every problem becomes a fix hint. Shot Guard queues the graph again with the hint and a new seed, under a credit cap. "
                   "Here the generator is simulated."),
        scene_pair("hidden_cut", "hidden_cut-trimmed", ("BEFORE · RETRY", "AUTO-TRIM · PASS"), "AUTO-TRIM",
                   "No new generation. No credits.",
                   "And if the only problem is a hidden cut, auto trim keeps the clean part. No new generation, no credits.",
                   window_l=(2.5, None)),
        scene_best_of(),
        scene_attention(),
        scene_dailies(),
        scene_plug(),
        scene_text("BLIND TEST", "17 takes it had never seen", "2 sent back, both real hidden cuts. No good take sent back.",
                   "On seventeen takes it had never seen, it sent back two. Both were real cuts, and no good take was sent back.", min_s=5.0),
        scene_text("SHOT GUARD", "Every take, checked.", "ComfyUI node pack · Apache-2.0 · JK Automations",
                   "Shot Guard. Open source, and it runs on CPU.", min_s=3.5),
    ]


# ---- sound: voice lines (Kokoro, CPU), the takes' own sound under the voice, a tick per catch ----

def voice_lines(lines):
    os.makedirs(VOICE, exist_ok=True)
    todo = [(k, ln) for k, ln in enumerate(lines) if not os.path.exists(os.path.join(VOICE, f"{k:02d}.wav"))
            or open(os.path.join(VOICE, f"{k:02d}.txt")).read() != ln]
    if todo:
        job = os.path.join(VOICE, "job.json")
        json.dump(todo, open(job, "w"))
        code = ("import json,sys,numpy as np,soundfile as sf\nfrom kokoro import KPipeline\np=KPipeline(lang_code='a',device='cpu')\n"
                "for k,ln in json.load(open(sys.argv[1])):\n a=np.concatenate([x.audio.numpy() for x in p(ln,voice='af_heart',speed=1.05)])\n"
                f" sf.write('{VOICE}/%02d.wav'%k,a,24000); open('{VOICE}/%02d.txt'%k,'w').write(ln)\n")
        subprocess.run(["nice", "-n", "10", KOKORO, "-c", code, job], check=True, env=dict(os.environ, HF_HUB_OFFLINE="1"))
    out = []
    for k in range(len(lines)):
        with wave.open(os.path.join(VOICE, f"{k:02d}.wav")) as w:
            x = np.frombuffer(w.readframes(w.getnframes()), np.int16 if w.getsampwidth() == 2 else np.int32).astype(np.float32)
            x /= 32768.0 if w.getsampwidth() == 2 else 2 ** 31
            out.append(np.interp(np.arange(int(len(x) * SR / w.getframerate())) * w.getframerate() / SR, np.arange(len(x)), x))
    return out


def take_audio(path):
    r = subprocess.run([FFMPEG, "-nostdin", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"], capture_output=True)
    return np.frombuffer(r.stdout, np.float32)


def tick():
    t = np.arange(int(0.09 * SR)) / SR
    return 0.25 * np.sin(2 * np.pi * 1400 * t) * np.exp(-t * 45)


def caption_chunks(text, n=6):
    words = text.split()
    return [" ".join(words[k:k + n]) for k in range(0, len(words), n)]


def draw_caption(fr, text, W, H, band):
    d = ImageDraw.Draw(fr)
    f = font(40 if W > H else 46)
    tw = d.textlength(text, font=f)
    y = H - band + (band - 60) // 2
    d.rounded_rectangle([W / 2 - tw / 2 - 22, y - 8, W / 2 + tw / 2 + 22, y + 58], radius=12, fill=(16, 14, 12))
    d.text((W / 2 - tw / 2, y), text, font=f, fill=INK)


def build():
    sc = scenes()
    voices = voice_lines([s["voice"] for s in sc])
    for s, v in zip(sc, voices):
        s["dur"] = max(s["min"], 0.35 + len(v) / SR + 0.45)
    total = sum(s["dur"] for s in sc)
    mix, t0 = np.zeros(int((total + 1) * SR), np.float32), 0.0
    for s, v in zip(sc, voices):
        a = int(t0 * SR)
        if s.get("audio"):
            x = take_audio(s["audio"])[: int(s["dur"] * SR)]
            # the take's own sound: ducked under the narration, up in the gaps so the real lines are heard
            g = np.full(len(x), 0.5, np.float32)
            v0, v1 = int(0.2 * SR), int((0.5 + len(v) / SR) * SR)
            g[v0:v1] = 0.1
            g = np.convolve(g, np.ones(int(0.15 * SR)) / int(0.15 * SR), "same")
            mix[a:a + len(x)] += g[: len(x)] * x
        b = a + int(0.35 * SR)
        mix[b:b + len(v)] += 0.9 * v
        if s.get("tick") is not None:
            c = a + int(s["tick"] * SR)
            mix[c:c + len(tick())] += tick()
        s["t0"] = t0
        # caption timing: chunks share the voice time in proportion to their length
        chunks = caption_chunks(s["voice"])
        vs, ve = 0.35, 0.35 + len(v) / SR
        total_chars = sum(len(c) for c in chunks)
        s["caps"], c0 = [], vs
        for c in chunks:
            c1 = c0 + (ve - vs) * len(c) / total_chars
            s["caps"].append((c0, c1 + 0.12, c))
            c0 = c1
        t0 += s["dur"]
    mix = np.clip(mix / max(1.0, float(np.abs(mix).max()) / 0.95), -1, 1)
    os.makedirs(OUT, exist_ok=True)
    wav = os.path.join(OUT, "mix.wav")
    with wave.open(wav, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((mix * 32767).astype(np.int16).tobytes())
    for W, H, name in ((1920, 1080, "demo-16x9.mp4"), (1080, 1920, "demo-9x16.mp4")):
        path = os.path.join(OUT, name)
        p = subprocess.Popen([FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
                              "-i", "-", "-i", wav, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "19", "-c:a", "aac", "-b:a", "160k",
                              "-shortest", "-movflags", "+faststart", path], stdin=subprocess.PIPE)
        band = 130 if W > H else 230
        bg, pic_bg = Image.new("RGB", (W, H), BG), Image.new("RGB", (W, H - band), BG)
        for s in sc:
            n = int(round(s["dur"] * FPS))
            for f in range(n):
                t = f / FPS
                pic = pic_bg.copy()
                s["draw"](pic, t, W, H - band)
                fade = min(1.0, (f + 1) / 5, (n - f) / 5)
                if fade < 1.0:
                    pic = Image.blend(pic_bg, pic, fade)
                fr = bg.copy()
                fr.paste(pic, (0, 0))
                cap = next((c for a, b, c in s["caps"] if a <= t < b), None)
                if cap:
                    draw_caption(fr, cap, W, H, band)
                p.stdin.write(fr.tobytes())
        p.stdin.close()
        assert p.wait() == 0
        print(path, f"{total:.1f} s")
    json.dump([{"t0": round(s["t0"], 2), "dur": round(s["dur"], 2), "voice": s["voice"]} for s in sc],
              open(os.path.join(OUT, "edit.json"), "w"), indent=1)


if __name__ == "__main__":
    build()
