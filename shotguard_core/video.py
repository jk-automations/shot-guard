"""Frame and audio access for one take. Uses OpenCV for frames and ffmpeg (on PATH) for audio."""
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass

import cv2
import numpy as np


def ffmpeg_bin():
    return shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"


@dataclass
class Frames:
    times: list        # seconds
    images: list       # BGR uint8 arrays
    fps: float
    duration: float
    width: int
    height: int


def is_image(path):
    return os.path.splitext(path)[1].lower() in (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def read_frames(path, every=0.25, max_side=None):
    """Sample one frame every `every` seconds (0 = every frame). A still image is a one-frame take."""
    if is_image(path):
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(path)
        h, w = img.shape[:2]
        return Frames([0.0], [_shrink(img, max_side)], 0.0, 0.0, w, h)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    step = max(1, int(round(every * fps))) if every else 1
    times, images, i = [], [], 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if i % step == 0:
            times.append(i / fps)
            images.append(_shrink(img, max_side))
        i += 1
    cap.release()
    return Frames(times, images, fps, (n or i) / fps, w, h)


def _shrink(img, max_side):
    if not max_side:
        return img
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else img


def frame_at(path, t):
    if is_image(path):
        return cv2.imread(path)
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
    ok, img = cap.read()
    cap.release()
    return img if ok else None


def extract_wav(path, out=None):
    """16 kHz mono wav for speech recognition. Returns None when the take has no audio."""
    out = out or tempfile.mktemp(suffix=".wav")
    r = subprocess.run([ffmpeg_bin(), "-nostdin", "-v", "error", "-y", "-i", path, "-vn", "-ar", "16000", "-ac", "1", out],
                       capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(out) or os.path.getsize(out) < 1000:
        return None
    return out


def has_audio(path):
    if is_image(path):
        return False
    r = subprocess.run([ffmpeg_bin(), "-nostdin", "-v", "error", "-i", path, "-map", "0:a:0", "-t", "0.1", "-f", "null", "-"],
                       capture_output=True, text=True)
    return r.returncode == 0


def hist(img):
    hsv = cv2.cvtColor(cv2.resize(img, (160, 284)), cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1, 2], None, [16, 8, 8], [0, 180, 0, 256, 0, 256])
    return cv2.normalize(h, h).flatten()


def gray_small(img):
    return cv2.cvtColor(cv2.resize(img, (90, 160)), cv2.COLOR_BGR2GRAY).astype(np.float32)
