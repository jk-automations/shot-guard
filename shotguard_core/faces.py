"""Face tracks for the identity and blocking checks.
Detector: OpenCV YuNet. Recognizer: OpenCV SFace. Both from opencv_zoo (MIT / Apache-2.0), both CPU.
No InsightFace (its weights are non-commercial). DINOv2 was tried first and could not tell the cast apart
(docs/LESSONS.md), so identity uses SFace."""
import os
import urllib.request

import cv2
import numpy as np

from .video import read_frames

ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models/"
MODELS = {
    "yunet": ("face_detection_yunet_2023mar.onnx", ZOO + "face_detection_yunet/face_detection_yunet_2023mar.onnx"),
    "sface": ("face_recognition_sface_2021dec.onnx", ZOO + "face_recognition_sface/face_recognition_sface_2021dec.onnx"),
}
_cache = {}


def models_dir():
    d = os.environ.get("SHOTGUARD_MODELS")
    if not d:
        try:
            import folder_paths
            d = os.path.join(folder_paths.models_dir, "shot_guard")
        except ImportError:
            d = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".models")
    os.makedirs(d, exist_ok=True)
    return d


def model_path(key):
    name, url = MODELS[key]
    path = os.path.join(models_dir(), name)
    if not os.path.exists(path):
        print(f"[Shot Guard] downloading {name}")
        urllib.request.urlretrieve(url, path + ".part")
        os.replace(path + ".part", path)
    return path


def detector():
    if "yunet" not in _cache:
        _cache["yunet"] = cv2.FaceDetectorYN.create(model_path("yunet"), "", (320, 320), 0.7, 0.3, 5000)
    return _cache["yunet"]


def recognizer():
    if "sface" not in _cache:
        _cache["sface"] = cv2.FaceRecognizerSF.create(model_path("sface"), "")
    return _cache["sface"]


def detect(img, min_score=0.8, min_rel=0.035):
    """Faces: box (x, y, w, h) px, cx (0..1 across the frame), score, lm (5 landmarks), edge (cut off by the frame
    border), vec (SFace)."""
    h, w = img.shape[:2]
    det = detector()
    det.setInputSize((w, h))
    _, raw = det.detect(img)
    out = []
    for f in ([] if raw is None else raw):
        x, y, fw, fh, score = (float(v) for v in (f[0], f[1], f[2], f[3], f[14]))
        if score < min_score or fw < min_rel * w:
            continue
        v = recognizer().feature(recognizer().alignCrop(img, f)).flatten()
        out.append({"box": (x, y, fw, fh), "cx": (x + fw / 2) / w, "score": score,
                    "lm": [(float(f[4 + 2 * k]), float(f[5 + 2 * k])) for k in range(5)],   # eyes, nose tip, mouth corners
                    "edge": x < 0.01 * w or x + fw > 0.99 * w or y < 0.01 * h,
                    "vec": v / (np.linalg.norm(v) + 1e-9)})
    return out


def flatten_alpha(img):
    """Reference cut-outs (RGBA) go onto mid grey so the detector sees a normal photo."""
    if img.ndim == 3 and img.shape[2] == 4:
        a = img[:, :, 3:4].astype(np.float32) / 255.0
        img = (img[:, :, :3] * a + 128 * (1 - a)).astype(np.uint8)
    return img


def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    return flatten_alpha(img)


def gallery(images):
    """Every face in every reference image (a sheet with several views gives several vectors)."""
    vecs = []
    for img in images:
        img = load_image(img) if isinstance(img, str) else flatten_alpha(img)
        s = 1280 / max(img.shape[:2])
        if s < 1:
            img = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
        vecs += [f["vec"] for f in detect(img, min_score=0.7, min_rel=0.02)]
    return np.array(vecs).reshape(-1, 128)


def scan(path, every=0.25, max_side=960):
    """Faces on every sampled frame. Cached per file, so identity and blocking share one pass."""
    key = (os.path.abspath(path), os.path.getmtime(path), every, max_side)
    if key not in _cache:
        F = read_frames(path, every=every, max_side=max_side)
        frames = [{"t": t, "faces": detect(img), "scale": F.width / img.shape[1]} for t, img in zip(F.times, F.images)]
        _cache[key] = (F, frames)
    return _cache[key]


def build_tracks(frames, cuts=(), max_jump=0.12):
    """Link faces frame to frame by position. A cut always starts new tracks."""
    tracks, live = [], []
    cut_list = sorted(cuts)
    for fr in frames:
        if live and any(live[0]["pts"][-1]["t"] < c <= fr["t"] for c in cut_list):
            live = []
        used, nxt = set(), []
        for f in sorted(fr["faces"], key=lambda f: -f["box"][2]):
            best = None
            for tr in live:
                if id(tr) in used:
                    continue
                d = abs(tr["pts"][-1]["cx"] - f["cx"])
                if d < max_jump and (best is None or d < best[0]):
                    best = (d, tr)
            if best:
                tr = best[1]
            else:
                tr = {"pts": [], "segment": sum(c <= fr["t"] for c in cut_list)}
                tracks.append(tr)
            used.add(id(tr))
            tr["pts"].append(dict(f, t=fr["t"], scale=fr["scale"]))
            nxt.append(tr)
        live = nxt
    return tracks


def label_tracks(tracks, galleries, min_match=0.3, min_len=3):
    """Name the tracks, one segment (between cuts) at a time, one name per track (Hungarian on median similarity).
    galleries: {name: ref vectors}. Each confirmed track is added to that name's gallery, so later segments are
    also compared with how the character looked earlier in this same take."""
    from scipy.optimize import linear_sum_assignment
    names = [n for n in galleries if len(galleries[n])]
    self_gal = {n: [] for n in names}
    for seg in sorted({t["segment"] for t in tracks}):
        cand = [t for t in tracks if t["segment"] == seg and len(t["pts"]) >= min_len]
        if not cand or not names:
            continue
        S = np.zeros((len(cand), len(names)))
        for i, tr in enumerate(cand):
            V = np.array([p["vec"] for p in tr["pts"] if not p["edge"]] or [p["vec"] for p in tr["pts"]])
            for j, n in enumerate(names):
                G = galleries[n] if not self_gal[n] else np.vstack([galleries[n], np.array(self_gal[n])])
                S[i, j] = float(np.median((V @ G.T).max(axis=1)))
        # tracks that overlap in time cannot be the same person; split the segment into overlapping groups
        rows, cols = linear_sum_assignment(-S)
        for i, j in zip(rows, cols):
            if S[i, j] >= min_match:
                cand[i]["name"], cand[i]["match"] = names[j], S[i, j]
        for tr in cand:
            if tr.get("name"):
                V = np.array([p["vec"] for p in tr["pts"] if not p["edge"]] or [p["vec"] for p in tr["pts"]])
                m = V.mean(axis=0)
                self_gal[tr["name"]].append(m / (np.linalg.norm(m) + 1e-9))
    return tracks


def character_samples(tracks, name, gallery_vecs):
    """Time-ordered samples for one character: t, cx, similarity to the references, box, edge."""
    out = []
    for tr in tracks:
        if tr.get("name") != name:
            continue
        for p in tr["pts"]:
            out.append({"t": p["t"], "cx": p["cx"], "sim": float((gallery_vecs @ p["vec"]).max()), "edge": p["edge"],
                        "box": tuple(v * p["scale"] for v in p["box"])})
    return sorted(out, key=lambda s: s["t"])
