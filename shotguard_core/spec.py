"""The take contract: script lines, characters (references + screen side), allowed on-screen text, thresholds,
retry limits. Plain dicts so the same spec works in ComfyUI, in tests and as a JSON file."""
import copy
import json
import os

DEFAULTS = {
    "sample_every": 0.25,      # seconds between face samples
    "cut_hard": 1.45,          # frame-to-frame cut score: real cuts 1.56-10.6; occluders, flour, pop-ins up to 1.32
    "cut_window": 4.5,         # 6-frame window score for whip / morph jumps: real 9.3; fast camera moves up to 4.1
    "cut_window_frames": 6,
    "id_min_match": 0.3,       # SFace similarity needed to name a face track
    "id_drift_drop": 0.12,     # drift = likeness this far below the take's own start ...
    "id_min": 0.25,            # ... or below this absolute floor ...
    "id_min_dur": 1.0,         # ... for at least this long (seconds). Reported as a warning:
    "id_major_drop": 0.25,     # only a drop this big fails the take (head turns and laughs drop 0.15 too)
    "side_margin": 0.08,       # "left" means the face centre stays left of 0.5 + margin
    "side_min_dur": 0.5,
    "leave_min_dur": 1.0,
    "line_accuracy": 0.8,      # share of a line's words that must be heard
    "ocr_every": 2.0,          # seconds between OCR frames (1.0 = slower, catches briefer text)
    "ocr_min_hits": 2,         # a bad word must be read on 2+ frames of a video (one for a still)
    "allow_start_text": True,
    "min_text_height": 0.03,   # ignore text shorter than 3% of the frame (background chalkboards, menus); 0 = check all, for close-up ads
    "audio_silent_db": -50.0,  # a scripted take quieter than this overall has no usable sound (fails)
    "audio_quiet_db": -35.0,   # quieter than this overall: warning (real takes: -27 to -17 dBFS)
    "audio_clip_share": 0.01,  # share of samples above 95% of full scale that warns of distortion (real takes: at most 0.0003)
    "audio_clip_major": 0.05,  # ... and that fails the take
    "audio_gap_db": -45.0,     # dead air: quieter than this ...
    "audio_gap_s": 2.0,        # ... for this long, mid-take, in a take with dialogue (real takes: at most 0.8 s)
    "audio_bed_db": 8.0,       # loud vs quiet parts closer than this: the music may cover the voice (real: 9.8 to 31.5)
    "glitch_still_shot": 0.3,  # a shot that changes less than this per frame (median) is calm, not frozen
    "glitch_frozen_diff": 0.35,  # frozen = change under this (and under a quarter of the shot's median) ...
    "glitch_freeze_s": 0.5,    # ... for this long: warning (real takes: at most 0.46 s, in a calm shot)
    "glitch_freeze_major_s": 1.0,   # ... and this long fails the take
    "glitch_stutter_motion": 1.0,   # a repeated frame between two steps bigger than this ...
    "glitch_stutter_events": 3,     # ... this many times is a stutter (real takes: none)
    "glitch_flicker_luma": 2.0,     # brightness above/below its local median by this much on alternate frames ...
    "glitch_flicker_events": 4,     # ... this many times is flicker (real takes: none)
    "speaker_min_s": 0.8,      # lines shorter than this are too short to judge whose mouth moves
    "speaker_min_seen": 0.5,   # the scripted speaker's face must be measurable for half the line
    "speaker_ratio": 1.6,      # someone else's mouth moving this much more: warning (real takes: head turns up to 1.4x) ...
    "speaker_ratio_major": 2.0,   # ... this much more: fails the take
    "attention_motion_full": 3.0,   # frame change that counts as full movement
    "attention_hook_s": 2.0,  # the hook: the first seconds of the take
    "attention_hook_min": 0.35,   # hook attention below this: slow start (warning)
    "attention_dead": 0.2,    # attention below this ...
    "attention_dead_s": 1.5,  # ... for this long after the hook: dead stretch (warning)
    "trim_min_s": 2.0,         # Auto-trim keeps a piece only if it is at least this long
    "whisper_model": None,
    "facts_frames": 4,         # frames the vision judge looks at
    "facts_min_hits": 2,       # a fact must be broken on 2+ of them (one for a still)
    "judge_url": None,         # None = SHOTGUARD_JUDGE_URL or the fleet router
    "judge_model": None,
}
SIDES = ("any", "left", "right", "center")


def make(script=None, characters=None, allowed_text=None, thresholds=None, max_retries=2, credit_cap=0.0,
         cost_per_take=0.0, attempt=0, name="take", facts=None, fact_refs=None):
    th = dict(DEFAULTS)
    th.update({k: v for k, v in (thresholds or {}).items() if k in DEFAULTS})
    chars = []
    for c in characters or []:
        side = c.get("side", "any")
        if side not in SIDES:
            raise ValueError(f"side must be one of {SIDES}, got {side!r}")
        chars.append({"name": c["name"], "refs": list(c.get("refs", [])), "side": side,
                      "stay_in_frame": bool(c.get("stay_in_frame", True))})
    return {"name": name, "script": [{"speaker": s.get("speaker", ""), "text": s["text"]} for s in script or []],
            "characters": chars, "allowed_text": list(allowed_text or []), "thresholds": th,
            "facts": [f for f in (facts.splitlines() if isinstance(facts, str) else facts or []) if str(f).strip()],
            "fact_refs": list(fact_refs or []),
            "max_retries": int(max_retries), "credit_cap": float(credit_cap), "cost_per_take": float(cost_per_take),
            "attempt": int(attempt)}


def parse_script(text):
    """One line per script line: "Speaker: words". Lines without a colon have no speaker."""
    out = []
    for raw in (text or "").splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        spk, sep, rest = raw.partition(":")
        out.append({"speaker": spk.strip(), "text": rest.strip()} if sep and len(spk) <= 40 else {"speaker": "", "text": raw})
    return out


def load(path_or_json, base_dir=None):
    """A spec from a JSON file or string. Relative reference paths resolve against the file's folder."""
    if os.path.exists(str(path_or_json)):
        base_dir = base_dir or os.path.dirname(os.path.abspath(path_or_json))
        with open(path_or_json) as f:
            d = json.load(f)
    else:
        d = json.loads(path_or_json)
    for c in d.get("characters", []):
        c["refs"] = [r if os.path.isabs(r) or not base_dir else os.path.join(base_dir, r) for r in c.get("refs", [])]
    return make(**{k: d[k] for k in ("script", "characters", "allowed_text", "thresholds", "max_retries", "credit_cap",
                                      "cost_per_take", "attempt", "name", "facts") if k in d})


def public(spec):
    """The spec as written into receipts: no cache, reference file names only."""
    s = copy.deepcopy({k: v for k, v in spec.items() if not k.startswith("_")})
    for c in s["characters"]:
        c["refs"] = [os.path.basename(r) if isinstance(r, str) else "<image>" for r in c["refs"]]
    s["fact_refs"] = [os.path.basename(r) if isinstance(r, str) else "<image>" for r in s.get("fact_refs", [])]
    return s
