"""Shot Guard ComfyUI nodes. Plug in after any video node: Spec -> checks -> Gate -> Report, and Retry to re-queue
the same graph with a new seed and the fix hint. All checks run on CPU."""
import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shotguard_core as sg  # noqa: E402

try:
    from comfy_api.input_impl import VideoFromFile
except ImportError:  # older ComfyUI
    try:
        from comfy_api.latest._input_impl import VideoFromFile
    except ImportError:
        VideoFromFile = None

CAT = "Shot Guard"
SPEC, CHECK, RESULT = "SG_SPEC", "SG_CHECK", "SG_RESULT"


def _out_dir(*parts):
    try:
        import folder_paths
        root = folder_paths.get_output_directory()
    except ImportError:
        root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
    d = os.path.join(root, "shot_guard", *parts)
    os.makedirs(d, exist_ok=True)
    return d


def _to_bgr(t):
    """One ComfyUI IMAGE frame (H, W, 3 float 0..1, RGB) -> BGR uint8."""
    a = (t.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return a[:, :, ::-1].copy()


def take_path(video=None, image=None, video_path=""):
    """A file path for the take: a typed path, a ComfyUI VIDEO (file-backed or saved to temp), or the first IMAGE frame."""
    if video_path and video_path.strip():
        p = video_path.strip()
        if not os.path.isabs(p):
            try:
                import folder_paths
                for root in (folder_paths.get_input_directory(), folder_paths.get_output_directory()):
                    if os.path.exists(os.path.join(root, p)):
                        p = os.path.join(root, p)
                        break
            except ImportError:
                pass
        return p
    if video is not None:
        src = video.get_stream_source() if hasattr(video, "get_stream_source") else None
        if isinstance(src, str) and os.path.exists(src):
            return src
        p = tempfile.mktemp(prefix="shotguard-", suffix=".mp4", dir=_out_dir("tmp"))
        video.save_to(p)
        return p
    if image is not None:
        import cv2
        p = tempfile.mktemp(prefix="shotguard-", suffix=".png", dir=_out_dir("tmp"))
        cv2.imwrite(p, _to_bgr(image[0]))
        return p
    raise ValueError("Shot Guard: connect a video, an image, or type a video_path")


class ShotGuardSpec:
    """The contract for one take: who says what in which order, who stands where, which words may appear on screen,
    and how many retries / credits a fix may use."""
    CATEGORY = CAT
    RETURN_TYPES = (SPEC, "STRING")
    RETURN_NAMES = ("spec", "spec_json")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        side = (list(sg.spec.SIDES), {"default": "any"})
        req = {
            "take_name": ("STRING", {"default": "take"}),
            "script": ("STRING", {"multiline": True, "default": "",
                                  "tooltip": "One line per script line, in order: 'Speaker: words'. Empty = no dialogue check."}),
            "allowed_text": ("STRING", {"multiline": True, "default": "",
                                        "tooltip": "Words allowed on signs and screens, one per line. Empty = no readable text allowed."}),
            "max_retries": ("INT", {"default": 2, "min": 0, "max": 10}),
            "credit_cap": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10000.0, "step": 0.1,
                                     "tooltip": "Stop retrying once spent + one more take would pass this. 0 = no cap."}),
            "cost_per_take": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
            "attempt": ("INT", {"default": 0, "min": 0, "max": 100, "tooltip": "Set by Retry. Leave at 0."}),
        }
        for k in (1, 2, 3):
            req[f"character_{k}"] = ("STRING", {"default": ""})
            req[f"side_{k}"] = side
        opt = {f"reference_{k}": ("IMAGE",) for k in (1, 2, 3)}
        opt["locked_facts"] = ("STRING", {"multiline": True, "default": "",
                                          "tooltip": "Facts a vision judge checks (Continuity Lock), one per line, e.g. 'the van is plain white, no logos'. [minor] = low weight. Empty = Facts check skips."})
        opt["facts_reference"] = ("IMAGE", {"tooltip": "Optional ground-truth image for the locked facts (product, prop, set)."})
        opt["thresholds_json"] = ("STRING", {"multiline": True, "default": "", "tooltip": "Optional overrides, e.g. {\"line_accuracy\": 0.7}"})
        return {"required": req, "optional": opt}

    def run(self, take_name, script, allowed_text, max_retries, credit_cap, cost_per_take, attempt,
            character_1, side_1, character_2, side_2, character_3, side_3,
            reference_1=None, reference_2=None, reference_3=None, thresholds_json="", locked_facts="", facts_reference=None):
        chars = []
        for name, side, ref in ((character_1, side_1, reference_1), (character_2, side_2, reference_2), (character_3, side_3, reference_3)):
            if name.strip():
                chars.append({"name": name.strip(), "side": side, "refs": [] if ref is None else [_to_bgr(f) for f in ref]})
        sp = sg.spec.make(script=sg.spec.parse_script(script), characters=chars,
                          allowed_text=[w.strip() for w in allowed_text.replace(",", "\n").splitlines() if w.strip()],
                          thresholds=json.loads(thresholds_json) if thresholds_json.strip() else None,
                          max_retries=max_retries, credit_cap=credit_cap, cost_per_take=cost_per_take, attempt=attempt, name=take_name,
                          facts=locked_facts, fact_refs=[] if facts_reference is None else [_to_bgr(f) for f in facts_reference])
        return (sp, json.dumps(sg.spec.public(sp), indent=1))


class ShotGuardSpecFromPrompt:
    """The spec read from the generator's own prompt: every quoted line becomes a script line, its speaker is the cast
    member the prompt mentions just before it, quotes after "sign" or "reads" become allowed text, and "on the left /
    right" sets sides. Connect the same prompt the generator gets, so the check always matches what was asked for."""
    CATEGORY = CAT
    RETURN_TYPES = (SPEC, "STRING", "STRING")
    RETURN_NAMES = ("spec", "spec_json", "script")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        req = {
            "prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "The generator's prompt (link it from the same text the generator gets)."}),
            "cast": ("STRING", {"multiline": True, "default": "",
                                "tooltip": "One per line: 'Name: words the prompt uses for them', e.g. 'Rosa: baker, yellow headscarf'. Reference k goes with line k."}),
            "take_name": ("STRING", {"default": "take"}),
            "max_retries": ("INT", {"default": 2, "min": 0, "max": 10}),
            "credit_cap": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10000.0, "step": 0.1}),
            "cost_per_take": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
            "attempt": ("INT", {"default": 0, "min": 0, "max": 100, "tooltip": "Set by Retry. Leave at 0."}),
        }
        opt = {f"reference_{k}": ("IMAGE",) for k in (1, 2, 3)}
        opt["allowed_text"] = ("STRING", {"multiline": True, "default": "", "tooltip": "Extra words allowed on screen, one per line."})
        opt["thresholds_json"] = ("STRING", {"multiline": True, "default": ""})
        return {"required": req, "optional": opt}

    def run(self, prompt, cast, take_name, max_retries, credit_cap, cost_per_take, attempt,
            reference_1=None, reference_2=None, reference_3=None, allowed_text="", thresholds_json=""):
        names = list(sg.prompt_spec.parse_cast(cast))
        refs = {n: [_to_bgr(f) for f in r] for n, r in zip(names, (reference_1, reference_2, reference_3)) if r is not None}
        sp = sg.prompt_spec.make_spec(prompt, cast, refs, name=take_name, max_retries=max_retries, credit_cap=credit_cap,
                                      cost_per_take=cost_per_take, attempt=attempt,
                                      allowed_text=[w.strip() for w in allowed_text.replace(",", "\n").splitlines() if w.strip()],
                                      thresholds=json.loads(thresholds_json) if thresholds_json.strip() else None)
        script = "\n".join(f"{L['speaker'] or '?'}: {L['text']}" for L in sp["script"])
        return (sp, json.dumps(sg.spec.public(sp), indent=1), script)


class _Check:
    CATEGORY = CAT
    RETURN_TYPES = (CHECK, "STRING")
    RETURN_NAMES = ("check", "summary")
    FUNCTION = "run"
    KEY = ""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"spec": (SPEC,)},
                "optional": {"video": ("VIDEO",), "image": ("IMAGE",), "video_path": ("STRING", {"default": ""})}}

    def run(self, spec, video=None, image=None, video_path=""):
        path = take_path(video, image, video_path)
        r = sg.CHECKS[self.KEY](path, spec)
        r["video"] = path
        lines = [f"{self.KEY}: {r['status'].upper()}"] + [f"- {i['message']}" for i in r["issues"]]
        return (r, "\n".join(lines))


class ShotGuardDialogue(_Check):
    """Transcribes the take (whisper.cpp, CPU) and reports missing, extra, merged and out-of-order lines."""
    KEY = "dialogue"


class ShotGuardIdentity(_Check):
    """Per-character likeness to the reference images over time (YuNet + SFace): min / mean curve, worst frame."""
    KEY = "identity"


class ShotGuardBlocking(_Check):
    """Each character's position over time: flags side swaps and characters leaving the frame."""
    KEY = "blocking"


class ShotGuardText(_Check):
    """Reads signs and screens (Florence-2 OCR, CPU) and flags words that are not on the allowed list."""
    KEY = "text"


class ShotGuardCuts(_Check):
    """Finds scene cuts and angle jumps inside what should be one continuous take."""
    KEY = "cuts"


class ShotGuardAudio(_Check):
    """Clipping, a silent or very quiet track, dead air, music as loud as the voice. No model."""
    KEY = "audio"


class ShotGuardSpeaker(_Check):
    """Is the right person saying the line? Compares whose mouth moves while each line is heard with the script's
    speaker (YuNet mouth landmarks + SFace names + whisper.cpp timing, CPU). Needs 2+ characters with references."""
    KEY = "speaker"


class ShotGuardAttention(_Check):
    """Will people keep watching? Hook score for the first seconds, an attention curve, dead stretches. The open,
    rule-based take on a virality predictor (movement, faces, speech, sound; no trained model). Warnings only."""
    KEY = "attention"


class ShotGuardGlitch(_Check):
    """Playback faults: a frozen picture in a moving shot, stutter (repeated frames), brightness flicker. No model."""
    KEY = "glitch"


class ShotGuardFacts(_Check):
    """Locked facts (props, wardrobe, colours, no logos) judged on a few frames by a vision LLM: Continuity Lock's judge.
    Needs an OpenAI-compatible endpoint that takes images (SHOTGUARD_JUDGE_URL / _MODEL)."""
    KEY = "facts"


class ShotGuardSequence:
    """Takes of one scene in edit order: each character keeps the same side of the frame across every cut."""
    CATEGORY = CAT
    RETURN_TYPES = (CHECK, "STRING")
    RETURN_NAMES = ("check", "summary")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"spec": (SPEC,), "take_1": ("VIDEO",), "take_2": ("VIDEO",)},
                "optional": {"take_3": ("VIDEO",), "take_4": ("VIDEO",)}}

    def run(self, spec, take_1, take_2, take_3=None, take_4=None):
        paths = [take_path(v) for v in (take_1, take_2, take_3, take_4) if v is not None]
        r = sg.sequence.check(paths, spec)
        r["video"] = paths[-1]
        return (r, "\n".join([f"sequence: {r['status'].upper()}"] + [f"- {i['message']}" for i in r["issues"]]))


class ShotGuardGate:
    """PASS / RETRY / FAIL plus plain-English fix hints, ready to add to the generator prompt."""
    CATEGORY = CAT
    RETURN_TYPES = (RESULT, "STRING", "STRING", "STRING", "BOOLEAN")
    RETURN_NAMES = ("result", "verdict", "fix_hint", "summary", "passed")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"spec": (SPEC,)},
                "optional": {k: (CHECK,) for k in ("dialogue", "identity", "blocking", "text", "cuts", "facts", "sequence", "audio",
                                                   "speaker", "glitch", "attention")}}

    def run(self, spec, dialogue=None, identity=None, blocking=None, text=None, cuts=None, facts=None, sequence=None, audio=None,
            speaker=None, glitch=None, attention=None):
        checks = [c for c in (dialogue, speaker, identity, blocking, text, cuts, glitch, audio, facts, sequence, attention) if c]
        if not checks:
            raise ValueError("Shot Guard Gate: connect at least one check")
        r = sg.gate.decide(spec, checks, video_path=checks[0].get("video"))
        summary = "\n".join([f"{r['verdict']} (score {r['score']}): {r['why']}"] + [f"- {i['check']}: {i['message']}" for i in r["issues"]])
        return (r, r["verdict"], r["prompt_hint"], summary, r["passed"])


class ShotGuardReport:
    """Contact sheet (verdict, timeline, one card per flag with its timecode, fix hints) + JSON receipt, a clickable
    HTML report (the take plays; click a flag to jump to it) and a 1080x1350 share card."""
    CATEGORY = CAT
    RETURN_TYPES = ("IMAGE", "STRING", "IMAGE", "STRING")
    RETURN_NAMES = ("contact_sheet", "receipt_json", "share_card", "html_report")
    FUNCTION = "run"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"result": (RESULT,), "filename_prefix": ("STRING", {"default": "shotguard/report"}),
                             "blur_faces": ("BOOLEAN", {"default": False, "tooltip": "Blur every face on the sheet (for sharing)."})}}

    def run(self, result, filename_prefix, blur_faces):
        import torch
        name = f"{os.path.basename(filename_prefix)}_{result['spec'].get('name', 'take')}_a{result['attempt']}"
        sub = os.path.dirname(filename_prefix)
        out = _out_dir(sub) if sub else _out_dir()
        img, png, js = sg.report.render(result, out, name, blur=blur_faces)
        card = sg.card.render(result, os.path.join(out, name + "_card.png"), blur=blur_faces)
        page = sg.html_report.render(result, os.path.join(out, name + "_html"), name)
        t = torch.from_numpy(np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0)[None]
        tc = torch.from_numpy(np.asarray(card.convert("RGB"), dtype=np.float32) / 255.0)[None]
        try:
            import folder_paths
            rel = os.path.relpath(out, folder_paths.get_output_directory())
            ui = {"images": [{"filename": os.path.basename(png), "subfolder": rel, "type": "output"},
                             {"filename": name + "_card.png", "subfolder": rel, "type": "output"}]}
        except ImportError:
            ui = {}
        return {"ui": ui, "result": (t, open(js).read(), tc, page)}


def _video_ui(path):
    try:
        import folder_paths
        root = folder_paths.get_output_directory()
        return {"images": [{"filename": os.path.basename(path), "subfolder": os.path.relpath(os.path.dirname(path), root),
                            "type": "output"}], "animated": (True,)}
    except ImportError:
        return {}


def _video(path):
    return VideoFromFile(path) if VideoFromFile else path


class ShotGuardAnnotate:
    """The take as a video with the findings drawn on at the second they happen: playhead timeline, check chips that
    light up, boxes, cut flashes, and the verdict at the end. Original sound kept. Share it instead of a bug list."""
    CATEGORY = CAT
    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("annotated_video", "path")
    FUNCTION = "run"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"result": (RESULT,), "filename_prefix": ("STRING", {"default": "shotguard/annotated"}),
                             "blur_faces": ("BOOLEAN", {"default": False, "tooltip": "Blur every face (for sharing)."})}}

    def run(self, result, filename_prefix, blur_faces):
        sub = os.path.dirname(filename_prefix)
        out = os.path.join(_out_dir(sub) if sub else _out_dir(),
                           f"{os.path.basename(filename_prefix)}_{result['spec'].get('name', 'take')}_a{result['attempt']}.mp4")
        sg.annotate.render(result, out, blur=blur_faces)
        return {"ui": _video_ui(out), "result": (_video(out), out)}


class ShotGuardTrim:
    """Auto-trim: if a take's problem is a hidden cut, keep the longest clean piece instead of paying for a retry.
    fixed = True when the piece has no major problem left; saved = the take cost that retry would have spent."""
    CATEGORY = CAT
    RETURN_TYPES = ("VIDEO", "STRING", "BOOLEAN", "STRING", "FLOAT")
    RETURN_NAMES = ("video", "path", "fixed", "note", "saved")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"result": (RESULT,), "filename_prefix": ("STRING", {"default": "shotguard/trimmed"})}}

    def run(self, result, filename_prefix):
        path = result["video"]
        p = sg.trim.plan(result, result["spec"]["thresholds"]["trim_min_s"])
        if not p or not p["ok"] and p.get("seconds") is None:
            return (_video(path), path, result["passed"], p["note"] if p else "no hidden cut to trim", 0.0)
        sub = os.path.dirname(filename_prefix)
        out = os.path.join(_out_dir(sub) if sub else _out_dir(),
                           f"{os.path.basename(filename_prefix)}_{result['spec'].get('name', 'take')}_a{result['attempt']}.mp4")
        sg.trim.cut(path, p["t0"], p["t1"], out)
        again = sg.trim.recheck(out, result["spec"])
        fixed = p["ok"] and again["status"] != "fail"
        note = p["note"] + ("" if again["status"] != "fail" else "; the piece still has: " + "; ".join(i["message"] for i in again["issues"]))
        return (_video(out), out, fixed, note, result["spec"]["cost_per_take"] if fixed else 0.0)


class ShotGuardBestOf:
    """Best of N: connect the Gate results of up to 4 takes of the same shot; outputs the best take and the ranking."""
    CATEGORY = CAT
    RETURN_TYPES = (RESULT, "VIDEO", "STRING", "INT")
    RETURN_NAMES = ("best_result", "best_video", "ranking", "best_index")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"result_1": (RESULT,), "result_2": (RESULT,)},
                "optional": {"result_3": (RESULT,), "result_4": (RESULT,)}}

    def run(self, result_1, result_2, result_3=None, result_4=None):
        rs = [r for r in (result_1, result_2, result_3, result_4) if r is not None]
        ranked, order = sg.best.rank(rs)
        return (ranked[0], _video(ranked[0]["video"]), sg.best.summary(rs), order[0] + 1)


class ShotGuardDailies:
    """Dailies: checks every take in a folder (each with its own <name>.json spec, or this one), auto-trims hidden
    cuts, ranks the takes best first and draws one page with the credits meter: takes checked, caught, retries saved."""
    CATEGORY = CAT
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("page", "summary", "receipt_json")
    FUNCTION = "run"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"spec": (SPEC,), "folder": ("STRING", {"default": "", "tooltip": "A folder of takes (input/ or absolute)."}),
                             "auto_trim": ("BOOLEAN", {"default": True}),
                             "filename_prefix": ("STRING", {"default": "shotguard/dailies"}),
                             "blur_faces": ("BOOLEAN", {"default": False})}}

    def run(self, spec, folder, auto_trim, filename_prefix, blur_faces):
        import torch
        folder = take_path(video_path=folder)
        sub = os.path.dirname(filename_prefix)
        out = _out_dir(sub) if sub else _out_dir()
        d = sg.batch.run_folder(folder, spec, out_dir=out, auto_trim=auto_trim)
        img, png, js = sg.batch.render(d, out, os.path.basename(filename_prefix), blur=blur_faces)
        sg.html_report.index(d, os.path.join(out, os.path.basename(filename_prefix) + "_html"))
        t = torch.from_numpy(np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0)[None]
        try:
            import folder_paths
            ui = {"images": [{"filename": os.path.basename(png), "subfolder": os.path.relpath(out, folder_paths.get_output_directory()),
                              "type": "output"}]}
        except ImportError:
            ui = {}
        return {"ui": ui, "result": (t, sg.batch.summary(d), open(js).read())}


class ShotGuardRetry:
    """On RETRY: queues this same graph again through the ComfyUI /prompt API with a new seed and the fix hint added
    to the prompt. Stops at max retries or the credit cap and keeps the best take."""
    CATEGORY = CAT
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("status", "best_take_path", "ledger_path")
    FUNCTION = "run"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"result": (RESULT,), "enabled": ("BOOLEAN", {"default": True}),
                             "seed_target": ("STRING", {"default": "", "tooltip": "node_id.input, e.g. 3.seed. Empty = first seed in the graph."}),
                             "prompt_target": ("STRING", {"default": "", "tooltip": "node_id.input, e.g. 6.text. Empty = longest prompt text."}),
                             "server_url": ("STRING", {"default": "", "tooltip": "Empty = this ComfyUI."}),
                             "run_id": ("STRING", {"default": "", "tooltip": "Set by Retry. Leave empty."})},
                "optional": {"video": ("VIDEO",)},
                "hidden": {"prompt": "PROMPT", "unique_id": "UNIQUE_ID", "api_key_comfy_org": "API_KEY_COMFY_ORG",
                           "auth_token_comfy_org": "AUTH_TOKEN_COMFY_ORG"}}

    @classmethod
    def IS_CHANGED(cls, **kw):   # always run; a string, because NaN would end up in /history and break its JSON
        import uuid
        return uuid.uuid4().hex

    def run(self, result, enabled, seed_target, prompt_target, server_url, run_id, video=None, prompt=None, unique_id=None,
            api_key_comfy_org=None, auth_token_comfy_org=None):
        server = server_url.strip() or _local_server()
        # pass this run's partner-node credentials on to the next attempt (paid partner generators need them)
        extra = {k: v for k, v in (("api_key_comfy_org", api_key_comfy_org), ("auth_token_comfy_org", auth_token_comfy_org)) if v}
        vp = take_path(video) if video is not None else result.get("video")
        r = sg.retry.step(result, graph=prompt, video_path=vp, run_id=run_id, ledger_root=_out_dir("runs"),
                          post=lambda g: sg.retry.post_prompt(server, g, extra_data=extra), retry_id=str(unique_id) if unique_id else None,
                          seed_target=seed_target, prompt_target=prompt_target, enabled=enabled)
        return {"ui": {"text": [r["status"]]}, "result": (r["status"], r.get("best_path") or "", r["ledger"])}


class ShotGuardMockGenerator:
    """Demo stand-in for any video generator: the seed picks one of the listed takes (files in input/). Nothing is
    generated and nothing is paid for. Lets the Retry loop run end to end through the real /prompt API."""
    CATEGORY = CAT + "/demo"
    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("video", "prompt_used")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"seed": ("INT", {"default": 0, "min": 0, "max": 2 ** 63 - 1}),
                             "prompt": ("STRING", {"multiline": True, "default": ""}),
                             "takes": ("STRING", {"multiline": True, "default": "", "tooltip": "One file per line (input/ or absolute)."})}}

    def run(self, seed, prompt, takes):
        files = [t.strip() for t in takes.splitlines() if t.strip()]
        if not files:
            raise ValueError("Mock generator: list at least one take")
        path = take_path(video_path=files[seed % len(files)])
        print(f"[Shot Guard mock generator] seed {seed} -> {os.path.basename(path)}; prompt: {prompt[-160:]}")
        return (VideoFromFile(path), prompt)


def _local_server():
    try:
        from server import PromptServer
        port = getattr(PromptServer.instance, "port", None)
    except Exception:
        port = None
    if not port:
        try:
            from comfy.cli_args import args
            port = args.port
        except Exception:
            port = 8188
    return f"http://127.0.0.1:{port}"


NODE_CLASS_MAPPINGS = {
    "ShotGuardSpec": ShotGuardSpec, "ShotGuardSpecFromPrompt": ShotGuardSpecFromPrompt, "ShotGuardDialogue": ShotGuardDialogue, "ShotGuardIdentity": ShotGuardIdentity,
    "ShotGuardBlocking": ShotGuardBlocking, "ShotGuardText": ShotGuardText, "ShotGuardCuts": ShotGuardCuts, "ShotGuardFacts": ShotGuardFacts,
    "ShotGuardSequence": ShotGuardSequence, "ShotGuardAudio": ShotGuardAudio, "ShotGuardSpeaker": ShotGuardSpeaker,
    "ShotGuardGlitch": ShotGuardGlitch, "ShotGuardDailies": ShotGuardDailies, "ShotGuardAttention": ShotGuardAttention,
    "ShotGuardAnnotate": ShotGuardAnnotate, "ShotGuardTrim": ShotGuardTrim, "ShotGuardBestOf": ShotGuardBestOf,
    "ShotGuardGate": ShotGuardGate, "ShotGuardRetry": ShotGuardRetry, "ShotGuardReport": ShotGuardReport,
    "ShotGuardMockGenerator": ShotGuardMockGenerator,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ShotGuardSpec": "ShotGuard Spec", "ShotGuardSpecFromPrompt": "ShotGuard Spec from Prompt", "ShotGuardDialogue": "ShotGuard Check: Dialogue",
    "ShotGuardIdentity": "ShotGuard Check: Identity drift", "ShotGuardBlocking": "ShotGuard Check: Blocking",
    "ShotGuardText": "ShotGuard Check: Text", "ShotGuardCuts": "ShotGuard Check: Hidden cuts",
    "ShotGuardFacts": "ShotGuard Check: Facts (vision judge)",
    "ShotGuardSequence": "ShotGuard Check: Sequence (across cuts)",
    "ShotGuardAudio": "ShotGuard Check: Audio", "ShotGuardAnnotate": "ShotGuard Annotated Video",
    "ShotGuardSpeaker": "ShotGuard Check: Speaker", "ShotGuardGlitch": "ShotGuard Check: Glitch",
    "ShotGuardDailies": "ShotGuard Dailies (folder)", "ShotGuardAttention": "ShotGuard Check: Attention (hook)",
    "ShotGuardTrim": "ShotGuard Auto-trim", "ShotGuardBestOf": "ShotGuard Best of N",
    "ShotGuardGate": "ShotGuard Gate", "ShotGuardRetry": "ShotGuard Retry", "ShotGuardReport": "ShotGuard Report",
    "ShotGuardMockGenerator": "ShotGuard Mock Generator (demo)",
}
