"""Vision-LLM fact judge, vendored from JK Automations' Continuity Lock node pack
(open-adfield, custom_nodes/comfyui-continuity-lock/continuity_core.py, commit 6d492aa).
Changes for Shot Guard: default endpoint (fleet router, batch priority), report grid removed.

Original licence:

MIT License

Copyright (c) 2026 JK Automations

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SEVERITY_WEIGHT = {"major": 1.0, "minor": 0.4}
PASS_SCORE = 0.85


def llm_defaults() -> dict:
    # Shot Guard: default to the JK fleet router (any OpenAI-compatible endpoint that takes images works);
    # SHOTGUARD_JUDGE_* first, then Continuity Lock's own ADFIELD_* variables.
    env = os.environ.get
    return {
        "url": env("SHOTGUARD_JUDGE_URL") or env("ADFIELD_JUDGE_URL") or env("ADFIELD_LLM_URL") or "http://127.0.0.1:7700/v1",
        "model": env("SHOTGUARD_JUDGE_MODEL") or env("ADFIELD_JUDGE_MODEL") or env("ADFIELD_LLM_MODEL") or "local-big",
        "api_key": env("SHOTGUARD_JUDGE_KEY") or env("ADFIELD_JUDGE_KEY") or env("ADFIELD_LLM_KEY") or "",
    }


# ---------------------------------------------------------------- bible

@dataclass
class Fact:
    text: str
    severity: str = "major"


def parse_bible(text: str | list) -> list[Fact]:
    """One fact per line, or simple YAML (`- fact`, `key: value`, `# comment`).

    Prefix a line with `[minor]` (or `~`) for a low-weight fact. Lists pass through.
    A bare YAML section header such as `car:` is ignored; `car: red coupe` becomes "car: red coupe".
    """
    lines = text.splitlines() if isinstance(text, str) else list(text)
    facts: list[Fact] = []
    for raw in lines:
        if isinstance(raw, Fact):
            facts.append(raw)
            continue
        s = str(raw).strip()
        if not s or s.startswith("#") or s in ("---", "..."):
            continue
        s = re.sub(r"^[-*•]\s+", "", s)
        if s.endswith(":") and " " not in s.rstrip(":"):
            continue  # YAML section header
        sev = "major"
        m = re.match(r"^\[(minor|major)\]\s*(.*)$", s, re.I)
        if m:
            sev, s = m.group(1).lower(), m.group(2)
        elif s.startswith("~"):
            sev, s = "minor", s[1:].strip()
        s = s.strip().strip('"').strip("'")
        if s:
            facts.append(Fact(s, sev))
    return facts


@dataclass
class Bible:
    facts: list[Fact]
    refs: list = field(default_factory=list)  # PIL images

    @classmethod
    def make(cls, facts, refs=None) -> "Bible":
        return cls(parse_bible(facts) if not (isinstance(facts, list) and facts and isinstance(facts[0], Fact)) else facts,
                   [to_pil(r) for r in (refs or [])])


# ---------------------------------------------------------------- images

def to_pil(img) -> Image.Image:
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    return Image.open(str(img)).convert("RGB")


def _b64(img: Image.Image, max_side: int = 768) -> str:
    img = img.copy()
    img.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def image_hash(img: Image.Image) -> str:
    return hashlib.sha1(img.resize((64, 64)).tobytes()).hexdigest()[:16]


# ---------------------------------------------------------------- judge

SYSTEM = ("You are a strict continuity supervisor for AI-generated images. You compare an image against a bible of "
          "locked facts. Be literal and objective. Judge only what is visible. Reply with JSON only, no prose.")


def _build_messages(img, facts: list[Fact], refs: list, compare_to=None) -> list:
    content = []
    n = 0
    for r in refs:
        n += 1
        content += [{"type": "text", "text": f"Reference image {n} (ground truth for the locked facts):"},
                    {"type": "image_url", "image_url": {"url": _b64(r)}}]
    if compare_to is not None:
        content += [{"type": "text", "text": "Previously approved frame of the same sequence (must match it):"},
                    {"type": "image_url", "image_url": {"url": _b64(compare_to)}}]
    flist = "\n".join(f"{i}. {f.text}" for i, f in enumerate(facts, 1))
    content += [
        {"type": "text", "text": "FRAME TO CHECK:"},
        {"type": "image_url", "image_url": {"url": _b64(img)}},
        {"type": "text", "text": (
            f"LOCKED FACTS:\n{flist}\n\n"
            "For every fact: write what you actually observe in the FRAME TO CHECK (max 12 words), then a status.\n"
            'status "ok" = frame consistent with the fact; "violated" = frame clearly contradicts it; '
            '"unclear" = cannot be seen.\n'
            'If violated, "fix" is a short imperative edit that corrects ONLY that fact (e.g. "Change the car paint to glossy red"). '
            'Otherwise fix is "".\n'
            'Return ONLY: {"facts":[{"id":1,"observed":"...","status":"ok|violated|unclear","fix":"..."}]}')},
    ]
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]


def chat(messages, *, url=None, model=None, api_key=None, max_tokens=900, timeout=600) -> str:
    d = llm_defaults()
    url = (url or d["url"]).rstrip("/")
    body = {"model": model or d["model"], "temperature": 0, "max_tokens": max_tokens, "messages": messages}
    headers = {"Content-Type": "application/json", "X-Priority": "batch"}
    key = api_key if api_key is not None and api_key != "" else d["api_key"]
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(url + "/chat/completions", json.dumps(body).encode(), headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.load(r)
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise RuntimeError(f"Continuity judge unreachable at {url}: {e}") from e
    return data["choices"][0]["message"].get("content") or ""


def parse_json_loose(text: str):
    """Pull the first JSON object out of model output (fences, <think> blocks, trailing prose, trailing commas)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                chunk = text[start:i + 1]
                for cand in (chunk, re.sub(r",\s*([}\]])", r"\1", chunk)):
                    try:
                        return json.loads(cand)
                    except Exception:
                        pass
                return None
    return None


_cache: dict[str, dict] = {}


def check_image(img, bible: Bible, *, compare_to=None, url=None, model=None, api_key=None,
                threshold: float = PASS_SCORE, use_cache: bool = True) -> dict:
    """Check one frame. Returns {pass, score, issues:[{fact, observed, severity, fix}], fix_instruction, seconds, facts:[...]}."""
    img = to_pil(img)
    facts = bible.facts
    if not facts:
        return {"pass": True, "score": 1.0, "issues": [], "fix_instruction": "", "seconds": 0.0, "facts": []}
    key = None
    if use_cache:
        key = "|".join([image_hash(img), "\n".join(f"{f.severity}:{f.text}" for f in facts),
                        ",".join(image_hash(r) for r in bible.refs), image_hash(compare_to) if compare_to is not None else "",
                        str(model), str(threshold)])
        if key in _cache:
            return dict(_cache[key], cached=True)
    msgs = _build_messages(img, facts, bible.refs, compare_to)
    t0 = time.time()
    parsed = None
    raw = ""
    for attempt in range(2):
        try:
            raw = chat(msgs, url=url, model=model, api_key=api_key)
        except RuntimeError:
            if attempt:
                raise
            continue  # one retry: the shared local judge sometimes stalls
        parsed = parse_json_loose(raw)
        if parsed and isinstance(parsed.get("facts"), list):
            break
        parsed = None
    secs = time.time() - t0
    if parsed is None:
        raise RuntimeError(f"Continuity judge returned unparseable output: {raw[:300]!r}")
    by_id = {}
    for k, item in enumerate(parsed["facts"], 1):
        if isinstance(item, dict):
            try:
                by_id[int(item.get("id", k))] = item
            except (TypeError, ValueError):
                by_id[k] = item
    rows, issues = [], []
    total = sum(SEVERITY_WEIGHT[f.severity] for f in facts)
    lost = 0.0
    major_fail = False
    for i, f in enumerate(facts, 1):
        it = by_id.get(i, {})
        status = str(it.get("status", "unclear")).lower().strip()
        status = "violated" if status.startswith("viol") or status in ("fail", "no", "false") else ("ok" if status in ("ok", "pass", "yes", "true") else "unclear")
        rows.append({"fact": f.text, "severity": f.severity, "status": status, "observed": str(it.get("observed", ""))})
        if status == "violated":
            lost += SEVERITY_WEIGHT[f.severity]
            major_fail |= f.severity == "major"
            issues.append({"fact": f.text, "observed": str(it.get("observed", "")), "severity": f.severity,
                           "fix": str(it.get("fix") or "") or f"Change the image so that: {f.text}"})
    score = round(max(0.0, 1.0 - lost / total), 3)
    res = {"pass": (not major_fail) and score >= threshold, "score": score, "issues": issues,
           "fix_instruction": make_fix_instruction(issues), "seconds": round(secs, 1), "facts": rows}
    if key:
        _cache[key] = res
    return res


def make_fix_instruction(issues: list[dict]) -> str:
    if not issues:
        return ""
    edits = "; ".join(i["fix"].strip().rstrip(".") for i in sorted(issues, key=lambda i: i["severity"] != "major"))
    return (f"Change ONLY the following: {edits}. Keep everything else identical: composition, camera angle, "
            "lighting, background, pose, and all other details.")


