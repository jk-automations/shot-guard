"""Dialogue check: transcribe the take (whisper.cpp on CPU), then match every script line to what was said.
Reports per line word accuracy, missing lines, extra speech, lines spoken out of order, and lines of two
different speakers merged into one breath."""
import json
import os
import re
import shutil
import subprocess
import tempfile
from difflib import SequenceMatcher

from .video import extract_wav

DEFAULT_MODEL = os.path.expanduser("~/.cache/whisper.cpp/ggml-base.en.bin")
ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def num_words(n):
    if n >= 1000:
        return num_words(n // 1000) + " thousand" + ("" if n % 1000 == 0 else " " + num_words(n % 1000))
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + ("" if n % 10 == 0 else " " + ONES[n % 10])
    if n < 1000:
        return ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + num_words(n % 100))


def words(text):
    """Lower-case words; numbers spelled out; sound tags like (gasps) or [music] removed."""
    text = re.sub(r"\([^)]*\)|\[[^\]]*\]|\*[^*]*\*", " ", text.lower())
    text = text.replace(",", "")  # 50,000
    text = re.sub(r"\d+", lambda m: num_words(int(m.group())), text)
    text = text.replace("-", " ")
    text = re.sub(r"([a-z])\1{2,}", r"\1", text)   # stretched words: "ooooooh" -> "oh"
    return re.findall(r"[a-z']+", text)


def same_word(a, b):
    """ASR spelling slips ("crisants" for "croissants") count as the same word; short words must match exactly."""
    return a == b or (min(len(a), len(b)) >= 5 and SequenceMatcher(None, a, b).ratio() >= 0.8)


def transcribe(path, model=None, whisper=None, threads=4):
    """Segments [{t0, t1, text}] from whisper-cli. CPU only (-ng)."""
    whisper = whisper or os.environ.get("SHOTGUARD_WHISPER") or shutil.which("whisper-cli") or "/opt/homebrew/bin/whisper-cli"
    model = model or os.environ.get("SHOTGUARD_WHISPER_MODEL") or DEFAULT_MODEL
    tmp = tempfile.mkdtemp(prefix="shotguard-asr-")
    try:
        wav = extract_wav(path, os.path.join(tmp, "a.wav"))
        if not wav:
            return None
        base = os.path.join(tmp, "out")
        r = subprocess.run([whisper, "-ng", "-t", str(threads), "-m", model, "-f", wav, "-oj", "-of", base, "-np"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"whisper-cli failed: {r.stderr[-400:]}")
        with open(base + ".json") as f:
            data = json.load(f)
        return [{"t0": s["offsets"]["from"] / 1000.0, "t1": s["offsets"]["to"] / 1000.0, "text": s["text"].strip()}
                for s in data.get("transcription", []) if s["text"].strip()]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _heard(segments):
    """Flatten segments into words with a segment index and an estimated time."""
    out = []
    for si, s in enumerate(segments):
        ws = words(s["text"])
        # word positions in the raw text, to see whether whisper put sentence punctuation after each word
        raw = re.findall(r"[A-Za-z0-9',]+[.?!]*", re.sub(r"\([^)]*\)|\[[^\]]*\]", " ", s["text"]))
        ends = [bool(re.search(r"[.?!]$", r)) for r in raw]
        for k, w in enumerate(ws):
            out.append({"w": w, "seg": si, "t": s["t0"] + (k + 0.5) / len(ws) * (s["t1"] - s["t0"]),
                        "stop": ends[k] if len(ends) == len(ws) else None})
    return out


def _best_window(line, heard, taken=()):
    """Where in the transcript this line was said: the window with the most matching words. Words already used by
    an earlier line with the same text are skipped (two identical lines must match two different places)."""
    n, best = len(line), (0.0, 0, 0, 0)
    hw = [h["w"] for h in heard]
    for s in range(len(hw)):
        if s in taken:
            continue
        for ln in range(max(1, n - 2), n + 3):
            if s + ln > len(hw):
                break
            win = hw[s:s + ln]
            m = SequenceMatcher(None, line, [next((x for x in line if same_word(x, y)), y) for y in win], autojunk=False)
            matched = sum(b.size for b in m.get_matching_blocks())
            score = (matched / n, m.ratio())
            if score > best[:2]:
                best = (matched / n, m.ratio(), s, s + ln)
    return {"accuracy": best[0], "start": best[2], "end": best[3]}


def _lis(seq):
    """Indices of a longest increasing subsequence."""
    import bisect
    tails, idx, prev = [], [], [-1] * len(seq)
    for i, v in enumerate(seq):
        k = bisect.bisect_left(tails, v)
        if k == len(tails):
            tails.append(v); idx.append(i)
        else:
            tails[k] = v; idx[k] = i
        prev[i] = idx[k - 1] if k else -1
    out, i = [], idx[-1] if idx else -1
    while i >= 0:
        out.append(i); i = prev[i]
    return set(out)


def analyze(script, segments, line_accuracy=0.8, found_accuracy=0.5, min_extra_words=2):
    """script: [{speaker, text}]. segments: whisper segments. Returns issues + per-line metrics (no I/O)."""
    heard = _heard(segments)
    lines, issues = [], []
    used = {}
    for i, ln in enumerate(script):
        lw = words(ln["text"])
        taken = used.get(tuple(lw), set())
        m = _best_window(lw, heard, taken) if heard and lw else {"accuracy": 0.0, "start": -1, "end": -1}
        if m["start"] >= 0:
            used.setdefault(tuple(lw), set()).update(range(m["start"], m["end"]))
        found = m["accuracy"] >= found_accuracy
        t = heard[m["start"]]["t"] if found else None
        lines.append(dict(index=i + 1, speaker=ln.get("speaker", ""), text=ln["text"], accuracy=round(m["accuracy"], 3),
                          found=found, start=m["start"], end=m["end"], t=t,
                          heard=" ".join(h["w"] for h in heard[m["start"]:m["end"]]) if found else ""))
    for L in lines:
        if not L["found"]:
            issues.append({"kind": "missing_line", "severity": "major", "t0": None, "t1": None, "line": L["index"],
                           "message": f"line {L['index']} was not said (best match {L['accuracy']:.0%} of its words)"})
        elif L["accuracy"] < line_accuracy:
            issues.append({"kind": "wrong_words", "severity": "major", "t0": L["t"], "t1": L["t"], "line": L["index"],
                           "message": f"line {L['index']} only {L['accuracy']:.0%} right: heard \"{L['heard']}\""})
    found = [L for L in lines if L["found"]]
    keep = _lis([L["start"] for L in found])
    order = [L["index"] for L in sorted(found, key=lambda L: L["start"])]
    if len(keep) < len(found):
        late = [L for k, L in enumerate(found) if k not in keep]
        issues.append({"kind": "out_of_order", "severity": "major", "t0": min(L["t"] for L in late), "t1": max(L["t"] for L in late),
                       "lines": [L["index"] for L in late],
                       "message": f"lines heard in order {', '.join(map(str, order))} (script: {', '.join(str(L['index']) for L in found)})"})
    for a, b in zip(lines, lines[1:]):
        if a["found"] and b["found"] and a["speaker"] != b["speaker"] and a["end"] > 0 and b["start"] >= 0:
            sa = {heard[k]["seg"] for k in range(a["start"], a["end"])}
            sb = {heard[k]["seg"] for k in range(b["start"], b["end"])}
            gap = heard[a["end"] - 1] if a["end"] - 1 < len(heard) and a["end"] <= b["start"] else None
            # whisper marks pauses with punctuation; a line boundary with a full stop is not a merge, and a
            # segment with no punctuation at all tells us nothing
            trusted = any(h["stop"] for h in heard if h["seg"] in sa | sb)
            if sa & sb and trusted and not (gap and gap["stop"]):
                issues.append({"kind": "merged_lines", "severity": "major", "t0": min(a["t"], b["t"]), "t1": max(a["t"], b["t"]),
                               "lines": [a["index"], b["index"]],
                               "message": f"lines {a['index']} ({a['speaker']}) and {b['index']} ({b['speaker']}) run together in one breath"})
    covered = set()
    for L in found:
        covered.update(range(L["start"], L["end"]))
    run = []
    for k, h in enumerate(heard + [None]):
        if h is not None and k not in covered:
            run.append(h); continue
        if len(run) >= min_extra_words:
            issues.append({"kind": "extra_speech", "severity": "minor", "t0": run[0]["t"], "t1": run[-1]["t"],
                           "message": f"extra speech not in the script: \"{' '.join(r['w'] for r in run)}\""})
        run = []
    acc = [L["accuracy"] for L in lines]
    return {"lines": lines, "issues": issues, "heard_order": order,
            "metrics": {"mean_accuracy": round(sum(acc) / len(acc), 3) if acc else None, "lines": len(lines),
                        "transcript": " / ".join(s["text"] for s in segments)}}


def check(path, spec):
    th = spec["thresholds"]
    script = spec.get("script") or []
    if not script:
        return {"check": "dialogue", "status": "skip", "issues": [], "metrics": {}, "note": "no script lines in the spec"}
    from .speaker import segments as cached
    segments = cached(path, spec)   # shared with the speaker check: one transcription per take
    if segments is None:
        return {"check": "dialogue", "status": "fail", "metrics": {}, "issues": [
            {"kind": "no_audio", "severity": "major", "t0": None, "t1": None, "message": "the take has no audio track"}]}
    r = analyze(script, segments, line_accuracy=th["line_accuracy"])
    r.update(check="dialogue", status="fail" if r["issues"] and any(i["severity"] == "major" for i in r["issues"]) else "pass",
             segments=segments)
    return r
