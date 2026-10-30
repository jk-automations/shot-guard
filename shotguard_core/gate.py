"""Gate: all checks -> PASS / RETRY / FAIL, a score for picking the best take, and plain-English fix hints that
can be added to the generator prompt. Severity weights follow Continuity Lock (major 1.0, minor 0.4)."""
import time

WEIGHT = {"major": 1.0, "minor": 0.4}
ORDER = ("dialogue", "speaker", "identity", "blocking", "text", "cuts", "glitch", "audio", "facts", "sequence", "attention")


def _line(spec, n):
    s = spec["script"][n - 1]
    return s["text"]


def hint(issue, spec):
    k, name = issue["kind"], issue.get("character", "")
    if k == "side_swap":
        return f"keep {name} on the {issue['expected']} of the frame for the whole take; the camera stays on this side of the action"
    if k == "left_frame":
        return f"keep {name} in frame for the whole take"
    if k == "identity_drift":
        return f"{name} keeps exactly the same face, hair and features from the first frame to the last"
    if k == "not_found":
        return f"show {name} clearly, matching the reference"
    if k in ("missing_line", "wrong_words"):
        return f"say line {issue['line']} word for word: \"{_line(spec, issue['line'])}\""
    if k == "out_of_order":
        return "speak the lines in this order, one at a time: " + " ".join(
            f"{i + 1}) \"{s['text']}\"" for i, s in enumerate(spec["script"]))
    if k == "merged_lines":
        a, b = issue["lines"]
        return f"leave a clear pause between \"{_line(spec, a)}\" and \"{_line(spec, b)}\"; one speaker at a time"
    if k == "extra_speech":
        return "no extra lines beyond the script"
    if k == "no_audio":
        return "the take needs the spoken dialogue with sound"
    if k == "garbled":
        return f"the sign reads exactly \"{issue['expected']}\", spelled correctly"
    if k == "not_allowed":
        allowed = spec.get("allowed_text") or []
        return ("no readable text other than " + ", ".join(f"\"{a}\"" for a in allowed)) if allowed else "no readable text, logos or brand names"
    if k == "side_flip_across_cut":
        return f"keep {name} on the {issue['expected']} of the frame, as in the previous shot"
    if k == "fact_violated":
        return f"keep this exactly as in the reference: {issue['fact']}"
    if k == "silent_track":
        return "the take needs the spoken dialogue with sound"
    if k == "too_quiet":
        return "clear, full-level sound"
    if k == "clipping":
        return "clean sound with no distortion; keep the levels below clipping"
    if k == "dead_air":
        return "no long silences between the lines"
    if k == "music_over_voice":
        return "keep the music low under the dialogue so every word is clear"
    if k == "wrong_speaker":
        return f"only {name} speaks line {issue['line']} (\"{_line(spec, issue['line'])}\"); {issue['speaking']} listens with a closed mouth"
    if k == "frozen":
        return "smooth continuous motion for the whole take, no frozen frames"
    if k == "stutter":
        return "smooth motion at a steady frame rate"
    if k == "flicker":
        return "steady lighting and exposure, no flicker"
    if k == "slow_hook":
        return "open on movement or the first line within the first second"
    if k == "dead_stretch":
        return "keep something happening the whole time: movement or a line, no long still pauses"
    if k in ("hard_cut", "angle_jump"):
        return "one continuous shot: no cuts and no change of camera angle"
    return ""


def decide(spec, checks, video_path=None):
    checks = [c for c in checks if c]
    issues = [dict(i, check=c["check"]) for c in checks for i in c.get("issues", [])]
    lost = sum(WEIGHT.get(i["severity"], 1.0) for i in issues)
    passed = not any(i["severity"] == "major" for i in issues)
    attempt, cost, cap = spec["attempt"], spec["cost_per_take"], spec["credit_cap"]
    spent = (attempt + 1) * cost
    budget_ok = cap <= 0 or spent + cost <= cap + 1e-9
    if passed:
        verdict = "PASS"
    elif attempt < spec["max_retries"] and budget_ok:
        verdict = "RETRY"
    else:
        verdict = "FAIL"
    hints = []
    for i in sorted(issues, key=lambda i: (i["severity"] != "major", ORDER.index(i["check"]) if i["check"] in ORDER else 9)):
        h = hint(i, spec)
        if h and h not in hints:
            hints.append(h)
    prompt_hint = ("Shot Guard fix: " + "; ".join(hints) + ". Keep everything else identical.") if hints else ""
    why = ("all checks passed" if passed else
           f"{sum(i['severity'] == 'major' for i in issues)} major issue(s)"
           + ("" if verdict == "RETRY" else
              f"; no retries left ({attempt}/{spec['max_retries']})" if attempt >= spec["max_retries"] else
              f"; credit cap {cap:g} reached (spent {spent:g})"))
    return {"verdict": verdict, "passed": passed, "score": round(100.0 / (1.0 + lost), 1), "issues": issues,
            "hints": hints, "prompt_hint": prompt_hint, "why": why, "attempt": attempt, "spent": spent,
            "checks": {c["check"]: {k: v for k, v in c.items() if k not in ("segments",)} for c in checks},
            "video": video_path, "spec": spec, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
