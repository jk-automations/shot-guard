"""Spec from the prompt: read the prompt the generator was given and fill in what the take must contain, so the
script is never typed twice and the check always matches what was asked for.
- Every quoted line is a script line. Its speaker is the cast member mentioned last before the quote (the cast list
  maps names to the words the prompt uses for them: "Rosa: baker, yellow headscarf").
- A quote after "sign", "reads", "logo", "banner" or "written" is allowed on-screen text, not dialogue.
- "<name or keyword> on the left/right" sets that character's side.
A prompt written as a script ("Name: line" per line, no quotes) is read with spec.parse_script. Plain rules, no model."""
import re

from . import spec as spec_mod

QUOTE = re.compile(r'"([^"]+)"|“([^”]+)”')
TEXT_WORDS = re.compile(r"\b(sign|signs|reads|read|logo|banner|written|writes|label|neon|poster|menu board)\b", re.I)
SIDE = re.compile(r"\bon the (left|right)\b", re.I)


def parse_cast(text):
    """"Name: keyword, keyword" per line -> {name: [keywords]}. The name itself always counts as a keyword."""
    cast = {}
    for raw in (text or "").splitlines():
        name, _, keys = raw.partition(":")
        name = name.strip()
        if name:
            cast[name] = [name.lower()] + [k.strip().lower() for k in re.split(r"[,;]", keys) if k.strip()]
    return cast


def _mentions(text, cast):
    """(position, name) for every cast keyword in text."""
    out = []
    low = text.lower()
    for name, keys in cast.items():
        for k in keys:
            for m in re.finditer(r"\b" + re.escape(k) + r"\b", low):
                out.append((m.start(), name))
    return sorted(out)


def extract(prompt, cast=None):
    """{script, allowed_text, sides} from a prompt. cast: {name: [keywords]} or the "Name: keywords" text."""
    cast = parse_cast(cast) if isinstance(cast, str) else (cast or {})
    from .retry import HINT_RE
    prompt = HINT_RE.sub("", prompt or "")   # Retry's own fix hint quotes the lines again; never read it as script
    if not QUOTE.search(prompt) and any(":" in ln for ln in prompt.splitlines()):
        return {"script": spec_mod.parse_script(prompt), "allowed_text": [], "sides": {}}
    script, allowed, sides = [], [], {}
    mentions = _mentions(prompt, cast)
    last_end, speaker = 0, ""
    for m in QUOTE.finditer(prompt):
        said = (m.group(1) or m.group(2)).strip()
        before = prompt[last_end:m.start()]
        # the words just before this quote, back to the end of the previous sentence or quote
        clause = re.split(r"(?<=[.!?])\s+", before)[-1] if before.strip() else before
        if TEXT_WORDS.search(clause[-60:]):
            allowed.append(said)
        else:
            in_clause = [n for p, n in mentions if m.start() - len(clause) <= p < m.start()]
            if in_clause:
                speaker = in_clause[-1]
            else:
                # "He says", a crowd, someone off screen: unknown. The speaker check then skips the line rather than guess
                speaker = ""
            script.append({"speaker": speaker, "text": said})
        last_end = m.end()
    for m in SIDE.finditer(prompt):
        near = [n for p, n in mentions if m.start() - 120 <= p < m.start()]
        if near:
            sides.setdefault(near[-1], m.group(1).lower())
    return {"script": script, "allowed_text": allowed, "sides": sides}


def make_spec(prompt, cast=None, refs=None, **kw):
    """A full spec from a prompt. refs: {name: [reference images]} for the cast members to check by face."""
    e = extract(prompt, cast)
    names = list(parse_cast(cast) if isinstance(cast, str) else (cast or {}))
    chars = [{"name": n, "refs": (refs or {}).get(n, []), "side": e["sides"].get(n, "any")} for n in names]
    allowed = list(dict.fromkeys((kw.pop("allowed_text", None) or []) + e["allowed_text"]))
    return spec_mod.make(script=e["script"], characters=chars, allowed_text=allowed, **kw)
