"""Shot Guard core: checks a finished take against its spec. Pure Python + OpenCV + whisper.cpp + Florence-2, CPU only."""
from . import annotate, attention, audio, batch, best, characters, card, credits, html_report, prompt_spec, cuts, dialogue, facts, gate, glitch, report, retry, sequence, speaker, spec, text, trim

CHECKS = {
    "dialogue": dialogue.check,
    "speaker": speaker.check,
    "identity": characters.check_identity,
    "blocking": characters.check_blocking,
    "text": text.check,
    "cuts": cuts.check,
    "audio": audio.check,
    "glitch": glitch.check,
    "attention": attention.check,   # warnings only: hook, attention curve, dead stretches
    "facts": facts.check,      # skips itself (no model call) when the spec has no locked facts
}


def run(path, sp, which=None):
    """Run the named checks (default all) and the gate. Returns the gate result."""
    results = [CHECKS[k](path, sp) for k in (which or CHECKS)]
    return gate.decide(sp, results, video_path=path)
