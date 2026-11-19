"""Shot Guard core: checks a finished take against its spec. Pure Python + OpenCV + whisper.cpp + Florence-2, CPU only."""
from . import annotate, attention, audio, batch, best, captions, characters, card, credits, html_report, places, prompt_spec, cuts, dialogue, facts, gate, glitch, report, retry, sequence, speaker, spec, stillness, text, trim

CHECKS = {
    "dialogue": dialogue.check,
    "speaker": speaker.check,
    "identity": characters.check_identity,
    "blocking": characters.check_blocking,
    "text": text.check,
    "cuts": cuts.check,
    "audio": audio.check,
    "glitch": glitch.check,
    "stillness": stillness.check,   # photograph with a camera move; motion against a reference ad
    "places": places.check,    # skips itself when the spec has no continuity groups
    "captions": captions.check,   # skips itself when the spec has no captions; calls a vision LLM
    "attention": attention.check,   # warnings only: hook, attention curve, dead stretches
    "facts": facts.check,      # skips itself (no model call) when the spec has no locked facts
}


def run(path, sp, which=None):
    """Run the named checks (default all) and the gate. Returns the gate result."""
    results = [CHECKS[k](path, sp) for k in (which or CHECKS)]
    return gate.decide(sp, results, video_path=path)
