"""Build the Shot Guard graphs (ComfyUI API format).
  workflows/shot-guard-api.json        Load Video -> Spec -> 10 checks -> Gate -> Report + Annotated Video (+ Retry, off by default)
  workflows/shot-guard-retry-demo-api.json   the same checks after a mock generator, Retry on: shows the loop
python3 tools/build_app.py"""
import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--video", default="shotguard_take.mp4")
ap.add_argument("--ref", default="shotguard_ref_1.png")
ap.add_argument("--outdir", default="workflows")
a = ap.parse_args()


def graph(retry_demo):
    g = {}
    nid = iter(range(1, 1000))

    def node(cls, title=None, **inputs):
        i = str(next(nid))
        g[i] = {"class_type": cls, "inputs": inputs}
        if title:
            g[i]["_meta"] = {"title": title}
        return i

    def o(i, k=0):
        return [i, k]

    if retry_demo:   # a stand-in for any generator: seed picks a take; the prompt text is recorded, nothing is generated
        gen = node("ShotGuardMockGenerator", "Generator (mock)", seed=0, prompt="Two characters talk across the bakery counter.",
                   takes="shotguard_bad.mp4\nshotguard_good.mp4")
        video = o(gen)
    else:
        video = o(node("LoadVideo", "Your take", file=a.video))
    ref = node("LoadImage", "Character 1 reference", image=a.ref)
    spec = node("ShotGuardSpec", "Spec", take_name="my take", script="", allowed_text="MEGABUN", max_retries=2,
                credit_cap=0.0, cost_per_take=0.0, attempt=0, character_1="Rosa", side_1="left", character_2="", side_2="any",
                character_3="", side_3="any", reference_1=o(ref), thresholds_json="",
                locked_facts="")
    checks = {k: node(cls, title, spec=o(spec), video=video) for k, cls, title in (
        ("dialogue", "ShotGuardDialogue", "Check: Dialogue"), ("identity", "ShotGuardIdentity", "Check: Identity drift"),
        ("blocking", "ShotGuardBlocking", "Check: Blocking"), ("text", "ShotGuardText", "Check: Text"),
        ("cuts", "ShotGuardCuts", "Check: Hidden cuts"), ("facts", "ShotGuardFacts", "Check: Facts (vision judge)"),
        ("speaker", "ShotGuardSpeaker", "Check: Speaker"), ("glitch", "ShotGuardGlitch", "Check: Glitch"),
        ("audio", "ShotGuardAudio", "Check: Audio"), ("attention", "ShotGuardAttention", "Check: Attention (hook)"))}
    gate = node("ShotGuardGate", "Gate", spec=o(spec), **{k: o(v) for k, v in checks.items()})
    node("ShotGuardReport", "Report", result=o(gate), filename_prefix="shotguard/report", blur_faces=False)
    node("ShotGuardAnnotate", "Annotated Video", result=o(gate), filename_prefix="shotguard/annotated", blur_faces=False)
    node("ShotGuardRetry", "Retry", result=o(gate), enabled=retry_demo, seed_target="", prompt_target="", server_url="",
         run_id="", video=video)
    node("PreviewAny", "Verdict", source=o(gate, 3))
    return g


out = Path(a.outdir)
out.mkdir(parents=True, exist_ok=True)
for name, demo in (("shot-guard-api.json", False), ("shot-guard-retry-demo-api.json", True)):
    g = graph(demo)
    (out / name).write_text(json.dumps(g, indent=1))
    print(f"{len(g)} nodes -> {out / name}")
