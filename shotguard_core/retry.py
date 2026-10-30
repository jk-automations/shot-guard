"""Retry: re-queue the same ComfyUI graph through the /prompt API with a new seed and the Gate's fix hint added to
the prompt, up to max_retries and the credit cap, and keep the best take. A ledger file per run remembers every
attempt, because each attempt is a separate ComfyUI prompt."""
import copy
import json
import os
import re
import shutil
import time
import urllib.request
import uuid

SEED_KEYS = ("seed", "noise_seed")
PROMPT_KEYS = ("text", "prompt", "positive_prompt")
HINT_RE = re.compile(r"\s*Shot Guard fix:.*$", re.S)
SKIP_CLASSES = ("ShotGuardSpec", "ShotGuardSpecFromPrompt", "ShotGuardDialogue", "ShotGuardIdentity", "ShotGuardBlocking", "ShotGuardText",
                "ShotGuardCuts", "ShotGuardGate", "ShotGuardRetry", "ShotGuardReport", "ShotGuardAudio", "ShotGuardFacts",
                "ShotGuardSequence", "ShotGuardSpeaker", "ShotGuardGlitch", "ShotGuardAnnotate", "ShotGuardTrim",
                "ShotGuardBestOf", "ShotGuardDailies", "ShotGuardAttention")   # never a generator's seed or prompt


def _target(spec_str):
    node, _, key = (spec_str or "").strip().partition(".")
    return (node, key) if node and key else None


def find_targets(graph, seed_target="", prompt_target=""):
    """(node_id, input) for the seed and the positive prompt. Explicit "id.input" wins; otherwise the first literal
    seed input and the longest literal prompt string outside Shot Guard nodes (negative prompts are usually shorter
    and are skipped when their node title says negative)."""
    seed, prompt = _target(seed_target), _target(prompt_target)
    ids = sorted(graph, key=lambda k: (len(k), k))
    if not seed:
        for nid in ids:
            n = graph[nid]
            if n["class_type"] in SKIP_CLASSES:
                continue
            k = next((k for k in SEED_KEYS if isinstance(n["inputs"].get(k), int)), None)
            if k:
                seed = (nid, k)
                break
    if not prompt:
        best = None
        for nid in ids:
            n = graph[nid]
            title = (n.get("_meta", {}).get("title") or "").lower()
            if n["class_type"] in SKIP_CLASSES or "neg" in title:
                continue
            for k in PROMPT_KEYS:
                v = n["inputs"].get(k)
                if isinstance(v, str) and (best is None or len(v) > best[0]):
                    best = (len(v), (nid, k))
        prompt = best[1] if best else None
    return seed, prompt


def next_graph(graph, result, run_id, retry_id=None, seed_target="", prompt_target=""):
    """The graph for the next attempt, plus a list of what changed."""
    g = copy.deepcopy(graph)
    seed, prompt = find_targets(g, seed_target, prompt_target)
    changes = []
    for nid, n in g.items():
        if n["class_type"] in ("ShotGuardSpec", "ShotGuardSpecFromPrompt"):
            n["inputs"]["attempt"] = int(n["inputs"].get("attempt", 0)) + 1
            changes.append(f"{nid}.attempt={n['inputs']['attempt']}")
    if retry_id and retry_id in g:
        g[retry_id]["inputs"]["run_id"] = run_id
    if seed:
        old = g[seed[0]]["inputs"][seed[1]]
        g[seed[0]]["inputs"][seed[1]] = (int(old) + 1) % (2 ** 63)
        changes.append(f"{seed[0]}.{seed[1]}: {old} -> {g[seed[0]]['inputs'][seed[1]]}")
    if prompt and result.get("prompt_hint"):
        old = g[prompt[0]]["inputs"][prompt[1]]
        g[prompt[0]]["inputs"][prompt[1]] = HINT_RE.sub("", old).rstrip() + " " + result["prompt_hint"]
        changes.append(f"{prompt[0]}.{prompt[1]} += fix hint")
    return g, changes


def post_prompt(server, graph, client_id="shot-guard"):
    body = json.dumps({"prompt": graph, "client_id": client_id}).encode()
    req = urllib.request.Request(server.rstrip("/") + "/prompt", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


class Ledger:
    def __init__(self, root, run_id):
        self.dir = os.path.join(root, run_id)
        os.makedirs(self.dir, exist_ok=True)
        self.path = os.path.join(self.dir, "ledger.json")
        self.data = json.load(open(self.path)) if os.path.exists(self.path) else {"run_id": run_id, "attempts": []}

    def add(self, entry):
        self.data["attempts"] = [a for a in self.data["attempts"] if a["attempt"] != entry["attempt"]] + [entry]
        self.data["attempts"].sort(key=lambda a: a["attempt"])
        self.save()

    def save(self):
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=1)

    def best(self):
        """Passing takes first, then the highest score, then the earliest attempt."""
        if not self.data["attempts"]:
            return None
        return max(self.data["attempts"], key=lambda a: (a["passed"], a["score"], -a["attempt"]))


def step(result, graph=None, video_path=None, run_id="", ledger_root="runs/shot-guard", post=None, retry_id=None,
         seed_target="", prompt_target="", enabled=True):
    """One Retry node execution: record this attempt, then either re-queue the next one or finish with the best take."""
    run_id = run_id or uuid.uuid4().hex[:12]
    sp = result["spec"]
    led = Ledger(ledger_root, run_id)
    attempt = result["attempt"]
    kept = None
    if video_path and os.path.exists(video_path):
        kept = os.path.join(led.dir, f"attempt_{attempt}{os.path.splitext(video_path)[1] or '.mp4'}")
        shutil.copyfile(video_path, kept)
    seed, _ = find_targets(graph or {}, seed_target, prompt_target)
    led.add({"attempt": attempt, "verdict": result["verdict"], "passed": result["passed"], "score": result["score"],
             "issues": len(result["issues"]), "hints": result["hints"], "video": kept, "spent": result["spent"],
             "seed": (graph or {}).get(seed[0], {}).get("inputs", {}).get(seed[1]) if seed else None,
             "time": time.strftime("%Y-%m-%dT%H:%M:%S")})
    over_cap = sp["credit_cap"] > 0 and result["spent"] + sp["cost_per_take"] > sp["credit_cap"] + 1e-9
    out = {"run_id": run_id, "ledger": led.path, "attempt": attempt, "requeued": False}
    if result["verdict"] == "RETRY" and enabled and graph and not over_cap and attempt < sp["max_retries"]:
        g, changes = next_graph(graph, result, run_id, retry_id, seed_target, prompt_target)
        resp = (post or (lambda gr: {"prompt_id": None}))(g)
        out.update(requeued=True, changes=changes, prompt_id=resp.get("prompt_id"), graph=g,
                   status=f"RETRY: queued attempt {attempt + 1} of {sp['max_retries']} ({'; '.join(changes)})")
    else:
        best = led.best()
        best_path = None
        if best and best["video"]:
            best_path = os.path.join(led.dir, "best" + os.path.splitext(best["video"])[1])
            shutil.copyfile(best["video"], best_path)
        led.data["final"] = {"best_attempt": best["attempt"] if best else None, "passed": bool(best and best["passed"]),
                             "best_video": best_path}
        led.save()
        why = "retry off" if not enabled else ("credit cap reached" if over_cap else result["why"])
        out.update(best=best, best_path=best_path,
                   status=(f"{'PASS' if best and best['passed'] else 'FAIL'}: best take is attempt {best['attempt']} "
                           f"(score {best['score']}) of {len(led.data['attempts'])}; {why}") if best else "no takes recorded")
    return out
