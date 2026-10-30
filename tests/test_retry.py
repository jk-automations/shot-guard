"""The Retry loop with the generator mocked: nothing is sent to any model or API. A fake 'ComfyUI' receives each
re-queued graph, 'generates' a take, and the gate judges it with a scripted outcome per seed."""
import json

import pytest

from shotguard_core import gate, retry, spec

GRAPH = {
    "3": {"class_type": "KSampler", "inputs": {"seed": 100, "steps": 20, "model": ["4", 0]}},
    "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "two people talk in a bakery, warm light", "clip": ["4", 1]}},
    "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry", "clip": ["4", 1]}, "_meta": {"title": "Negative"}},
    "10": {"class_type": "ShotGuardSpec", "inputs": {"attempt": 0, "take_name": "t"}},
    "20": {"class_type": "ShotGuardRetry", "inputs": {"run_id": "", "result": ["12", 0]}},
}
SWAP = {"kind": "side_swap", "severity": "major", "t0": 2.5, "t1": 7.8, "character": "Rosa", "expected": "left", "message": "swap"}
DRIFT = {"kind": "identity_drift", "severity": "major", "t0": 7, "t1": 8, "character": "Rosa", "message": "drift"}


class FakeComfy:
    """Stands in for the /prompt API and the generator. outcomes: seed -> list of issues for the take made with it."""

    def __init__(self, tmp_path, outcomes, max_retries=3, credit_cap=0.0, cost=1.0):
        self.tmp, self.outcomes, self.posted = tmp_path, outcomes, []
        self.sp = dict(max_retries=max_retries, credit_cap=credit_cap, cost_per_take=cost)

    def execute(self, graph):
        """One run of the graph: generate (fake file), check, gate, then the Retry node."""
        seed = graph["3"]["inputs"]["seed"]
        take = self.tmp / f"take_{seed}.mp4"
        take.write_bytes(f"video for seed {seed}".encode())
        sp = spec.make(attempt=graph["10"]["inputs"]["attempt"], **self.sp)
        issues = self.outcomes.get(seed, [])
        r = gate.decide(sp, [{"check": "blocking", "status": "fail" if issues else "pass", "issues": issues, "metrics": {}}], str(take))
        return retry.step(r, graph=graph, video_path=str(take), run_id=graph["20"]["inputs"]["run_id"],
                          ledger_root=str(self.tmp / "runs"), post=self.post, retry_id="20")

    def post(self, graph):
        self.posted.append(graph)
        return {"prompt_id": f"p{len(self.posted)}"}

    def run_until_done(self, graph, limit=20):
        out = self.execute(graph)
        while out["requeued"] and limit:
            out = self.execute(self.posted[-1])
            limit -= 1
        return out


def test_mock_generator_seed_and_prompt_are_targets():
    g = {"1": {"class_type": "ShotGuardMockGenerator", "inputs": {"seed": 0, "prompt": "a scene", "takes": "a.mp4"}},
         "2": {"class_type": "ShotGuardSpec", "inputs": {"attempt": 0, "take_name": "a much longer take name than the prompt"}}}
    assert retry.find_targets(g) == (("1", "seed"), ("1", "prompt"))


def test_targets_found_automatically_and_explicitly():
    assert retry.find_targets(GRAPH) == (("3", "seed"), ("6", "text"))
    assert retry.find_targets(GRAPH, "3.seed", "7.text") == (("3", "seed"), ("7", "text"))


def test_requeue_changes_seed_prompt_attempt_and_run_id(tmp_path):
    fc = FakeComfy(tmp_path, {100: [SWAP]})
    out = fc.execute(GRAPH)
    assert out["requeued"] and out["prompt_id"] == "p1" and len(fc.posted) == 1
    g = fc.posted[0]
    assert g["3"]["inputs"]["seed"] == 101 and g["10"]["inputs"]["attempt"] == 1 and g["20"]["inputs"]["run_id"] == out["run_id"]
    assert g["6"]["inputs"]["text"].startswith("two people talk in a bakery, warm light Shot Guard fix: keep Rosa on the left")
    assert g["7"]["inputs"]["text"] == "blurry"                      # negative prompt untouched
    assert GRAPH["3"]["inputs"]["seed"] == 100                       # original graph not mutated
    g2, _ = retry.next_graph(g, {"prompt_hint": "Shot Guard fix: new. Keep everything else identical."}, "r")
    assert g2["6"]["inputs"]["text"].count("Shot Guard fix:") == 1    # hints replace, never pile up


def test_loop_stops_on_first_pass_and_keeps_it(tmp_path):
    fc = FakeComfy(tmp_path, {100: [SWAP, DRIFT], 101: [SWAP], 102: []})
    out = fc.run_until_done(GRAPH)
    assert len(fc.posted) == 2 and out["status"].startswith("PASS: best take is attempt 2")
    assert open(out["best_path"]).read() == "video for seed 102"
    led = json.load(open(out["ledger"]))
    assert [a["attempt"] for a in led["attempts"]] == [0, 1, 2] and led["final"]["passed"]
    assert [a["seed"] for a in led["attempts"]] == [100, 101, 102]


def test_max_retries_cap_keeps_best_failing_take(tmp_path):
    fc = FakeComfy(tmp_path, {100: [SWAP, DRIFT], 101: [SWAP], 102: [SWAP, DRIFT, DRIFT]}, max_retries=2)
    out = fc.run_until_done(GRAPH)
    assert len(fc.posted) == 2                                       # attempts 0, 1, 2 then stop
    assert out["status"].startswith("FAIL: best take is attempt 1")  # fewest issues
    assert open(out["best_path"]).read() == "video for seed 101"


def test_credit_cap_stops_before_overspending(tmp_path):
    fc = FakeComfy(tmp_path, {s: [SWAP] for s in range(100, 120)}, max_retries=10, credit_cap=2.5, cost=1.0)
    out = fc.run_until_done(GRAPH)
    led = json.load(open(out["ledger"]))
    assert len(led["attempts"]) == 2 and len(fc.posted) == 1          # 2 takes = 2.0 spent; a 3rd would be 3.0 > 2.5
    assert max(a["spent"] for a in led["attempts"]) <= 2.5 and "credit cap" in out["status"]


def test_disabled_retry_never_posts(tmp_path):
    fc = FakeComfy(tmp_path, {100: [SWAP]})
    sp = spec.make(max_retries=3)
    r = gate.decide(sp, [{"check": "blocking", "status": "fail", "issues": [SWAP], "metrics": {}}], None)
    out = retry.step(r, graph=GRAPH, ledger_root=str(tmp_path / "runs"), post=fc.post, enabled=False)
    assert not out["requeued"] and fc.posted == []


@pytest.mark.parametrize("cls", ["ShotGuardSpec", "ShotGuardRetry"])
def test_graph_classes_used_by_retry_exist(comfy_path, cls):
    import shotguard_nodes
    assert cls in shotguard_nodes.NODE_CLASS_MAPPINGS
