# Demo plan: "Every take, checked."

Entry: Comfy "Open Source Higgsfield" challenge, category **Most Innovative**. Deadline 2026-10-30 9:00 PT.
Target length: under 2 minutes (v3 is about 110 s), 16:9 plus a 9:16 cut, narrated with a local voice (Kokoro-82M,
Apache-2.0, CPU, $0) and classic captions.

North star: closed platforms sell a virality predictor (upload a video, get hook and attention scores). Shot Guard is
the open version of that idea plus the step before it: is the take correct?

## The claim, and what backs it

| Claim on screen | Evidence (local) |
|---|---|
| Six catches on real takes from our own pilot film | the fixture takes (`fixtures/`, local only), annotated by `tools/make_annotated.py` |
| "Wrong lines, wrong mouth" | Speaker check: line 1 voiced by the other host, 2.1x mouth movement (`docs/LESSONS.md`) |
| "A frozen second (simulated)" | a good take with 1.2 s frozen on purpose: no real take froze. Said on screen and in the voice |
| The fix loop | **simulated generator**: `workflows/shot-guard-retry-demo.json` re-queues through the real `/prompt` API. Said in the voice |
| Auto-trim RETRY -> PASS | real: hidden_cut and end_drift trimmed and re-checked |
| Hook score and attention curve | Attention check, rules not a trained model; numbers come from the take shown |
| Dailies page | `tools/dailies.py` on the fixture folder |
| "17 takes it had never seen: 2 sent back" | `docs/BLIND-TEST.md` (v2 checks). Speaker and Glitch were run on the same takes on their own: no flags |

## Beats (v3)

1. Cold open: four 1 s bites of real failures. "AI video models make mistakes you only find by watching every take."
2. Title: the node pack and the checks it runs.
3. Six catches with a "CAUGHT" counter and a tick: side swap, wrong lines and wrong speaker, misspelled sign, hidden
   cut, cut then a different face, frozen second (simulated).
4. Fix loop split screen: attempt 0 RETRY next to attempt 1 PASS.
5. Auto-trim split screen: before RETRY, trimmed PASS. No new generation, no credits.
6. Best of N ranking.
7. Attention: hook score and attention curve on a passing take.
8. Dailies: a folder of takes in, one ranked page out, with the credits meter.
9. Plug in anywhere: Seedance 2.5 template, an LTX graph, a Veo graph.
10. Blind test card, end card.

## Render

1. `python tools/make_annotated.py --real` (runs/annotated-real) and `python tools/dailies.py runs/dailies-src --out runs/dailies-real --no-blur`
2. `python3 tools/make_demo.py --real` writes `runs/demo-real/demo-16x9.mp4` and `demo-9x16.mp4`.

Without `--real` the same scripts write the public-safe version (role names, blurred faces, muted real-person takes).
The hosts agreed on 2026-10-29 to be shown with their real faces, voices and first names in the demo; the repo itself
keeps role names and blurred faces. The `--real` outputs are never committed.

## Open before submission (founder gates)

- Public GitHub repo and push: founder approval.
- The submission form: founder submits or approves.
- A real paid retry to replace the simulated loop: founder approval and a price quote first.
