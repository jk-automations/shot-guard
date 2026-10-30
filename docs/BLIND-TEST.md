# Blind test (2026-10-28)

Shot Guard v1 was tuned on 8 known takes and passed all 8. That says little, because thresholds tuned on a set will
always pass that set. So we ran it on **17 takes it had never seen**: every other take from the same short film,
each checked against the prompt it was generated from (lines in order, "one continuous take, no cuts", "no on-screen
text except signs already in the frame"). Takes were split into a **dev** half (used to fix things) and a
**held-out** half (only measured). Every flag was judged by eye against the frames and the audio transcript; the 4 cases I could not settle alone were
judged by the founder (who listened to the audio), and agreed with my labels on all four.

## v1: before any change

| Check | Flags | Real problems | Main false alarms |
|---|---|---|---|
| Hidden cuts | 25 | 2 | a person walking past the lens, a flour explosion, fast camera moves, a phone popping in |
| Identity drift | 5 | 0 | head turns, laughing, a new close-up after a cut |
| Text | 4 | 0 (1 borderline) | a real "FOR LEASE" sign that is in the start frame, a sign cut off by the frame edge |
| Dialogue | 9 | 0 (3 unsure) | "Oh!" for "Ooooooh!", "50,000", two identical lines, an ASR slip, fast back-and-forth |
| Blocking | 0 | n/a | no side rules in these prompts, so not tested here |

13 of 17 takes got RETRY, almost all for the wrong reason. v1 was overfit.

## What changed in v2, and which half motivated it

| Change | Motivated by |
|---|---|
| Hard-cut threshold 0.8 -> 1.45 (real cuts score 1.56 to 10.6; false alarms up to 1.32) | dev + the original takes |
| The multi-frame "angle jump" score is now a warning: it never caught a cut the hard-cut score missed, and caused about 13 false alarms | dev + held-out |
| Identity drift is a warning unless the likeness drops by 0.25 or more: SFace cannot tell a real drift (0.78 -> 0.63) from a head turn or a laugh (0.72 -> 0.58) on stylised faces. Frontal-only filtering did not help (the real drift is not frontal) | dev + held-out |
| Words already readable in the first frame are allowed ("signs already in the frame") | dev |
| Stretched words, and "merged lines" only when whisper put no full stop between the two lines | dev |
| Numbers of 1,000 and more, two identical lines, small ASR spelling slips | held-out (bugs) |
| The absolute likeness floor only counts for a face that matched well at the start | held-out |
| Text smaller than 3% of the frame height is ignored (set `min_text_height` to 0 for close-up ads) | founder review of the borderline chalkboard |

Because some fixes came from the held-out half, only the **hidden-cut** numbers below are truly blind.

## v2: after the changes

| | v1 | v2 |
|---|---|---|
| Takes sent back (RETRY) | 13 of 17 | 2 of 17 |
| ...for a real problem | 2 | 2 (both real cuts) |
| Hidden cuts, held-out half | 2 real of 6 flags | **2 real of 2 flags** |
| Warnings to check by eye | n/a | 11 (none were real problems) |
| False fails left | many | 1: a likeness drop right after a real cut into a close-up (that take fails for its cut anyway) |

The original 8 takes: all 5 known-bad takes still fail, and the 2 good takes and the final sign pass. On the drift
take the drift is now a warning; the take still fails for the cut inside it.

## Not measured

- **Missed problems (recall).** We judged every flag, but only watched 2 of the 17 takes frame by frame for problems
  Shot Guard did not flag. Treat "PASS" as "nothing obvious", not as a guarantee.
- **Blocking on unseen takes.** None of these prompts set a side, so blocking was only tested on the known side-swap take.
- **Who is speaking.** We tried mouth movement and MediaPipe's jaw-open score to check that the right character says
  each line. Neither works on these stylised faces (the baker's jaw-open score is 0.00 even while she talks), so there is no speaker check.

Reproduce: `tools/blind_test.py` (specs from prompts live in the private `fixtures/blind/`).
