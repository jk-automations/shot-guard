# Shot Guard lessons (calibration notes, 2026-10-28)

Measured on the real pilot-film takes in `fixtures/` (local only). Feed into the JK Studio proof card after the contest.

## Identity: SFace, not DINOv2
- DINOv2-small CLS on face crops could not tell the two hosts apart (both about 0.6 against either reference) and
  showed no drop on the drifting take. CLIP ViT-B/32 was worse (every face about 0.93 to every other face).
- OpenCV SFace (opencv_zoo, Apache-2.0) after YuNet alignment separates the stylised 3D characters well:
  same character 0.87 to 0.97 across a take, different characters 0.03 to 0.67.
- Drift is measured against the take's own start, not an absolute bar, because small faces in wide shots score
  lower against a close-up reference (0.45 to 0.6) than close-ups do (0.75 to 0.8).
  The drifting take falls from 0.78 to 0.54 to 0.65 after its cut at 4.7 s; good takes never dip more than 0.1 for
  longer than half a second. Rule: 0.12 below the start for 1 s or more.
- Faces cut off by the frame edge are ignored for drift (a half face always scores low). On the drift take this
  means the very last second is not scored: the drifting face is half out of frame there. The drift is caught from
  4.8 s (0.61 against 0.78 at the start).
- Give the spec every visible character. With one reference only, the two hosts scored 0.60 and 0.57 against it
  after the cut and the wrong face got the name (the take was still flagged, for the wrong reason). With both
  references the assignment is clear (0.60 + 0.74 against 0.57 + 0.48). A regression test checks the box position.

## Blocking
- Name face tracks one segment (between cuts) at a time, Hungarian on median similarity, then add each named track
  to the character's gallery. After a cut, faces are also compared with how the character looked earlier in this
  same take, which is more reliable than the reference sheet alone.

## Hidden cuts
- Score = colour-histogram Bhattacharyya distance x mean grey-pixel change. Frame to frame: good takes peak at 0.2,
  real cuts 1.5 to 2.3 (threshold 0.8). Across a 6-frame window (whip or morph jumps that span several frames):
  good takes peak at 1.1, the known angle jump scores 9.3 (threshold 2.5).
- Computed in OpenCV on every frame, so there is no PySceneDetect or ffmpeg-filter dependency.

## Text
- Florence-2-base `<OCR_WITH_REGION>` on CPU: about 6 s a frame. The garbled sign reads "MEGA-BUKEN", the final
  sign "MEGABUN". Non-Latin junk (stray digits) is dropped: a token needs 3+ Latin letters.
- The remote code breaks with the new transformers KV cache: call `generate(..., use_cache=False)`, and patch out
  the `flash_attn` import on CPU.

## Dialogue
- whisper.cpp base.en on CPU is enough: it heard every scripted line of the good take word for word.
- Lines are located independently (best window per line), so reordering shows up; a longest increasing
  subsequence names the lines that are out of place. "Merged" only counts when two different speakers share one
  whisper segment (same-speaker lines in one breath are fine).
- The brief listed a different line order for the dialogue take than the approved edit (`edit-v5.json`). The approved
  edit's order is the truth here; the known-good take follows it.

## Blind test, v2 (2026-10-28 evening): details in docs/BLIND-TEST.md
- Tuning on the same takes you test on hides overfitting: v1 was 8/8 on its fixtures and sent back 13 of 17 unseen
  takes, nearly all for the wrong reason. Always keep a held-out set and say which fixes it motivated.
- Hidden cuts: the frame-to-frame score is the whole detector (real cuts 1.56 to 10.6, false alarms up to 1.32;
  threshold 1.45). The 6-frame window score only found camera moves, people walking past the lens and particle
  effects, and "does the scene come back afterwards" did not separate them either. It is a warning now.
- Identity: SFace drops as much for a head turn or a laugh (0.72 -> 0.58) as for the real drift (0.78 -> 0.63), and
  a yaw filter from YuNet landmarks does not help (the real drift is turned 0.7). Drift is a warning unless the drop
  is 0.25+. Lock hard identity details with a Facts line instead.
- Text: allow what the first frame already shows (image-to-video keeps the start frame's signs). OCR misreads a sign
  cut by the frame edge ("BARY").
- Dialogue: whisper's punctuation is the best pause signal available; only call two speakers "merged" with no full
  stop between them. Collapse stretched words, spell out big numbers, accept small ASR slips (ratio 0.8, 5+ letters).
- Speaker check abandoned: mouth-region motion and MediaPipe FaceLandmarker jawOpen (0.00 on the baker while she talks)
  do not work on these stylised faces at 40 to 150 px.
- Facts (Continuity Lock judge) through the fleet router `local-big` (Qwen3.8, accepts images): about 35 s a frame on
  the Studio; it independently failed the "MEGA-BUKEN" still and passed the final sign.

## Audio check (10-29)

- Measured on 24 real takes before setting any threshold: overall level -27 to -17 dBFS, no clipping (at most 0.03% of
  samples above 95% of full scale), longest mid-take silence 0.8 s, loud-to-quiet spread 9.8 to 31.5 dB.
- AAC rounds off the flat tops of clipped audio, so counting samples at exactly full scale misses clipping after
  decoding; counting samples above 95% of full scale works (a clipped test clip: 18%).
- Tried and dropped: "sound cut off at the end". It fired on 5 of 24 takes; on 3 of them whisper shows the last line
  ending 2 to 4 s before the end and the loud part is music. Without a voice-vs-music split it is noise.
- With these defaults the audio check flags nothing on the 24 real takes. Its value is catching failures we have not
  had yet (silent track, distortion, dead air, music over the voice), which the synthetic tests prove it finds.

## Speaker check (10-29)

- v2 dropped it: plain mouth-region motion and MediaPipe jawOpen could not tell who talks on stylised faces. What works:
  crop the mouth from YuNet's two mouth-corner landmarks, scaled by the eye distance so the crop follows the head, and
  compare the mean frame-to-frame change of each named face while a line is heard.
- Truth: 23 lines with the scripted speaker on screen, judged by eye from per-line face strips. 4 were spoken by the
  wrong host (two in the scrambled take, and two in a take we had approved). Shot Guard flags 3 of them (1.8x, 1.9x,
  2.1x) and misses 1 (1.0x); the 19 right lines stay below 1.4x. A head turn toward profile is the main false signal
  (1.4x). Subtracting eye-region motion to cancel head turns made it worse (it flagged a correct line at 1.9x), so it was
  dropped.
- Thresholds were set on these same 23 lines (no held-out half: too few). Treat 1.6x / 2x as a starting point.

## Glitch check (10-29)

- Measured on 24 real takes: the longest near-still run was 0.46 s, in a calm shot (median change 0.40 per frame), so a
  freeze is judged against the shot's own motion. No take had a repeated-frame stutter pattern or alternating
  brightness. With the defaults it flags none of them; synthetic freeze, stutter and flicker clips prove it finds each.
- No real take froze, so the demo shows a good take with 1.2 s frozen on purpose, labelled as simulated.

## Attention check (10-29)

- On 23 real 8 s takes the hook score (first 2 s) ranged 58 to 85 and no take had a dead stretch: these takes are
  dense. The check exists for longer, slower edits; on these takes it gives scores and an attention curve only.
- It is rules (movement 0.3, speech 0.3, face 0.25, sound 0.15), not a viewer model, and says so.

## Same take, same verdict (10-29)

- All 8 real fixture takes were run twice through every check, in two separate processes with no shared cache:
  verdict, score and every finding (kind, time, message) were identical on all 8, so nothing needed pinning.
  Measured on one machine (the Studio, CPU); `tests/test_consistency.py` keeps it that way on a synthetic take.

## Stillness, Places, Captions (2026-11-18)

Built after the founder stopped the Honest Downgrade ad climb on round 7 with "way too many static images": Shot Guard
had measured the gap (Glitch saw 8 times less motion and 72 repeated frames) and still passed it, because Glitch looks
for a freeze inside one take, not an ad made of stills with a zoom on top. Scored on 9 ads (v8, rounds 1 to 7 and the
Tacos El Guero sample the founder approved) against an answer key in which every label quotes the founder or the
review council. Thresholds were written down before the run. Files: dojo-orchestration
`evidence/judge-three-checks-20261118/` (answer key, pre-registration, scores, logs). 9 ads is not a calibration.

### Stillness
- Probe first (one threshold, 4.0 on the residual after undoing the camera move): r7 49% photo-like pairs, El Guero 16%.
  The share limits (warn 20%, fail 33%) were set knowing those two numbers, so r7 and El Guero do not test them.
- Dark night shots have so little texture that their residual is tiny even when something moves. They are now reported
  as "too dark to judge" and left out of the share. Side effect: on r7 27% of the pairs were dark, its lit share fell to
  32% (a warning) and it failed on the reference comparison instead (median motion 7 against El Guero's 17).
- Result: all 5 ads labelled "too static" fail, El Guero passes (one 2 s stretch warned at 8.6 s). It also fails v8, r1
  and r6, which have no label. Two of the five hits (r4, r5) rest on inherited labels.
- Changed after the results: the full test suite then failed an approved pilot take (good_W4, a calm 3D dialogue shot:
  36% photo-like; good_W1 32%, the scrambled-dialogue take 57%). A photo-like share alone is now a warning; only the
  comparison with a reference ad fails. On the 9 ads both versions give the same verdicts (every ad that failed on the
  share also failed against the reference), so the score is unchanged.
- The LTX food clips (hero plate, lime squeeze) read as photo-like: the build log says the hand "barely moves". The
  check cannot tell a slow real shot from a still; a reference ad is what makes the call fair.

### Places
- DINOv2-small CLS, best frame pair per two spans. On Shot Guard's own pilot clips (calibration before the ads):
  same place in another shot 0.42 to 0.68, different places 0.24 to 0.70 (AUC 0.84); mean-patch and CLS+mean were worse.
  Fail below 0.40 (no pilot same-place pair is that low), warn below 0.50.
- Result: 1 hit (v8 truck, 0.31), 8 misses. The known failure (r2's truck changing model between 7.25 and 12.25 s)
  scores 0.79 to 0.87: a whole-frame embedding sees "a food truck at night in a lot" both times. The v8 hit is probably
  for the framing (a wide shot against a close-up of the window), not the truck model. Plate re-arrangements score 0.88.
- Whole-frame similarity does not catch a same-category object that changes model. Next: crop boxes per span plus an
  instance-level comparison (local feature matching, or a pairwise vision-LLM question "same truck?"), measured on a
  held-out set before it fails anything.

### Captions
- Judge: `local-fast` on the fleet router (Qwen3.6-35B-A3B 4-bit on the two minis, takes images, about 2 s for a probe
  image). `local-big` timed out at 180 s under a Studio load of 115 to 156, so it was not used.
- Result: 0 of 6 hits. Every time, the judge called the opener "Our 200th location." over a glass office tower
  "no_claim" (it saw "a tall building with a crowd and a red ribbon"): a slogan-like caption is not something it treats
  as a claim about the picture. The council's complaint was that an office tower is not a restaurant; that is a
  plausibility judgement, not a contradiction. Descriptive captions ("It's a truck.", "Made by hand.") matched, as they
  should.
- 1 false alarm, on the El Guero sample: with no caption list, OCR can't tell a caption from a sign. It read the truck's
  garbled sign "TACOS EL GÚPRO" as a caption, and it stretched "but people line up at 11pm for these" over the following
  food shots, where the judge then said "no people visible". Give captions in the spec (an edit manifest has them); OCR
  captions are a fallback.
- One local answer in about 80 was broken JSON (an unquoted string): the judge now retries once, like Facts.
