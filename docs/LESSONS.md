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
