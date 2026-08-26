# Annotation Quality Report

Disagreements between the curated metadata in `manifest.json` and the model
predictions under `annotations/`. A disagreement is a prompt for human review,
not a verdict: either side may be wrong. Nothing here has been applied to
`manifest.json`.

## How to read this report

Several checks below compare a model's incidental proxy for a curated label against a purpose-built one, and the proxy's blind spots matter more than the raw numbers. This dataset is largely close-up, hand-first gesture crops, so **facial affect is a weak proxy for gesture affect here** -- 78/93 (84%) of detected faces read as Neutral, while the curated `emotional_state` label is `neutral` for only 49/99 gestures: the emotion is carried by hands and posture, not facial expression. Likewise, detected face count is a proxy for curated person count and fails exactly where a gesture is framed hand-first (no face visible) or where a multi-person scene only yields one clearly detected face. CLIP image-to-intent similarity scores are only meaningful relative to each other within this run, not as an absolute threshold. None of this is a verdict on the curated data -- it is context for reading the disagreements below.

**49 finding(s)** across 99 gestures.

| Severity | Check | Gesture | Curated | Predicted | Image |
|---|---|---|---|---|---|
| high | number_of_people | `bicycle-01` | single | 0 | `gesture_images/bicycle/bicycle-01-left-turn-signal.png` |
| high | number_of_people | `bicycle-03` | single | 0 | `gesture_images/bicycle/bicycle-03-stop-signal.png` |
| high | number_of_people | `dir-02` | single | 0 | `gesture_images/directional-and-wayfinding/dir-02-open-palm-point.png` |
| high | number_of_people | `dir-11` | 2 person | 1 | `gesture_images/directional-and-wayfinding/dir-11-after-you-open-palm-sweep.png` |
| high | number_of_people | `greet-05` | 2 person | 1 | `gesture_images/greetings-and-farewells/greet-05-high-five.png` |
| high | number_of_people | `greet-10` | 2 person | 1 | `gesture_images/greetings-and-farewells/greet-10-about-to-hug.png` |
| high | number_of_people | `meme-01` | single | 0 | `gesture_images/meme-and-fantasy/meme-01-dab-pose.png` |
| high | number_of_people | `prac-03` | single | 0 | `gesture_images/practical-and-functional/prac-03-hand-as-camera-framing.png` |
| high | number_of_people | `team-01` | 3 or more | 1 | `gesture_images/team-building/team-01-team-chant-hands.png` |
| high | number_of_people | `urg-02` | single | 0 | `gesture_images/urgency-and-attention/urg-02-hand-clap-for-attention.png` |
| high | number_of_people | `wish-01` | 2 person | 1 | `gesture_images/wishful-and-luck/wish-01-pinky-promise.png` |
| medium | arousal | `affirm-03` | high | low (-0.09) | `gesture_images/affirmative-and-positive/affirm-03-clapping.png` |
| medium | arousal | `affirm-06` | high | low (-0.11) | `gesture_images/affirmative-and-positive/affirm-06-raised-fist-solidarity.png` |
| medium | arousal | `affirm-08` | low | high (+0.13) | `gesture_images/affirmative-and-positive/affirm-08-head-nod-with-chin-lift.png` |
| medium | arousal | `bicycle-01` | low | high (+0.14) | `gesture_images/bicycle/bicycle-01-left-turn-signal.png` |
| medium | arousal | `bicycle-02` | low | high (+0.20) | `gesture_images/bicycle/bicycle-02-right-turn-signal.png` |
| medium | arousal | `bicycle-03` | low | high (+0.12) | `gesture_images/bicycle/bicycle-03-stop-signal.png` |
| medium | arousal | `conv-04` | high | low (-0.09) | `gesture_images/conversational-emphasis/conv-04-hand-chop-into-palm.png` |
| medium | arousal | `dir-02` | low | high (+0.20) | `gesture_images/directional-and-wayfinding/dir-02-open-palm-point.png` |
| medium | arousal | `neg-08` | low | high (+0.11) | `gesture_images/negative-disapproval/neg-08-dismissive-backhand-wave.png` |
| medium | arousal | `number-05` | low | high (+0.09) | `gesture_images/numbers/number-05-five.png` |
| medium | arousal | `number-09` | low | high (+0.11) | `gesture_images/numbers/number-09-nine.png` |
| medium | arousal | `number-10` | low | high (+0.15) | `gesture_images/numbers/number-10-ten.png` |
| medium | arousal | `prac-03` | low | high (+0.18) | `gesture_images/practical-and-functional/prac-03-hand-as-camera-framing.png` |
| medium | arousal | `prac-06` | low | high (+0.09) | `gesture_images/practical-and-functional/prac-06-shoulder-shrug-palms-up.png` |
| medium | arousal | `prac-07` | low | high (+0.11) | `gesture_images/practical-and-functional/prac-07-call-me-hand.png` |
| medium | arousal | `resp-02` | low | high (+0.09) | `gesture_images/respect-and-religious/resp-02-praying-hands.png` |
| medium | arousal | `rude-04` | low | high (+0.10) | `gesture_images/rude-and-offensive/rude-04-cuckoo-sign.png` |
| medium | arousal | `sil-03` | low | high (+0.13) | `gesture_images/silence-and-secrecy/sil-03-hand-cupped-behind-ear.png` |
| medium | arousal | `size-03` | low | high (+0.12) | `gesture_images/size-and-degree/size-03-hands-apart-for-height.png` |
| medium | arousal | `urg-06` | high | low (-0.19) | `gesture_images/urgency-and-attention/urg-06-rapid-come-on-beckoning.png` |
| medium | emotional_state | `conv-05` | positive | negative (-0.39) | `gesture_images/conversational-emphasis/conv-05-palms-together-pleading.png` |
| medium | emotional_state | `dir-06` | positive | negative (-0.32) | `gesture_images/directional-and-wayfinding/dir-06-beckoning-with-full-hand-palm-up.png` |
| medium | emotional_state | `love-03` | positive | negative (-0.27) | `gesture_images/love/love-03-two-hands-heart.png` |
| medium | emotional_state | `neg-07` | negative | positive (+0.18) | `gesture_images/negative-disapproval/neg-07-facepalm.png` |
| medium | emotional_state | `neg-09` | negative | positive (+0.18) | `gesture_images/negative-disapproval/neg-09-exasperated-shrug-with-hands-raised.png` |
| medium | emotional_state | `neg-10` | negative | positive (+0.20) | `gesture_images/negative-disapproval/neg-10-doh-facepalm.png` |
| medium | emotional_state | `urg-05` | negative | positive (+0.20) | `gesture_images/urgency-and-attention/urg-05-tapping-wrist-hurry-up.png` |
| medium | emotional_state | `wish-01` | positive | negative (-0.17) | `gesture_images/wishful-and-luck/wish-01-pinky-promise.png` |
| medium | emotional_state | `wish-02` | positive | negative (-0.16) | `gesture_images/wishful-and-luck/wish-02-cross-fingers-luck.png` |
| low | intent_similarity | `affirm-07` | Celebrate a triumph or express elation | 0.162 | `gesture_images/affirmative-and-positive/affirm-07-arms-raised-in-v-victory.png` |
| low | intent_similarity | `conv-03` | Enumerate key arguments or items in a discussion | 0.154 | `gesture_images/conversational-emphasis/conv-03-counting-off-points-on-fingers.png` |
| low | intent_similarity | `conv-07` | Question, emphasize, or express disbelief about what was sai | 0.154 | `gesture_images/conversational-emphasis/conv-07-italian-pinched-fingers.png` |
| low | intent_similarity | `creative-01` | Estimate or communicate the approximate size of an object | 0.165 | `gesture_images/creative/creative-01-about-yea-big-framing.png` |
| low | intent_similarity | `dir-06` | Warmly invite someone to approach | 0.166 | `gesture_images/directional-and-wayfinding/dir-06-beckoning-with-full-hand-palm-up.png` |
| low | intent_similarity | `greet-07` | Show formal respect or acknowledge authority | 0.145 | `gesture_images/greetings-and-farewells/greet-07-salute.png` |
| low | intent_similarity | `prac-08` | Indicate working on a computer, writing, or data entry | 0.128 | `gesture_images/practical-and-functional/prac-08-typing-keyboard-air.png` |
| low | intent_similarity | `sport-01` | Call an official timeout during a sporting event | 0.169 | `gesture_images/sports-and-tanting/sport-01-timeout-t-shape.png` |
| low | intent_similarity | `team-02` | Request a pause or timeout during group activity | 0.167 | `gesture_images/team-building/team-02-timeout-t-shape.png` |

## Notes

- **number_of_people** -- 11 finding(s) across 99 gesture(s) considered. Detected face count (uniface) is a proxy for the curated participant-count bucket (single -> 1, '2 person' -> 2, '3 or more' -> 3+). It fails on hand-first crops in both directions: zero faces does not mean zero people (no face is visible in a hand-only framing), and a multi-person scene often yields only one detected face (e.g. a high-five or a hug). Read a disagreement as a proxy limitation unless the image genuinely shows a different number of people than faces detected.
- **body_parts** -- 0 finding(s) across 96 gesture(s) considered. MediaPipe hand detection is checked only in the claims-a-hand-but-none-detected direction (the reverse -- a detected hand the curator did not list -- is not checked, since curated body_parts can legitimately omit incidental parts). Zero findings here is a genuine result: MediaPipe agreed with all 96 curated hand claims evaluated in this run, not a sign the check did not run.
- **emotional_state** -- 9 finding(s) across 50 gesture(s) considered. HSEmotion facial valence is a proxy for the curated emotional_state label; a finding fires only when the predicted sign is confident (|valence| >= 0.15). In this dataset facial affect is a weak proxy for gesture affect: emotion is largely carried by hands and posture rather than facial expression, so most detected faces read as Neutral regardless of the curated label. A contradiction is informative only when the detected face is visibly expressive; it is not evidence the curated label is wrong on its own.
- **arousal** -- 20 finding(s) across 99 gesture(s) considered. Predicted arousal terciles (low/medium/high, ranked within this run's scored gestures) are compared against the curated low/medium/high bucket; only opposite-extreme disagreements (curated low vs. predicted high, or vice versa) are flagged -- a low/medium or medium/high difference is too coarse a boundary to be informative. Terciles are relative to this run, not an absolute arousal scale.
- **intent_similarity** -- 9 finding(s) across 99 gesture(s) considered. CLIP image-to-intent cosine similarity is reported for the bottom decile of this run. CLIP has a well-known image-text modality gap, so raw scores are only meaningful relative to each other within this dataset, not as an absolute quality threshold -- a low score flags a candidate for human review, not a confirmed mislabel.

## Dataset notes

- `sport-01`, `team-02` share a byte-identical source image (the same gesture cross-listed across categories); they share this finding because their embeddings are identical -- a dataset property, not an anomaly.
