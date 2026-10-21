# Image + Speech Fusion Report

Does adding the voice channel improve intent and emotion recognition over the image alone, and what happens as speech degrades? Speech here is **synthesised** (utterance -> SpeechT5 -> white noise -> Whisper), so absolute numbers describe one clean synthetic voice; the comparison between channels, and its trend with noise, is the result. Fusion combines per-query z-scores either equally or weighted by Whisper's own confidence in each transcript; neither weight was tuned on this data.

99 gestures; 97 with speech (2 performed silently -> image-only by design).

## Speech-to-text

| Condition | SNR | Mean WER | Mean Whisper confidence |
|---|---|---|---|
| clean | clean | 8% | 0.59 |
| snr10 | +10 dB | 15% | 0.59 |
| snr0 | +0 dB | 39% | 0.55 |
| snr-5 | -5 dB | 74% | 0.49 |

Does Whisper know when it is wrong? Spearman correlation between its confidence and the transcript's WER, over all 388 transcripts: **-0.53** (negative means confidence falls as errors rise -- the property confidence weighting relies on).

## Intent: own `intent` retrieved among all gestures

Top-5 accuracy (median rank). Chance top-5 is 5%. The shuffled control pairs each image with another gesture's transcript; it should fall to or below image-only, or the gain is not coming from matched speech.

| Condition | Image only | Speech only | Fused, equal | Fused, confidence | Fused, shuffled speech |
|---|---|---|---|---|---|
| clean | 23% (21) | 34% (18) | 34% (13) | 33% (17) | 15% (33) |
| snr10 | 23% (21) | 29% (21) | 33% (14) | 31% (19) | 16% (30) |
| snr0 | 23% (21) | 27% (24) | 32% (20) | 33% (19) | 14% (31) |
| snr-5 | 23% (21) | 18% (35) | 24% (25) | 27% (25) | 16% (33) |

### Which differences are real?

With 99 gestures a few points is a few gestures, so each comparison is a paired exact sign test on the same gestures: top-5 hits gained/lost (McNemar), then rank improved/worsened. Read p < 0.05 as a real difference; anything else as no evidence either way.

| Condition | Equal vs image | Confidence vs image | Confidence vs equal | Shuffled vs image |
|---|---|---|---|---|
| clean | top-5 15/4 (p=0.019); rank 57/32 (p=0.011) | top-5 13/3 (p=0.021); rank 54/29 (p=0.008) | top-5 1/2 (p=1); rank 32/41 (p=0.35) | top-5 4/12 (p=0.077); rank 39/54 (p=0.15) |
| snr10 | top-5 14/4 (p=0.031); rank 54/32 (p=0.023) | top-5 11/3 (p=0.057); rank 56/29 (p=0.0045) | top-5 2/4 (p=0.69); rank 34/41 (p=0.49) | top-5 4/11 (p=0.12); rank 41/52 (p=0.3) |
| snr0 | top-5 14/5 (p=0.064); rank 50/38 (p=0.24) | top-5 13/3 (p=0.021); rank 53/35 (p=0.069) | top-5 2/1 (p=1); rank 36/39 (p=0.82) | top-5 3/12 (p=0.035); rank 35/57 (p=0.028) |
| snr-5 | top-5 9/8 (p=1); rank 50/40 (p=0.34) | top-5 7/3 (p=0.34); rank 48/35 (p=0.19) | top-5 5/2 (p=0.45); rank 39/41 (p=0.91) | top-5 3/10 (p=0.092); rank 38/52 (p=0.17) |

## Emotion: curated positive vs negative, by valence

AUC (chance 0.50) with coverage n. Face valence exists only where a face was cropped; transcript valence only where speech was recognised. The last column compares every channel on the same gestures -- the fair comparison, since coverage differs. Caveat on the speech column: each utterance was written knowing its gesture, so its sentiment tends to match the gesture's obvious valence ("Screw you!" for the middle finger). The speech AUC therefore shows the pipeline carrying sentiment intact through TTS, noise and STT -- not how often real people's words match their gesture's affect. That needs recorded speech.

| Condition | Face (n) | Speech (n) | Fused, equal (n) | Fused, confidence (n) | Same set: face / speech / equal / confidence |
|---|---|---|---|---|---|
| clean | 0.74 (93) | 0.92 (97) | 0.94 (99) | 0.94 (99) | 0.74 / 0.92 / 0.94 / 0.93 (n=91) |
| snr10 | 0.74 (93) | 0.91 (97) | 0.94 (99) | 0.94 (99) | 0.74 / 0.91 / 0.94 / 0.94 (n=91) |
| snr0 | 0.74 (93) | 0.89 (97) | 0.92 (99) | 0.92 (99) | 0.74 / 0.89 / 0.92 / 0.92 (n=91) |
| snr-5 | 0.74 (93) | 0.85 (97) | 0.87 (99) | 0.88 (99) | 0.74 / 0.87 / 0.90 / 0.91 (n=91) |

## Circularity check

Utterances were written from gesture names without reading the curated `intent` text, so the speech channel is not a paraphrase of its own label. Residual content-word overlap (Jaccard) averages 0.069; 33 of 97 utterances share any content word with their intent. Highest:

- `number-10` 0.33 -- "Ten."
- `number-09` 0.33 -- "Nine."
- `number-08` 0.33 -- "Eight."
- `number-07` 0.33 -- "Seven."
- `number-06` 0.33 -- "Six."

