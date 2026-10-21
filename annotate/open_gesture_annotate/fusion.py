"""Image + speech fusion: does adding the voice channel help, and does it fail safe?

Reads embeddings.json (image), valence_arousal.json (face) and speech.json
(transcripts at several noise levels) and asks three questions per noise level:

1. Intent -- can each gesture's own curated `intent` be picked out of all 99,
   from the image alone, the transcript alone, and both fused?
2. Emotion -- does face valence, transcript valence, or their fusion separate
   curated-positive from curated-negative gestures?
3. Failure -- as noise rises and transcripts go wrong, does fusion degrade
   towards image-only, or below it?

Fusion is late and fits nothing. Image->intent and transcript->intent cosines
live on different scales (CLIP's image-text modality gap), so each query's 99
scores are z-scored per channel and combined two ways: equally, and weighted by
Whisper's own per-token confidence in the transcript. Neither weight is tuned:
with 99 gestures, tuning one on the same data would report the tuning, not the
method.

Like report.py, this module reports; it never writes manifest.json.
"""

from __future__ import annotations

import json
from math import comb
from pathlib import Path

import numpy as np

from open_gesture_annotate.backends.speech_whisper import CONDITIONS, _words, word_error_rate
from open_gesture_annotate.io import load_manifest

STOPWORDS = frozenset(
    "a an the and or but to of in on at for with by from is are was were be been it its "
    "this that these those i you he she we they me my your our their them his her do does "
    "did not no so as up out about just very can will would should could have has had "
    "what who how which there here then than too all any some".split()
)


def zscore(v: np.ndarray) -> np.ndarray:
    sd = v.std()
    return (v - v.mean()) / sd if sd > 0 else np.zeros_like(v)


def fuse(image_scores: np.ndarray, text_scores: np.ndarray | None,
         weight: float | None = 1.0) -> np.ndarray:
    """Weighted mean of per-channel z-scores: (z_image + w * z_text) / (1 + w).

    w = 1 is plain equal weighting. No transcript (a silent gesture, or Whisper
    heard nothing) means image alone.

    What the measurement on this dataset showed (fusion_report.md, paired tests):
    noise-garbled speech is NOT the danger it looks like. At -5 dB, with 74% word
    errors, equal-weight fusion is statistically indistinguishable from
    image-only -- garbled transcripts become semantically diffuse and stop
    pointing anywhere in particular. The consistent danger is fluent speech about
    something else: the shuffled-transcript control loses top-5 hits against
    image-only at every noise level (significant at 0 dB, a trend elsewhere).
    That is a bystander, not noise, and no STT confidence can flag it, because
    Whisper is rightly confident about clearly heard speech.

    Passing Whisper's per-token confidence as `weight` (stt_confidence() in the
    speech backend) fits nothing to the labels, but it measured no better than
    w = 1 at any noise level: the confidence tracks errors (Spearman -0.53) yet
    is also low for short, correct answers like "Ten.", underweighting them.
    """
    if text_scores is None or weight is None:
        return zscore(image_scores)
    return (zscore(image_scores) + weight * zscore(text_scores)) / (1.0 + weight)


def own_rank(scores: np.ndarray, own: int) -> int:
    """1-based rank of the query's own candidate; ties count against it.

    Conservative on purpose: `>=` (which counts the candidate itself, hence no
    +1) ranks a tie below its rivals, so identical scores can never inflate a
    retrieval number.
    """
    return int((scores >= scores[own]).sum())


def retrieval(ranks: list[int]) -> dict:
    if not ranks:
        return {"n": 0, "top1": None, "top5": None, "median_rank": None}
    r = np.asarray(ranks)
    return {"n": len(r), "top1": float((r <= 1).mean()), "top5": float((r <= 5).mean()),
            "median_rank": float(np.median(r))}


def auc(pos: list[float], neg: list[float]) -> float | None:
    """P(random positive > random negative), ties counted half (Mann-Whitney U / n1n2)."""
    if not pos or not neg:
        return None
    p, n = np.asarray(pos)[:, None], np.asarray(neg)[None, :]
    return float(((p > n).sum() + 0.5 * (p == n).sum()) / (p.size * n.size))


def sign_test(wins: int, losses: int) -> float:
    """Two-sided exact sign test on paired outcomes (ties dropped), p = 0.5.

    For per-gesture top-5 hits this is McNemar's exact test. With 99 gestures a
    3-point gap is 3 gestures; this says whether that could be chance.
    """
    n = wins + losses
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(wins, losses) + 1)) / 2 ** n)


def _paired(a_ranks: list[int], b_ranks: list[int]) -> dict:
    """Is method a better than method b on the same gestures?"""
    a, b = np.asarray(a_ranks), np.asarray(b_ranks)
    hit_a, hit_b = a <= 5, b <= 5
    w5, l5 = int((hit_a & ~hit_b).sum()), int((~hit_a & hit_b).sum())
    wr, lr = int((a < b).sum()), int((a > b).sum())
    return {"top5": {"wins": w5, "losses": l5, "p": sign_test(w5, l5)},
            "rank": {"wins": wr, "losses": lr, "p": sign_test(wr, lr)}}


def _average_ranks(values) -> np.ndarray:
    a = np.asarray(values, dtype=np.float64)
    ranks = np.empty(len(a))
    ranks[a.argsort(kind="stable")] = np.arange(len(a))
    for v in np.unique(a):  # ties share their mean rank; WER has many exact 0.0s
        ranks[a == v] = ranks[a == v].mean()
    return ranks


def spearman(x, y) -> float | None:
    """Rank correlation, tie-aware. Used to ask whether Whisper knows when it is wrong."""
    if len(x) < 3:
        return None
    rx, ry = _average_ranks(x), _average_ranks(y)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def content_overlap(a: str, b: str) -> float:
    """Jaccard similarity of content words -- the circularity measure."""
    wa, wb = set(_words(a)) - STOPWORDS, set(_words(b)) - STOPWORDS
    return len(wa & wb) / len(wa | wb) if wa | wb else 0.0


def _ok(path: Path) -> dict:
    if not path.is_file():
        return {}
    records = json.loads(path.read_text(encoding="utf-8"))["records"]
    return {k: v for k, v in records.items() if v.get("status") == "ok"}


def _unit(rows) -> np.ndarray:
    m = np.asarray(rows, dtype=np.float64)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def _emotion(source: dict, manifest: dict, subset=None) -> dict:
    keys = [k for k in source if subset is None or k in subset]
    by = {s: [source[k] for k in keys if manifest[k].get("emotional_state") == s]
          for s in ("positive", "neutral", "negative")}
    return {"n": len(keys), "auc": auc(by["positive"], by["negative"]),
            "means": {s: (float(np.mean(v)) if v else None) for s, v in by.items()}}


def evaluate(root: Path, out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    manifest = {g.id: g.raw for g in load_manifest(root)}
    emb, va, speech = (_ok(out_dir / f) for f in
                       ("embeddings.json", "valence_arousal.json", "speech.json"))
    ids = sorted(set(emb) & set(speech))
    if not ids:
        return {"n": 0}

    candidates = _unit([emb[i]["intent"] for i in ids])
    image_scores = _unit([emb[i]["image"] for i in ids]) @ candidates.T
    rng = np.random.default_rng(0)
    speaking = [i for i in ids if speech[i]["speech_present"]]

    results = {"n": len(ids), "n_speaking": len(speaking), "conditions": {}}
    for cond, snr in CONDITIONS.items():
        text_scores, confidence = {}, {}
        for i in ids:
            rec = speech[i]["conditions"][cond]
            text_scores[i] = np.asarray(rec["text_embedding"]) @ candidates.T \
                if rec["text_embedding"] else None
            confidence[i] = rec.get("stt_confidence")

        # Null control: same transcripts, pairing broken among the speakers.
        perm = dict(zip(speaking, rng.permutation(speaking)))
        img_r, txt_r, fused_r, conf_r, shuf_r = [], [], [], [], []
        for row, gid in enumerate(ids):
            img_r.append(own_rank(image_scores[row], row))
            if text_scores[gid] is not None:
                txt_r.append(own_rank(text_scores[gid], row))
            fused_r.append(own_rank(fuse(image_scores[row], text_scores[gid]), row))
            conf_r.append(own_rank(fuse(image_scores[row], text_scores[gid], confidence[gid]), row))
            shuf_r.append(own_rank(fuse(image_scores[row], text_scores.get(perm.get(gid))), row))

        # Emotion: face valence only where HSEmotion saw a cropped face -- its
        # full-image fallback is, per that backend, near-meaningless for valence.
        face = {i: va[i]["valence"] for i in ids
                if i in va and va[i].get("face_source") == "uniface"}
        text = {i: speech[i]["conditions"][cond]["text_valence"] for i in ids
                if speech[i]["conditions"][cond]["text_valence"] is not None}
        fused_v = {i: float(np.mean([s[i] for s in (face, text) if i in s]))
                   for i in ids if i in face or i in text}
        fused_cv = {}
        for i in fused_v:
            w = confidence[i] if i in text else None
            if i in face and w is not None:
                fused_cv[i] = (face[i] + w * text[i]) / (1.0 + w)
            else:
                fused_cv[i] = face.get(i, text.get(i))
        common = set(face) & set(text)

        # Recomputed from the stored text rather than read from the sidecar, so the
        # report always reflects the current scoring rules.
        wers = [word_error_rate(speech[i]["utterance"], speech[i]["conditions"][cond]["transcript"])
                for i in speaking]
        confs = [confidence[i] for i in speaking if confidence[i] is not None]
        results["conditions"][cond] = {
            "snr_db": snr,
            "mean_wer": float(np.mean(wers)) if wers else None,
            "mean_confidence": float(np.mean(confs)) if confs else None,
            "intent": {"image": retrieval(img_r), "text": retrieval(txt_r),
                       "fused": retrieval(fused_r), "fused_confidence": retrieval(conf_r),
                       "fused_shuffled": retrieval(shuf_r)},
            "paired": {"equal_vs_image": _paired(fused_r, img_r),
                       "confidence_vs_image": _paired(conf_r, img_r),
                       "confidence_vs_equal": _paired(conf_r, fused_r),
                       "shuffled_vs_image": _paired(shuf_r, img_r)},
            "emotion": {
                "face": _emotion(face, manifest),
                "text": _emotion(text, manifest),
                "fused": _emotion(fused_v, manifest),
                "fused_confidence": _emotion(fused_cv, manifest),
                "common": {"n": len(common),
                           "face": _emotion(face, manifest, common)["auc"],
                           "text": _emotion(text, manifest, common)["auc"],
                           "fused": _emotion(fused_v, manifest, common)["auc"],
                           "fused_confidence": _emotion(fused_cv, manifest, common)["auc"]},
            },
        }

    pairs = [(c["stt_confidence"], word_error_rate(speech[i]["utterance"], c["transcript"]))
             for i in speaking for c in speech[i]["conditions"].values()
             if c.get("stt_confidence") is not None]
    results["confidence_vs_wer"] = {"n": len(pairs),
                                    "spearman": spearman(*zip(*pairs)) if pairs else None}

    overlaps = sorted(((content_overlap(speech[i]["utterance"], manifest[i].get("intent", "")), i)
                       for i in speaking), reverse=True)
    results["circularity"] = {
        "mean_overlap": float(np.mean([o for o, _ in overlaps])) if overlaps else 0.0,
        "with_any_overlap": sum(1 for o, _ in overlaps if o > 0),
        "top": [{"id": i, "overlap": o, "utterance": speech[i]["utterance"]}
                for o, i in overlaps[:5]],
    }
    return results


def _pct(x) -> str:
    return "–" if x is None else f"{x:.0%}"


def _num(x, fmt: str = ".2f") -> str:
    return "–" if x is None else format(x, fmt)


def _retrieval_cell(x: dict) -> str:
    return f"{_pct(x['top5'])} ({_num(x['median_rank'], '.0f')})"


def _paired_cell(x: dict) -> str:
    t, r = x["top5"], x["rank"]
    return (f"top-5 {t['wins']}/{t['losses']} (p={t['p']:.2g}); "
            f"rank {r['wins']}/{r['losses']} (p={r['p']:.2g})")


def _auc_cell(x: dict) -> str:
    return f"{_num(x['auc'])} ({x['n']})"


def write_fusion_report(root: Path, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    r = evaluate(root, out_dir)
    lines = ["# Image + Speech Fusion Report", ""]
    if not r.get("n"):
        lines += ["No overlapping embeddings.json and speech.json records; run "
                  "`og-annotate run --backends embed,va,speech` first.", ""]
    else:
        conds = r["conditions"]
        lines += [
            "Does adding the voice channel improve intent and emotion recognition over the "
            "image alone, and what happens as speech degrades? Speech here is **synthesised** "
            "(utterance -> SpeechT5 -> white noise -> Whisper), so absolute numbers describe "
            "one clean synthetic voice; the comparison between channels, and its trend with "
            "noise, is the result. Fusion combines per-query z-scores either equally or "
            "weighted by Whisper's own confidence in each transcript; neither weight was "
            "tuned on this data.",
            "",
            f"{r['n']} gestures; {r['n_speaking']} with speech "
            f"({r['n'] - r['n_speaking']} performed silently -> image-only by design).",
            "",
            "## Speech-to-text",
            "",
            "| Condition | SNR | Mean WER | Mean Whisper confidence |",
            "|---|---|---|---|",
        ]
        for c, d in conds.items():
            snr = "clean" if d["snr_db"] is None else f"{d['snr_db']:+g} dB"
            lines.append(f"| {c} | {snr} | {_pct(d['mean_wer'])} "
                         f"| {_num(d.get('mean_confidence'))} |")
        cw = r.get("confidence_vs_wer", {})
        lines += [
            "",
            "Does Whisper know when it is wrong? Spearman correlation between its confidence "
            f"and the transcript's WER, over all {cw.get('n', 0)} transcripts: "
            f"**{_num(cw.get('spearman'), '+.2f')}** (negative means confidence falls as "
            "errors rise -- the property confidence weighting relies on).",
        ]
        lines += [
            "",
            "## Intent: own `intent` retrieved among all gestures",
            "",
            f"Top-5 accuracy (median rank). Chance top-5 is {5 / r['n']:.0%}. The shuffled "
            "control pairs each image with another gesture's transcript; it should fall to "
            "or below image-only, or the gain is not coming from matched speech.",
            "",
            "| Condition | Image only | Speech only | Fused, equal | Fused, confidence "
            "| Fused, shuffled speech |",
            "|---|---|---|---|---|---|",
        ]
        for c, d in conds.items():
            i = d["intent"]
            lines.append(f"| {c} | {_retrieval_cell(i['image'])} | {_retrieval_cell(i['text'])} "
                         f"| {_retrieval_cell(i['fused'])} "
                         f"| {_retrieval_cell(i['fused_confidence'])} "
                         f"| {_retrieval_cell(i['fused_shuffled'])} |")
        lines += [
            "",
            "### Which differences are real?",
            "",
            "With 99 gestures a few points is a few gestures, so each comparison is a paired "
            "exact sign test on the same gestures: top-5 hits gained/lost (McNemar), then "
            "rank improved/worsened. Read p < 0.05 as a real difference; anything else as "
            "no evidence either way.",
            "",
            "| Condition | Equal vs image | Confidence vs image | Confidence vs equal "
            "| Shuffled vs image |",
            "|---|---|---|---|---|",
        ]
        for c, d in conds.items():
            pr = d["paired"]
            lines.append(f"| {c} | {_paired_cell(pr['equal_vs_image'])} "
                         f"| {_paired_cell(pr['confidence_vs_image'])} "
                         f"| {_paired_cell(pr['confidence_vs_equal'])} "
                         f"| {_paired_cell(pr['shuffled_vs_image'])} |")
        lines += [
            "",
            "## Emotion: curated positive vs negative, by valence",
            "",
            "AUC (chance 0.50) with coverage n. Face valence exists only where a face was "
            "cropped; transcript valence only where speech was recognised. The last column "
            "compares every channel on the same gestures -- the fair comparison, since "
            "coverage differs. Caveat on the speech column: each utterance was written "
            "knowing its gesture, so its sentiment tends to match the gesture's obvious "
            "valence (\"Screw you!\" for the middle finger). The speech AUC therefore shows "
            "the pipeline carrying sentiment intact through TTS, noise and STT -- not how "
            "often real people's words match their gesture's affect. That needs recorded "
            "speech.",
            "",
            "| Condition | Face (n) | Speech (n) | Fused, equal (n) | Fused, confidence (n) "
            "| Same set: face / speech / equal / confidence |",
            "|---|---|---|---|---|---|",
        ]
        for c, d in conds.items():
            e, cm = d["emotion"], d["emotion"]["common"]
            lines.append(f"| {c} | {_auc_cell(e['face'])} | {_auc_cell(e['text'])} "
                         f"| {_auc_cell(e['fused'])} | {_auc_cell(e['fused_confidence'])} "
                         f"| {_num(cm['face'])} / {_num(cm['text'])} / {_num(cm['fused'])} / "
                         f"{_num(cm['fused_confidence'])} (n={cm['n']}) |")
        circ = r["circularity"]
        lines += [
            "",
            "## Circularity check",
            "",
            "Utterances were written from gesture names without reading the curated `intent` "
            "text, so the speech channel is not a paraphrase of its own label. Residual "
            f"content-word overlap (Jaccard) averages {circ['mean_overlap']:.3f}; "
            f"{circ['with_any_overlap']} of {r['n_speaking']} utterances share any content "
            "word with their intent. Highest:",
            "",
        ]
        lines += [f"- `{t['id']}` {t['overlap']:.2f} -- \"{t['utterance']}\"" for t in circ["top"]]
        lines.append("")

    path = out_dir / "fusion_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
