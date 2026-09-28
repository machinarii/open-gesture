"""Speech round-trip helpers and image + speech fusion, tested without any model.

speech_whisper imports its models lazily (inside SpeechBackend._load), so the
signal processing, scoring and fusion logic here all run on a plain checkout.
"""

import json

import numpy as np
import pytest

from open_gesture_annotate.backends.speech_whisper import (
    CONDITIONS,
    SpeechBackend,
    add_noise,
    load_utterances,
    signed_valence,
    stt_confidence,
    word_error_rate,
)
from open_gesture_annotate.fusion import (
    auc,
    content_overlap,
    evaluate,
    fuse,
    own_rank,
    sign_test,
    spearman,
    write_fusion_report,
    zscore,
)
from open_gesture_annotate.io import Gesture, load_manifest, repo_root

# --- speech round-trip helpers ----------------------------------------------


def test_backend_identity():
    b = SpeechBackend()
    assert b.name == "speech-roundtrip"
    assert b.sidecar == "speech.json"


@pytest.mark.parametrize("snr_db", [10.0, 0.0, -5.0])
def test_noise_is_mixed_at_the_requested_snr(snr_db):
    wav = np.sin(np.linspace(0, 2000 * np.pi, 160_000)).astype(np.float32)
    noise = add_noise(wav, snr_db, np.random.default_rng(0)).astype(np.float64) - wav
    measured = 10 * np.log10(np.mean(wav.astype(np.float64) ** 2) / np.mean(noise ** 2))
    assert measured == pytest.approx(snr_db, abs=0.1)


def test_clean_condition_leaves_audio_untouched():
    wav = np.ones(10, np.float32)
    assert add_noise(wav, None, np.random.default_rng(0)) is wav


def test_conditions_include_clean_and_run_below_zero_db():
    """The sweep must reach the regime where STT actively misleads, not stop at
    the comfortable end (see the speech_whisper docstring).
    """
    assert CONDITIONS["clean"] is None
    assert min(v for v in CONDITIONS.values() if v is not None) < 0


def test_wer_ignores_case_and_punctuation():
    assert word_error_rate("Stop! Don't come any closer!", "stop don't come any closer") == 0.0


def test_wer_counts_one_substitution_in_five_words():
    assert word_error_rate("nice work that looks great", "nice work that looks good") == 0.2


def test_wer_scores_the_meaning_inverting_mishearing_from_the_spike():
    """don't->still and come->coming are substitutions, "any" is deleted: 3 of 6."""
    wer = word_error_rate("Stop, don't come any closer please", "Stop, still coming closer please")
    assert wer == pytest.approx(3 / 6)


def test_wer_does_not_count_digit_formatting_as_an_error():
    """Whisper wrote all of these for correctly heard speech in the real run."""
    assert word_error_rate("Ten.", "10.") == 0.0
    assert word_error_rate("One, two, three, four, five.", "1, 2, 3, 4, 5") == 0.0
    assert word_error_rate("One, two, three, team!", "1-2-3 team.") == 0.0
    assert word_error_rate("Two.", "Two. Two.") == 1.0  # a hallucinated repeat still counts


def test_wer_edge_cases():
    assert word_error_rate("", "") == 0.0
    assert word_error_rate("", "hallucinated words") == 1.0
    assert word_error_rate("one two", "") == 1.0


def test_stt_confidence_is_the_token_weighted_geometric_mean_probability():
    # One 3-token segment at log-prob -0.1 and one 1-token segment at -1.3:
    # token-weighted mean -0.4, so confidence exp(-0.4).
    assert stt_confidence([(-0.1, 3), (-1.3, 1)]) == pytest.approx(np.exp(-0.4))
    assert stt_confidence([]) is None
    assert 0 < stt_confidence([(-5.0, 2)]) < stt_confidence([(-0.05, 2)]) <= 1


def test_signed_valence_spans_minus_one_to_one():
    assert signed_valence([{"label": "POSITIVE", "score": 1.0},
                           {"label": "NEGATIVE", "score": 0.0}]) == 1.0
    assert signed_valence([{"label": "NEGATIVE", "score": 0.75},
                           {"label": "POSITIVE", "score": 0.25}]) == -0.5


def test_utterances_cover_the_manifest_exactly():
    """Every gesture gets a speech record, and no utterance names a gesture that
    does not exist -- a typo'd id would silently fall back to silence.
    """
    assert set(load_utterances()) == {g.id for g in load_manifest(repo_root())}


def test_silent_gesture_records_no_speech_without_loading_models():
    b = SpeechBackend()
    b._utterances = {"g": ""}
    rec = b.annotate(np.zeros((2, 2, 3), np.uint8),
                     Gesture(id="g", index=1, name="G", category="C", file="g.png", raw={}))
    assert rec["status"] == "ok" and rec["speech_present"] is False
    assert set(rec["conditions"]) == set(CONDITIONS)
    assert all(c["transcript"] == "" and c["text_embedding"] is None
               for c in rec["conditions"].values())
    assert b._models is None  # never loaded TTS or STT for a silent gesture


# --- fusion -----------------------------------------------------------------


def test_fuse_without_speech_is_exactly_image_only():
    img = np.array([0.1, 0.5, 0.2])
    assert np.array_equal(fuse(img, None), zscore(img))


def test_fuse_is_invariant_to_each_channels_scale():
    """The whole reason for z-scoring: image-text and text-text cosines live on
    different scales, and neither channel may dominate by scale alone.
    """
    rng = np.random.default_rng(0)
    img, txt = rng.normal(size=20), rng.normal(size=20)
    assert np.allclose(fuse(img, txt), fuse(img * 0.01, txt * 100 + 7))


def test_confidence_weight_interpolates_between_image_only_and_equal_fusion():
    """w -> 0 must converge to image-only -- the fail-safe property equal
    weighting lacks -- and w = 1 must reproduce equal fusion exactly.
    """
    rng = np.random.default_rng(1)
    img, txt = rng.normal(size=10), rng.normal(size=10)
    assert np.allclose(fuse(img, txt, 1.0), fuse(img, txt))
    assert np.allclose(fuse(img, txt, 1e-9), zscore(img))
    assert np.array_equal(fuse(img, txt, None), zscore(img))


def test_a_doubtful_wrong_transcript_cannot_outvote_a_confident_image():
    """Mirror-image scores: equal weighting ties the right and wrong candidates
    (a tie ranks against the truth), while a low-confidence transcript cannot.
    """
    img = np.array([0.9, 0.1, 0.2, 0.15])   # image is sure: candidate 0
    wrong = np.array([0.1, 0.9, 0.2, 0.15])  # noisy transcript points at 1
    assert own_rank(fuse(img, wrong), 0) == 2
    assert own_rank(fuse(img, wrong, 0.1), 0) == 1


def test_sign_test_matches_the_exact_binomial():
    # Values scipy.stats.binomtest gave for the real clean-speech comparison.
    assert sign_test(15, 4) == pytest.approx(0.01921, abs=1e-4)
    assert sign_test(9, 8) == 1.0
    assert sign_test(0, 0) == 1.0
    assert sign_test(4, 15) == sign_test(15, 4)  # two-sided


def test_spearman():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([0, 0, 0, 1], [1, 2, 3, 4]) is not None  # ties handled
    assert spearman([1, 2], [1, 2]) is None


def test_agreeing_speech_lifts_the_right_candidate():
    img = np.array([0.30, 0.31, 0.29, 0.28])  # image is unsure; truth is 0, ranked 2nd
    txt = np.array([0.90, 0.10, 0.20, 0.15])  # speech is confident about 0
    assert own_rank(img, 0) == 2
    assert own_rank(fuse(img, txt), 0) == 1


def test_own_rank_counts_ties_against_the_candidate():
    assert own_rank(np.array([0.5, 0.5, 0.1]), 0) == 2
    assert own_rank(np.array([0.9, 0.5, 0.1]), 0) == 1


def test_auc():
    assert auc([3, 4], [1, 2]) == 1.0
    assert auc([1, 2], [3, 4]) == 0.0
    assert auc([1], [1]) == 0.5
    assert auc([], [1]) is None


def test_content_overlap_ignores_stopwords():
    # "there" and "the" are stopwords: {stop, right} vs {stop, vehicle}.
    assert content_overlap("Stop right there", "stop the vehicle") == pytest.approx(1 / 3)
    assert content_overlap("the and of", "a an the") == 0.0


def _write(path, records):
    path.write_text(json.dumps({"_backend": {"name": "x", "version": "0", "run_at": "t"},
                                "records": records}), encoding="utf-8")


@pytest.fixture
def fusion_tree(tmp_path):
    """Three gestures in a 3-D space. The image cannot tell a from b; speech is
    decisive for both; c is performed silently. The face misreads b as happier
    than a, and speech corrects it.
    """
    states = [("a", "positive"), ("b", "negative"), ("c", "neutral")]
    gestures = [{"id": gid, "index": n, "name": gid, "category": "C", "file": f"{gid}.png",
                 "intent": f"intent {gid}", "emotional_state": state}
                for n, (gid, state) in enumerate(states)]
    (tmp_path / "manifest.json").write_text(json.dumps({"gestures": gestures}))
    out = tmp_path / "annotations"
    out.mkdir()
    axis = {"a": [1, 0, 0], "b": [0, 1, 0], "c": [0, 0, 1]}
    blur = [1, 1, 0.2]
    _write(out / "embeddings.json", {
        g: {"status": "ok", "intent": axis[g], "image": axis["c"] if g == "c" else blur}
        for g in axis
    })
    _write(out / "valence_arousal.json", {
        "a": {"status": "ok", "valence": 0.1, "face_source": "uniface"},
        "b": {"status": "ok", "valence": 0.2, "face_source": "uniface"},
        "c": {"status": "ok", "valence": 0.0, "face_source": "full-image"},
    })

    def spoken(gid, valence):
        return {"snr_db": None, "transcript": gid, "wer": 0.0, "stt_confidence": 0.9,
                "text_valence": valence, "text_embedding": axis[gid]}

    silent = {"snr_db": None, "transcript": "", "wer": None, "stt_confidence": None,
              "text_valence": None, "text_embedding": None}
    _write(out / "speech.json", {
        "a": {"status": "ok", "utterance": "yay", "speech_present": True,
              "conditions": {c: spoken("a", 0.9) for c in CONDITIONS}},
        "b": {"status": "ok", "utterance": "boo", "speech_present": True,
              "conditions": {c: spoken("b", -0.9) for c in CONDITIONS}},
        "c": {"status": "ok", "utterance": "", "speech_present": False,
              "conditions": {c: silent for c in CONDITIONS}},
    })
    return tmp_path, out


def test_evaluate_fuses_speech_and_falls_back_for_silence(fusion_tree):
    root, out = fusion_tree
    r = evaluate(root, out)
    clean = r["conditions"]["clean"]
    assert r["n"] == 3 and r["n_speaking"] == 2
    assert clean["intent"]["image"]["top1"] == pytest.approx(1 / 3)  # a/b tie; only c right
    assert clean["intent"]["fused"]["top1"] == 1.0   # speech resolves a vs b; c via fallback
    assert clean["intent"]["fused_confidence"]["top1"] == 1.0
    paired = clean["paired"]["equal_vs_image"]["rank"]
    assert (paired["wins"], paired["losses"]) == (2, 0)  # a and b improve; c unchanged
    assert clean["intent"]["text"]["n"] == 2         # silent c has no speech score
    assert clean["emotion"]["face"]["auc"] == 0.0    # face ranks b above a
    assert clean["emotion"]["fused"]["auc"] == 1.0   # speech corrects it
    assert clean["emotion"]["face"]["n"] == 2        # c's full-image valence excluded


def test_fusion_report_is_written(fusion_tree):
    root, out = fusion_tree
    path = write_fusion_report(root, out)
    text = path.read_text()
    assert path.name == "fusion_report.md"
    assert "Fused" in text and "Circularity" in text


def test_fusion_report_without_speech_explains_what_to_run(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"gestures": []}))
    text = write_fusion_report(tmp_path, tmp_path / "annotations").read_text()
    assert "og-annotate run" in text
