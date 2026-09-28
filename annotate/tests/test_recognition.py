"""Acceptance: does multimodal intent and emotion recognition work as the spec planned?

Runs against the real sidecars in annotations/, which are gitignored, so every
test skips on a fresh checkout. Run after `og-annotate run`:

    .venv/bin/python -m pytest tests/test_recognition.py -m slow

Two layers, mirroring docs/superpowers/specs/2026-08-25-multimodal-annotation-pipeline-design.md:

1. Conformance -- the outputs the spec promises exist and are wired as specified
   (HSEmotion crops from uniface's box; the exact 20 py-feat AUs; CLIP embeds
   both `intent` and `physical_description`).
2. Recognition -- the signals carry information, not just shape. Every
   threshold is "better than chance" at a conventional alpha (0.01 for
   retrieval, 0.05 for correlations), stated up front rather than fitted to a
   run: a bar tuned to today's numbers would pass by construction and catch
   nothing.

Deliberately NOT asserted: that facial emotion matches the curated label
gesture-by-gesture. The README's own finding is that facial affect is a weak
proxy for gesture affect, so the claim tested here is the weaker one the data
supports -- that valence and arousal move in the curated direction on average.
"""

import json

import numpy as np
import pytest

from open_gesture_annotate.io import load_manifest, repo_root

stats = pytest.importorskip("scipy.stats")

pytestmark = pytest.mark.slow

PLANNED_AUS = {
    "AU01", "AU02", "AU04", "AU05", "AU06", "AU07", "AU09", "AU10", "AU11", "AU12",
    "AU14", "AU15", "AU17", "AU20", "AU23", "AU24", "AU25", "AU26", "AU28", "AU43",
}
PLANNED_PYFEAT_EMOTIONS = {
    "anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral",
}
# uniface and HSEmotion spell the same classes differently.
CANONICAL_LABEL = {"happiness": "happy", "anger": "angry", "sadness": "sad"}

RETRIEVAL_ALPHA = 0.01
CORRELATION_ALPHA = 0.05


def _sidecar(name: str) -> dict:
    path = repo_root() / "annotations" / name
    if not path.is_file():
        pytest.skip(f"annotations/{name} absent; run og-annotate first")
    records = json.loads(path.read_text(encoding="utf-8"))["records"]
    return {k: v for k, v in records.items() if v.get("status") == "ok"}


@pytest.fixture(scope="module")
def manifest():
    return {g.id: g.raw for g in load_manifest(repo_root())}


@pytest.fixture(scope="module")
def faces():
    return _sidecar("faces.json")


@pytest.fixture(scope="module")
def va():
    return _sidecar("valence_arousal.json")


@pytest.fixture(scope="module")
def aus():
    return _sidecar("action_units.json")


@pytest.fixture(scope="module")
def embeddings():
    return _sidecar("embeddings.json")


def _with_face(faces: dict) -> set:
    return {k for k, v in faces.items() if v.get("face_count", 0) >= 1}


def _unit(rows) -> np.ndarray:
    m = np.asarray(rows, dtype=np.float64)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def _self_retrieval_ranks(queries: np.ndarray, keys: np.ndarray) -> np.ndarray:
    """1-based rank of each query's own key among all keys (ties count against)."""
    sims = queries @ keys.T
    return np.array([(sims[i] >= sims[i, i]).sum() for i in range(len(sims))])


# --- conformance ------------------------------------------------------------


def test_every_detected_face_carries_an_emotion_label(faces):
    unlabeled = [
        k for k in _with_face(faces)
        for f in faces[k]["faces"]
        if not (isinstance(f.get("emotion"), dict) and f["emotion"].get("label"))
    ]
    assert not unlabeled


def test_hsemotion_crops_from_uniface_exactly_when_a_face_exists(faces, va):
    """The spec requires reusing uniface's box rather than a second detector,
    and falling back to the full image only when there is no face.
    """
    with_face = _with_face(faces)
    wrong = [k for k, v in va.items() if (v["face_source"] == "uniface") != (k in with_face)]
    assert not wrong


def test_pyfeat_emits_exactly_the_planned_action_units_and_emotions(aus):
    """Exactly the 20 planned AUs and 7 emotions when py-feat finds a face;
    empty when it finds none (its own detector, so its faceless set differs
    from uniface's -- it misses the Facepalm, whose hand covers the face).
    """
    for gid, rec in aus.items():
        if rec["face_count"] >= 1:
            assert set(rec["action_units"]) == PLANNED_AUS, gid
            assert set(rec["emotions"]) == PLANNED_PYFEAT_EMOTIONS, gid
        else:
            assert rec["action_units"] == {} and rec["emotions"] == {}, gid


def test_clip_embeds_image_intent_and_description(embeddings):
    for gid, rec in embeddings.items():
        for field in ("image", "intent", "description"):
            assert len(rec[field]) == rec["dim"], (gid, field)


def test_stored_similarities_are_the_cosines_of_the_stored_embeddings(embeddings):
    """Guards the report's intent check against a sidecar whose similarity
    column has drifted from the vectors it claims to summarise.
    """
    ids = sorted(embeddings)
    image = _unit([embeddings[i]["image"] for i in ids])
    for field, column in (("intent", "similarity_intent"),
                          ("description", "similarity_description")):
        text = _unit([embeddings[i][field] for i in ids])
        recomputed = np.einsum("ij,ij->i", image, text)
        stored = np.array([embeddings[i][column] for i in ids])
        assert np.abs(recomputed - stored).max() < 1e-4, column


# --- intent recognition -----------------------------------------------------


@pytest.mark.parametrize("field", ["intent", "description"])
def test_each_image_retrieves_its_own_text_above_chance(embeddings, field):
    """Rank all N texts for each image; count how often its own text lands in
    the top 5. Chance is 5/N per image; a one-sided binomial test says whether
    the hit count could plausibly be chance.
    """
    ids = sorted(embeddings)
    ranks = _self_retrieval_ranks(
        _unit([embeddings[i]["image"] for i in ids]),
        _unit([embeddings[i][field] for i in ids]),
    )
    hits = int((ranks <= 5).sum())
    test = stats.binomtest(hits, n=len(ids), p=5 / len(ids), alternative="greater")
    assert test.pvalue < RETRIEVAL_ALPHA, (
        f"{field}: {hits}/{len(ids)} top-5 hits, p={test.pvalue:.3g}"
    )


def test_shuffled_texts_do_not_retrieve_above_chance(embeddings):
    """Null control for the test above: break the image-text pairing and the
    same measurement must fall back to chance, or the retrieval test is
    measuring something other than recognition.
    """
    ids = sorted(embeddings)
    image = _unit([embeddings[i]["image"] for i in ids])
    text = _unit([embeddings[i]["intent"] for i in ids])
    shuffled = text[np.random.default_rng(0).permutation(len(ids))]
    hits = int((_self_retrieval_ranks(image, shuffled) <= 5).sum())
    test = stats.binomtest(hits, n=len(ids), p=5 / len(ids), alternative="greater")
    assert test.pvalue >= RETRIEVAL_ALPHA


# --- emotion recognition ----------------------------------------------------


def _valence_by_state(faces, va, manifest) -> dict:
    with_face = _with_face(faces)
    groups = {"positive": [], "neutral": [], "negative": []}
    for gid, rec in va.items():
        state = manifest[gid].get("emotional_state")
        if gid in with_face and state in groups:
            groups[state].append(rec["valence"])
    return groups


def test_valence_orders_positive_over_neutral_over_negative(faces, va, manifest):
    """The PAD relationship the spec builds the valence check on."""
    groups = _valence_by_state(faces, va, manifest)
    means = {s: float(np.mean(v)) for s, v in groups.items()}
    assert means["positive"] > means["neutral"] > means["negative"], means


def test_valence_separates_positive_from_negative(faces, va, manifest):
    groups = _valence_by_state(faces, va, manifest)
    test = stats.mannwhitneyu(groups["positive"], groups["negative"], alternative="greater")
    assert test.pvalue < CORRELATION_ALPHA, f"p={test.pvalue:.3g}"


def test_predicted_arousal_rises_with_curated_arousal(faces, va, manifest):
    order = {"low": 0, "medium": 1, "high": 2}
    ks = [k for k in va if k in _with_face(faces) and manifest[k].get("arousal") in order]
    rho = stats.spearmanr([order[manifest[k]["arousal"]] for k in ks],
                          [va[k]["arousal"] for k in ks], alternative="greater")
    assert rho.pvalue < CORRELATION_ALPHA, f"rho={rho.statistic:+.3f} p={rho.pvalue:.3g}"


def test_independent_emotion_models_agree_on_happiness(faces, va, aus):
    """py-feat and HSEmotion are separate models; if both work, their
    happiness scores on the same faces should rise together.
    """
    ks = [k for k in va if k in _with_face(faces) and aus.get(k, {}).get("face_count", 0) >= 1]
    rho = stats.spearmanr([aus[k]["emotions"]["happiness"] for k in ks],
                          [va[k]["emotion_scores"]["Happiness"] for k in ks],
                          alternative="greater")
    assert rho.pvalue < CORRELATION_ALPHA, f"rho={rho.statistic:+.3f} p={rho.pvalue:.3g}"


def test_independent_emotion_labels_agree_beyond_chance(faces, va):
    """Cohen's kappa, not raw agreement: most faces read Neutral under both
    models, so raw agreement is high by default and would pass even if the
    models agreed on nothing else.
    """
    ks = [k for k in va if k in _with_face(faces)]
    canon = lambda s: CANONICAL_LABEL.get(s.lower(), s.lower())  # noqa: E731
    a = [canon(faces[k]["faces"][0]["emotion"]["label"]) for k in ks]
    b = [canon(va[k]["label"]) for k in ks]
    observed = np.mean([x == y for x, y in zip(a, b)])
    expected = sum(a.count(c) * b.count(c) for c in set(a)) / len(ks) ** 2
    kappa = (observed - expected) / (1 - expected)
    assert kappa > 0, f"kappa={kappa:.3f}"
