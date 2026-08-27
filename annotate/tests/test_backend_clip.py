import numpy as np
import pytest

pytest.importorskip("open_clip")

from open_gesture_annotate.backends.embed_clip import ClipEmbedBackend, export_npz
from open_gesture_annotate.io import load_image, load_manifest, repo_root
from open_gesture_annotate.schema import new_sidecar, ok, validate_record
from open_gesture_annotate.io import write_sidecar


def test_backend_identity():
    b = ClipEmbedBackend()
    assert b.name == "clip-embed"
    assert b.sidecar == "embeddings.json"


def test_provenance_names_the_library_licence_as_apache():
    """This asserts the *library* (open_clip_torch) licence field, not the
    ViT-B-32/laion2b_s34b_b79k checkpoint's licence -- that weight is recorded
    as MIT (see the module docstring and `test_provenance_names_the_checkpoint_as_mit`).
    """
    assert "Apache" in ClipEmbedBackend().provenance()["library"]["license"]


def test_provenance_names_the_checkpoint_as_mit():
    assert ClipEmbedBackend().provenance()["models"][0]["license"] == "MIT"


def test_export_npz_stacks_records_into_arrays(tmp_path):
    b = ClipEmbedBackend()
    data = new_sidecar(b)
    for gid in ("a", "b"):
        data["records"][gid] = ok(image=[1.0, 0.0], intent=[0.0, 1.0],
                                  description=[1.0, 0.0],
                                  similarity_intent=0.0,
                                  similarity_description=1.0, dim=2)
    write_sidecar(tmp_path / "embeddings.json", data)

    path = export_npz(tmp_path)
    loaded = np.load(path, allow_pickle=False)
    assert list(loaded["ids"]) == ["a", "b"]
    assert loaded["image"].shape == (2, 2)
    assert loaded["intent"].shape == (2, 2)


def test_export_npz_skips_error_records(tmp_path):
    from open_gesture_annotate.schema import error

    b = ClipEmbedBackend()
    data = new_sidecar(b)
    data["records"]["a"] = ok(image=[1.0, 0.0], intent=[0.0, 1.0], description=[1.0, 0.0],
                              similarity_intent=0.0, similarity_description=1.0, dim=2)
    data["records"]["b"] = error("boom")
    write_sidecar(tmp_path / "embeddings.json", data)

    loaded = np.load(export_npz(tmp_path), allow_pickle=False)
    assert list(loaded["ids"]) == ["a"]


@pytest.mark.slow
def test_embeds_a_real_image():
    b = ClipEmbedBackend()
    gestures = {g.id: g for g in load_manifest(repo_root())}
    g = gestures["affirm-01"]
    rec = b.annotate(load_image(repo_root(), g), g)
    validate_record(rec)
    assert rec["dim"] == 512
    assert len(rec["image"]) == 512
    assert abs(float(np.linalg.norm(rec["image"])) - 1.0) < 1e-3
    assert -1.0 <= rec["similarity_intent"] <= 1.0
