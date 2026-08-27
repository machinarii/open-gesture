"""CLIP backend: image and text embeddings for the README §4 embedding space.

Embeds each image alongside its curated `intent` and `physical_description`, and
records the image-text cosine similarity. Low similarity flags an image whose
visual content disagrees with its label.

All vectors are L2-normalised on write, so cosine similarity is a dot product and
consumers need no renormalisation.

Sidecar note: the spec assigns this backend the binary sidecar `embeddings.npz`.
Making one backend write a binary format would force a special case through the
whole I/O layer (which only knows how to read/write/validate JSON sidecars), so
this backend writes `embeddings.json` like every other backend, and `export_npz`
derives `embeddings.npz` from it as a separate, explicit step (see the `og-annotate
export-npz` CLI subcommand).

Licence verification (Task 11, 2026-08-25): as with Tasks 7 and 8, the library's
licence does not straightforwardly propagate to the checkpoint, and the library's
own packaging metadata is internally inconsistent:
  - `open_clip_torch`'s repository `LICENSE` file (mlfoundations/open_clip) has
    read MIT, unchanged, since 2021. However, the PyPI *trove classifier* for the
    minimum version this project pins (`open_clip_torch>=2.24`, i.e. 2.24.0) is
    "License :: OSI Approved :: Apache Software License" -- a stale/incorrect
    classifier later corrected; the currently-installed 3.3.0 correctly classifies
    as MIT. Both facts are real and verifiable (PyPI JSON API, GitHub LICENSE at
    each tag), so the library entry below records both rather than picking one.
  - The `laion2b_s34b_b79k` checkpoint (HF repo
    laion/CLIP-ViT-B-32-laion2B-s34B-b79K, per open_clip's own pretrained.py
    hf_hub mapping) is independently tagged `license:mit` on its HuggingFace model
    card/API (https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K) -- MIT,
    not Apache-2.0. It is recorded as MIT below rather than assumed to inherit an
    Apache label from the library string.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from open_gesture_annotate.io import Gesture
from open_gesture_annotate.schema import ok

MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"


def _normalise(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec


class ClipEmbedBackend:
    name = "clip-embed"
    sidecar = "embeddings.json"

    def __init__(self) -> None:
        self._model = None
        self._preprocess = None
        self._tokenizer = None

    @property
    def version(self) -> str:
        try:
            from importlib.metadata import version

            return version("open_clip_torch")
        except Exception:
            return "unknown"

    def available(self) -> tuple[bool, str]:
        try:
            import open_clip  # noqa: F401
            import torch  # noqa: F401
        except ImportError as exc:
            return False, f"open_clip/torch not installed ({exc}); pip install -e '.[embed]'"
        return True, f"open_clip {self.version} ({MODEL_NAME}/{PRETRAINED})"

    def _load(self):
        if self._model is None:
            import open_clip
            import torch

            model, _, preprocess = open_clip.create_model_and_transforms(
                MODEL_NAME, pretrained=PRETRAINED
            )
            model.eval()
            self._model = model
            self._preprocess = preprocess
            self._tokenizer = open_clip.get_tokenizer(MODEL_NAME)
            self._torch = torch
        return self._model, self._preprocess, self._tokenizer

    def provenance(self) -> dict:
        return {
            "library": {"name": "open_clip_torch", "version": self.version,
                        "license": "Apache-2.0 / MIT"},
            "models": [{"name": MODEL_NAME, "pretrained": PRETRAINED, "license": "MIT"}],
        }

    def annotate(self, image: np.ndarray, gesture: Gesture) -> dict:
        from PIL import Image

        model, preprocess, tokenizer = self._load()
        torch = self._torch

        pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        intent = gesture.raw.get("intent", "") or ""
        description = gesture.raw.get("physical_description", "") or ""

        with torch.no_grad():
            img_vec = model.encode_image(preprocess(pil).unsqueeze(0))[0].cpu().numpy()
            txt = model.encode_text(tokenizer([intent, description])).cpu().numpy()

        img_vec = _normalise(img_vec.astype(np.float32))
        intent_vec = _normalise(txt[0].astype(np.float32))
        desc_vec = _normalise(txt[1].astype(np.float32))

        return ok(
            image=[round(float(v), 6) for v in img_vec],
            intent=[round(float(v), 6) for v in intent_vec],
            description=[round(float(v), 6) for v in desc_vec],
            similarity_intent=round(float(img_vec @ intent_vec), 6),
            similarity_description=round(float(img_vec @ desc_vec), 6),
            dim=int(img_vec.size),
        )


def export_npz(out_dir: Path) -> Path:
    """Derive annotations/embeddings.npz from embeddings.json for numeric consumers."""
    out_dir = Path(out_dir)
    data = json.loads((out_dir / "embeddings.json").read_text(encoding="utf-8"))
    records = {gid: r for gid, r in data["records"].items() if r.get("status") == "ok"}
    ids = sorted(records)
    path = out_dir / "embeddings.npz"
    np.savez_compressed(
        path,
        ids=np.array(ids, dtype=object).astype(str),
        image=np.array([records[i]["image"] for i in ids], dtype=np.float32),
        intent=np.array([records[i]["intent"] for i in ids], dtype=np.float32),
        description=np.array([records[i]["description"] for i in ids], dtype=np.float32),
    )
    return path
