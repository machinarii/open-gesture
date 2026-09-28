"""Speech backend: the voice channel, as a TTS -> noise -> Whisper round-trip.

Why this exists. The README's premise is that machines sharing our spaces need
more than one channel, and that voice is exactly the one that fails -- in
noise, at distance, through glass. The pipeline so far is image-only. This
backend adds speech so `og-annotate fusion` can measure whether image + speech
beats either alone, and how that holds up as the voice channel degrades.

There is no audio in this dataset, so the speech is synthesised. Each gesture
has one utterance in utterances.json -- what a person might say while making
it, written without reading the curated `intent` it is scored against. Per
gesture the backend:

  1. synthesises the utterance once (SpeechT5, 16 kHz, one fixed speaker),
  2. mixes in white noise at each SNR in CONDITIONS,
  3. transcribes each mix with Whisper -- real STT on real audio, not a text
     shortcut, so recognition errors propagate the way they would live,
  4. embeds each transcript with the SAME CLIP text tower `embed` uses for the
     intents, so transcript and intent vectors share a space, and
  5. scores each transcript's sentiment as a signed valence on HSEmotion's
     [-1, 1] scale, so text and face valence can be fused.

The spike that motivated CONDITIONS: at 0 dB "Stop! Don't come any closer!"
came back as "Stop, still coming closer please" -- noise did not merely drop
the voice channel, it inverted it. Fusion has to survive that, so the sweep goes
below 0 dB rather than stopping where recognition is still comfortable.

LICENCES -- every component was checked, because the obvious choice was not
permissive. Piper (piper-tts 1.8) is GPL-3.0-or-later and phonemises with
espeak-ng (also GPL), so it is excluded, as the spec excluded AGPL Ultralytics.
SpeechT5 is character-based, needs no phonemiser, and is MIT; see provenance().
"""

from __future__ import annotations

import io
import json
import re
import zlib
from pathlib import Path

import numpy as np

from open_gesture_annotate.io import Gesture
from open_gesture_annotate.schema import ok

SAMPLE_RATE = 16_000  # SpeechT5 emits 16 kHz, which is also what Whisper expects

TTS_MODEL = "microsoft/speecht5_tts"
VOCODER = "microsoft/speecht5_hifigan"
SPEAKER_REPO = "Matthijs/cmu-arctic-xvectors"
SPEAKER_ARCHIVE = "spkrec-xvect.zip"
SPEAKER_PREFIX = "spkrec-xvect/cmu_us_slt_arctic-"  # US female; first file used
STT_MODEL = "base.en"
SENTIMENT_MODEL = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

# None = clean. Chosen to span "fine" to "actively misleading" (see docstring).
# ponytail: white noise only. Road and crowd noise is lower-frequency and
# non-stationary; mix real recordings here if the white-noise curve looks
# optimistic.
CONDITIONS = {"clean": None, "snr10": 10.0, "snr0": 0.0, "snr-5": -5.0}

UTTERANCES_PATH = Path(__file__).resolve().parents[1] / "utterances.json"


def load_utterances(path: Path = UTTERANCES_PATH) -> dict[str, str]:
    return json.loads(Path(path).read_text(encoding="utf-8"))["utterances"]


def add_noise(wav: np.ndarray, snr_db: float | None, rng: np.random.Generator) -> np.ndarray:
    """Mix white Gaussian noise into `wav` at `snr_db`; None returns it unchanged."""
    if snr_db is None:
        return wav
    signal_power = float(np.mean(wav.astype(np.float64) ** 2))
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=wav.shape)
    return (wav + noise).astype(np.float32)


# Whisper writes numbers as digits ("Ten." -> "10.", "one, two, three" ->
# "1, 2, 3"). That is correct transcription, so scoring must not count it as an
# error: before this map, 11 of the 20 clean-speech "errors" were formatting.
_NUMBER_WORDS = {
    str(n): w for n, w in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen "
        "fourteen fifteen sixteen seventeen eighteen nineteen twenty".split())
} | {"1st": "first", "2nd": "second", "3rd": "third"}


def _words(text: str) -> list[str]:
    """Lower-cased word tokens, digits spelled out. Hyphens split: "1-2-3" -> three words."""
    return [_NUMBER_WORDS.get(t, t) for t in re.findall(r"[a-z0-9']+", text.lower())]


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Word-level Levenshtein distance divided by reference length.

    Case and punctuation are ignored, so "Stop, don't" matches "stop don't" --
    Whisper's punctuation choices are not recognition errors.
    """
    ref, hyp = _words(reference), _words(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        cur = [i]
        for j, h in enumerate(hyp, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1] / len(ref)


def stt_confidence(segments: list[tuple[float, int]]) -> float | None:
    """Whisper's own confidence in a transcript: exp(token-weighted mean log-prob).

    Each (avg_logprob, n_tokens) pair is one segment. The result is the
    geometric-mean probability Whisper assigned to each token it emitted -- a
    number in (0, 1] that comes from the STT model itself, so using it to weight
    the speech channel fits nothing to this dataset's labels. None when Whisper
    emitted no tokens.
    """
    total = sum(n for _, n in segments)
    if not total:
        return None
    return float(np.exp(sum(lp * n for lp, n in segments) / total))


def signed_valence(scores: list[dict]) -> float:
    """SST-2's {POSITIVE, NEGATIVE} probabilities -> p(pos) - p(neg) in [-1, 1].

    ponytail: SST-2 is binary, with no neutral class, so a neutral sentence is
    forced towards one pole. A 3-class model would fit the curated
    positive/neutral/negative labels better; the permissive one available
    (cardiffnlp, CC-BY-4.0) is trained on tweets and was passed over.
    """
    by_label = {s["label"].upper(): float(s["score"]) for s in scores}
    return by_label.get("POSITIVE", 0.0) - by_label.get("NEGATIVE", 0.0)


class SpeechBackend:
    name = "speech-roundtrip"
    sidecar = "speech.json"

    def __init__(self) -> None:
        self._models = None
        self._utterances: dict[str, str] | None = None

    @property
    def version(self) -> str:
        try:
            from importlib.metadata import version

            return f"faster-whisper {version('faster-whisper')}, transformers {version('transformers')}"
        except Exception:
            return "unknown"

    def available(self) -> tuple[bool, str]:
        try:
            import faster_whisper  # noqa: F401
            import open_clip  # noqa: F401
            import sentencepiece  # noqa: F401
            from transformers import SpeechT5ForTextToSpeech  # noqa: F401
        except ImportError as exc:
            return False, f"speech stack not installed ({exc}); pip install -e '.[embed,speech]'"
        return True, f"SpeechT5 -> Whisper {STT_MODEL}, {len(CONDITIONS)} noise conditions"

    def _load(self):
        if self._models is None:
            import zipfile

            import torch
            from faster_whisper import WhisperModel
            from huggingface_hub import hf_hub_download
            from transformers import (
                SpeechT5ForTextToSpeech,
                SpeechT5HifiGan,
                SpeechT5Processor,
                pipeline,
            )
            from transformers.utils import logging as hf_logging

            from open_gesture_annotate.backends.embed_clip import ClipEmbedBackend

            hf_logging.set_verbosity_error()
            archive = zipfile.ZipFile(
                hf_hub_download(SPEAKER_REPO, SPEAKER_ARCHIVE, repo_type="dataset")
            )
            speaker_file = sorted(n for n in archive.namelist()
                                  if n.startswith(SPEAKER_PREFIX) and n.endswith(".npy"))[0]
            speaker = torch.tensor(np.load(io.BytesIO(archive.read(speaker_file)))).unsqueeze(0)

            self._models = {
                "processor": SpeechT5Processor.from_pretrained(TTS_MODEL),
                "tts": SpeechT5ForTextToSpeech.from_pretrained(TTS_MODEL).eval(),
                "vocoder": SpeechT5HifiGan.from_pretrained(VOCODER).eval(),
                "speaker": speaker,
                "stt": WhisperModel(STT_MODEL, device="cpu", compute_type="int8"),
                "sentiment": pipeline("sentiment-analysis", model=SENTIMENT_MODEL, top_k=None),
                # The embed backend's own loader: guarantees transcripts land in
                # the same CLIP text space as the intent embeddings they are
                # compared with, rather than a near-copy of its constants.
                "clip": ClipEmbedBackend()._load(),
            }
        return self._models

    def _utterance(self, gesture_id: str) -> str:
        if self._utterances is None:
            self._utterances = load_utterances()
        return self._utterances.get(gesture_id, "")

    def provenance(self) -> dict:
        return {
            "library": {"name": "faster-whisper + transformers", "version": self.version,
                        "license": "MIT / Apache-2.0"},
            "models": [
                {"name": TTS_MODEL, "role": "text-to-speech", "license": "MIT"},
                {"name": VOCODER, "role": "vocoder", "license": "MIT"},
                {"name": f"{SPEAKER_REPO} ({SPEAKER_PREFIX}*)", "role": "TTS speaker x-vector",
                 "license": "MIT"},
                {"name": f"Systran/faster-whisper-{STT_MODEL}", "role": "speech-to-text",
                 "license": "MIT"},
                {"name": SENTIMENT_MODEL, "role": "transcript valence", "license": "Apache-2.0"},
            ],
            "note": (
                "Speech is synthesised, not recorded: utterances.json -> SpeechT5 -> white "
                f"noise at {list(CONDITIONS)} -> Whisper. The x-vectors are MIT on the Hub; "
                "they derive from CMU ARCTIC, whose licence permits commercial use provided "
                "its copyright notice is retained. Transcripts are embedded with the embed "
                "backend's CLIP model, recorded under that backend."
            ),
            "excluded": (
                "piper-tts (GPL-3.0-or-later) and its espeak-ng phonemiser (GPL-3.0); "
                "cardiffnlp/twitter-roberta-base-sentiment-latest (CC-BY-4.0, tweet-trained); "
                "j-hartmann/emotion-english-distilroberta-base (no licence stated)."
            ),
        }

    def _synthesise(self, text: str) -> np.ndarray:
        import torch

        m = self._load()
        inputs = m["processor"](text=text, return_tensors="pt")
        with torch.no_grad():
            wav = m["tts"].generate_speech(inputs["input_ids"], m["speaker"], vocoder=m["vocoder"])
        return wav.numpy().astype(np.float32)

    def _transcribe(self, wav: np.ndarray) -> tuple[str, float | None, float | None]:
        """(transcript, stt_confidence, no_speech_prob) for one clip."""
        segments = list(self._load()["stt"].transcribe(wav, language="en")[0])
        text = " ".join(s.text.strip() for s in segments).strip()
        return (text,
                stt_confidence([(s.avg_logprob, len(s.tokens)) for s in segments]),
                max((float(s.no_speech_prob) for s in segments), default=None))

    def _embed_text(self, text: str) -> list[float]:
        import torch

        model, _, tokenizer = self._load()["clip"]
        with torch.no_grad():
            vec = model.encode_text(tokenizer([text])).cpu().numpy()[0].astype(np.float64)
        return (vec / np.linalg.norm(vec)).tolist()

    def _valence(self, text: str) -> float:
        out = self._load()["sentiment"](text)
        return signed_valence(out[0] if out and isinstance(out[0], list) else out)

    def annotate(self, image: np.ndarray, gesture: Gesture) -> dict:
        utterance = self._utterance(gesture.id)
        if not utterance:
            # Performed silently: nothing to say, so nothing to transcribe. Fusion
            # falls back to the image for these, which is the behaviour under test.
            silent = {"transcript": "", "wer": None, "stt_confidence": None, "no_speech_prob": None,
                      "text_valence": None, "text_embedding": None}
            return ok(utterance="", speech_present=False,
                      conditions={name: {"snr_db": snr, **silent} for name, snr in CONDITIONS.items()})

        wav = self._synthesise(utterance)
        # zlib.crc32, not hash(): Python salts str hashes per process, which would
        # give every run different noise and make --force runs incomparable.
        rng = np.random.default_rng(zlib.crc32(gesture.id.encode("utf-8")))
        conditions = {}
        for name, snr in CONDITIONS.items():
            transcript, confidence, no_speech = self._transcribe(add_noise(wav, snr, rng))
            conditions[name] = {
                "snr_db": snr,
                "transcript": transcript,
                "wer": word_error_rate(utterance, transcript),
                "stt_confidence": confidence,
                "no_speech_prob": no_speech,
                "text_valence": self._valence(transcript) if transcript else None,
                "text_embedding": self._embed_text(transcript) if transcript else None,
            }
        return ok(utterance=utterance, speech_present=True,
                  audio_seconds=round(len(wav) / SAMPLE_RATE, 3), conditions=conditions)
