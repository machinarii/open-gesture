import numpy as np
import pytest

from open_gesture_annotate.io import Gesture
from open_gesture_annotate.schema import ok


class FakeBackend:
    """A backend that succeeds, with no model dependency."""

    name = "fake"
    version = "0.0.1"
    sidecar = "fake.json"

    def __init__(self, available=True, reason="ready", fail_on=()):
        self._available = available
        self._reason = reason
        self._fail_on = set(fail_on)
        self.calls = []

    def available(self):
        return self._available, self._reason

    def provenance(self):
        return {"models": [{"name": "fake-model", "license": "MIT"}]}

    def annotate(self, image, gesture):
        self.calls.append(gesture.id)
        if gesture.id in self._fail_on:
            raise RuntimeError(f"synthetic failure for {gesture.id}")
        return ok(width=int(image.shape[1]), height=int(image.shape[0]))


class ExplodingBackend(FakeBackend):
    """A backend whose annotate() returns something unserialisable."""

    name = "exploding"
    sidecar = "exploding.json"

    def annotate(self, image, gesture):
        return {"status": "ok", "blob": object()}


@pytest.fixture
def gestures():
    return [
        Gesture(id=f"g-{i:02d}", index=i, name=f"G{i}", category="C",
                file=f"images/g-{i:02d}.png", raw={})
        for i in range(1, 6)
    ]


@pytest.fixture
def image_root(tmp_path, gestures):
    """A throwaway tree of tiny real PNGs matching `gestures`."""
    import cv2

    (tmp_path / "images").mkdir()
    for i, g in enumerate(gestures, start=1):
        cv2.imwrite(str(tmp_path / g.file), np.full((8, 8 + i, 3), 128, np.uint8))
    return tmp_path
