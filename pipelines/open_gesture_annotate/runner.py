"""Run one backend across the gesture set, isolating every failure."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from open_gesture_annotate.io import (
    Gesture,
    completed_ids,
    load_image,
    read_sidecar,
    sidecar_path,
    write_sidecar,
)
from open_gesture_annotate.schema import error, validate_record


@dataclass
class RunSummary:
    backend: str
    ok: int = 0
    errors: int = 0
    skipped: int = 0
    unavailable: str | None = None

    @property
    def attempted(self) -> int:
        return self.ok + self.errors


def run_backend(
    backend,
    gestures: list[Gesture],
    root: Path,
    out_dir: Path,
    force: bool = False,
    limit: int | None = None,
) -> RunSummary:
    """Annotate `gestures` with `backend`, writing after every image.

    Writing after each image is deliberate: a crash mid-run loses at most one
    result, and the next invocation resumes from the sidecar.
    """
    summary = RunSummary(backend=backend.name)

    # Tell the backend which output directory this run is actually using
    # BEFORE asking whether it can run at all. A backend's `available()` may
    # need to inspect that directory (e.g. `va` checking for `faces.json`
    # beneath it) -- calling the hook after available() would make that check
    # consult the wrong (default) location whenever `--out` overrides it,
    # silently defeating the very unavailability check it exists to make
    # possible. The hook has no side effects beyond storing a path, so calling
    # it unconditionally, before availability is known, is safe.
    set_output_dir = getattr(backend, "set_output_dir", None)
    if set_output_dir is not None:
        set_output_dir(out_dir)

    is_available, reason = backend.available()
    if not is_available:
        summary.unavailable = reason
        return summary

    path = sidecar_path(out_dir, backend)
    data = read_sidecar(path, backend)
    done = set() if force else completed_ids(data)

    attempted = 0
    for gesture in gestures:
        if gesture.id in done:
            summary.skipped += 1
            continue
        if limit is not None and attempted >= limit:
            break
        attempted += 1

        try:
            image = load_image(root, gesture)
            record = backend.annotate(image, gesture)
            validate_record(record)
        except Exception as exc:
            record = error(f"{type(exc).__name__}: {exc}")
            summary.errors += 1
        else:
            summary.ok += 1

        data["records"][gesture.id] = record
        write_sidecar(path, data)

    return summary
