"""Annotation record shape and validation."""

from __future__ import annotations

import json
from datetime import datetime, timezone

VALID_STATUSES = ("ok", "error")


class SchemaError(ValueError):
    """A record or sidecar does not match the annotation schema."""


def ok(**fields) -> dict:
    """Build a successful record. Payload keys are backend-specific."""
    return {"status": "ok", **fields}


def error(message: str) -> dict:
    """Build a failed record. Never raises; a failure is a recorded outcome."""
    return {"status": "error", "error": str(message)}


def validate_record(rec: dict) -> None:
    if not isinstance(rec, dict):
        raise SchemaError(f"record must be a dict, got {type(rec).__name__}")
    status = rec.get("status")
    if status not in VALID_STATUSES:
        raise SchemaError(f"record 'status' must be one of {VALID_STATUSES}, got {status!r}")
    if status == "error" and not rec.get("error"):
        raise SchemaError("record with status 'error' must carry a non-empty 'error' message")
    try:
        json.dumps(rec)
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"record is not JSON-serialisable: {exc}") from exc


def new_sidecar(backend) -> dict:
    return {
        "_backend": {
            "name": backend.name,
            "version": backend.version,
            "sidecar": backend.sidecar,
            "run_at": datetime.now(timezone.utc).isoformat(),
        },
        "records": {},
    }


def validate_sidecar(data: dict) -> None:
    meta = data.get("_backend")
    if not isinstance(meta, dict):
        raise SchemaError("sidecar must carry a '_backend' object")
    for key in ("name", "version", "run_at"):
        if not meta.get(key):
            raise SchemaError(f"sidecar '_backend' is missing '{key}'")
    records = data.get("records")
    if not isinstance(records, dict):
        raise SchemaError("sidecar must carry a 'records' object")
    for gesture_id, rec in records.items():
        try:
            validate_record(rec)
        except SchemaError as exc:
            raise SchemaError(f"record {gesture_id!r}: {exc}") from exc
