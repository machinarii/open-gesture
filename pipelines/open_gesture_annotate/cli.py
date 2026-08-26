"""og-annotate command-line interface."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from open_gesture_annotate import registry
from open_gesture_annotate.io import load_manifest, repo_root
from open_gesture_annotate.runner import run_backend


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="og-annotate", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="show backends and their availability")

    run = sub.add_parser("run", help="annotate the gesture set")
    run.add_argument("--backends", default="all",
                     help="comma-separated backend keys, or 'all' (default)")
    run.add_argument("--out", default=None, help="output directory (default: <repo>/annotations)")
    run.add_argument("--force", action="store_true", help="re-annotate already-completed records")
    run.add_argument("--limit", type=int, default=None, help="annotate at most N images per backend")

    report = sub.add_parser("report", help="regenerate annotations/quality_report.md")
    report.add_argument("--out", default=None, help="annotations directory")

    export = sub.add_parser("export-npz", help="derive annotations/embeddings.npz")
    export.add_argument("--out", default=None, help="annotations directory")

    return parser


def _resolve_out(out: str | None) -> Path:
    return Path(out) if out else repo_root() / "annotations"


def _cmd_list() -> int:
    for key, available, reason in registry.availability():
        mark = "available" if available else "UNAVAILABLE"
        print(f"  {key:<12} {mark:<12} {reason}")
    return 0


def _cmd_run(args) -> int:
    keys = registry.all_keys() if args.backends == "all" else \
        [k.strip() for k in args.backends.split(",") if k.strip()]

    unknown = [k for k in keys if k not in registry.all_keys()]
    if unknown:
        print(f"unknown backend(s): {', '.join(unknown)}", file=sys.stderr)
        print(f"known: {', '.join(registry.all_keys())}", file=sys.stderr)
        return 2

    out_dir = _resolve_out(args.out)
    gestures = load_manifest(repo_root())

    for key in keys:
        summary = run_backend(registry.get(key), gestures, repo_root(), out_dir,
                              force=args.force, limit=args.limit)
        if summary.unavailable:
            print(f"  {key:<12} unavailable: {summary.unavailable}")
        else:
            print(f"  {key:<12} {summary.ok} ok, {summary.errors} errors, "
                  f"{summary.skipped} skipped")
    return 0


def _cmd_report(args) -> int:
    from open_gesture_annotate.report import write_report

    path = write_report(repo_root(), _resolve_out(args.out))
    print(f"wrote {path}")
    return 0


def _cmd_export_npz(args) -> int:
    from open_gesture_annotate.backends.embed_clip import export_npz

    print(f"wrote {export_npz(_resolve_out(args.out))}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "list":
        return _cmd_list()
    if args.command == "run":
        return _cmd_run(args)
    if args.command == "report":
        return _cmd_report(args)
    if args.command == "export-npz":
        return _cmd_export_npz(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
