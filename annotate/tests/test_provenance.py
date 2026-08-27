import json

from open_gesture_annotate import registry
from open_gesture_annotate.provenance import (
    collect,
    is_permissive,
    licence_warnings,
    write_meta,
)
from tests.conftest import FakeBackend


def test_recognises_permissive_licences():
    assert is_permissive("MIT")
    assert is_permissive("Apache-2.0")
    assert is_permissive("BSD-3-Clause")


def test_rejects_non_permissive_licences():
    assert not is_permissive("CC-BY-NC-4.0")
    assert not is_permissive("AGPL-3.0")
    assert not is_permissive("check upstream weight licence")


def test_collect_records_each_backend(tmp_path):
    registry.register("prov-fake", FakeBackend)
    meta = collect(["prov-fake"])
    assert meta["backends"]["prov-fake"]["available"] is True
    assert meta["backends"]["prov-fake"]["provenance"]["models"][0]["license"] == "MIT"


def test_collect_records_unavailable_backends_without_raising():
    registry.register("prov-down", lambda: FakeBackend(available=False, reason="nope"))
    meta = collect(["prov-down"])
    assert meta["backends"]["prov-down"]["available"] is False
    assert meta["backends"]["prov-down"]["reason"] == "nope"


def test_no_warnings_for_an_all_permissive_run():
    registry.register("prov-fake", FakeBackend)
    assert licence_warnings(collect(["prov-fake"])) == []


def test_warns_about_a_non_permissive_weight():
    class Restricted(FakeBackend):
        name = "restricted"

        def provenance(self):
            return {"models": [{"name": "w", "license": "CC-BY-NC-4.0"}]}

    registry.register("prov-nc", Restricted)
    warnings = licence_warnings(collect(["prov-nc"]))
    assert len(warnings) == 1
    assert "CC-BY-NC-4.0" in warnings[0]


def test_write_meta_produces_readable_json(tmp_path):
    registry.register("prov-fake", FakeBackend)
    path = write_meta(tmp_path, ["prov-fake"])
    meta = json.loads(path.read_text(encoding="utf-8"))
    assert path.name == "_meta.json"
    assert "generated_at" in meta
    assert "prov-fake" in meta["backends"]
