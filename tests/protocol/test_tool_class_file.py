"""SPECAGENT_TOOL_CLASS_FILE: benchmark-supplied tool labels override the name heuristics."""

import json

import pytest

from specagent.protocol import side_effects
from specagent.protocol.side_effects import action_mutates_state, load_official_tool_classes


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("SPECAGENT_TOOL_CLASS_FILE", raising=False)
    side_effects._OFFICIAL_CACHE.clear()
    yield
    side_effects._OFFICIAL_CACHE.clear()


def _use(monkeypatch, tmp_path, table):
    f = tmp_path / "classes.json"
    f.write_text(json.dumps(table), encoding="utf-8")
    monkeypatch.setenv("SPECAGENT_TOOL_CLASS_FILE", str(f))
    return f


def test_file_labels_override_heuristics(monkeypatch, tmp_path):
    # create_form is a heuristic write, list_directory a heuristic read; the file inverts both.
    _use(monkeypatch, tmp_path, {"create_form": False, "list_directory": True})
    assert action_mutates_state({"tool": "create_form"}) is False
    assert action_mutates_state({"tool": "list_directory"}) is True


def test_file_label_beats_readonly_tools_and_mcp_prefix(monkeypatch, tmp_path):
    _use(monkeypatch, tmp_path, {"get_thing": True})
    assert action_mutates_state({"tool": "get_thing"}, readonly_tools=("get_thing",)) is True
    assert action_mutates_state({"tool": "srv__get_thing"}) is True


def test_names_absent_from_file_use_generic_logic(monkeypatch, tmp_path):
    _use(monkeypatch, tmp_path, {"only_this": False})
    assert action_mutates_state({"tool": "send_email"}) is True
    assert action_mutates_state({"tool": "list_directory"}) is False
    assert action_mutates_state({"tool": "claim_done"}) is True
    assert action_mutates_state({"tool": "never_seen_tool_xyz"}) is True


def test_unset_keeps_the_heuristics(monkeypatch):
    names = ("create_form", "list_directory", "send_email", "claim_done", "never_seen_tool_xyz")
    for name in names:
        action = {"tool": name}
        assert action_mutates_state(action) is action_mutates_state(action, readonly_tools=())
    assert side_effects.official_mutates_map() is None
    assert action_mutates_state({"tool": "create_form"}) is True
    assert action_mutates_state({"tool": "list_directory"}) is False


@pytest.mark.parametrize(
    "content",
    ["not json", "[1, 2]", '{"a": "yes"}', '{"a": 1}', '{"a": null}'],
)
def test_invalid_file_raises(monkeypatch, tmp_path, content):
    f = tmp_path / "bad.json"
    f.write_text(content, encoding="utf-8")
    monkeypatch.setenv("SPECAGENT_TOOL_CLASS_FILE", str(f))
    with pytest.raises(ValueError):
        action_mutates_state({"tool": "create_form"})
    with pytest.raises(ValueError):
        load_official_tool_classes(f)


def test_missing_file_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("SPECAGENT_TOOL_CLASS_FILE", str(tmp_path / "nope.json"))
    with pytest.raises(ValueError):
        action_mutates_state({"tool": "create_form"})
    with pytest.raises(ValueError):
        side_effects.official_mutates_map()
