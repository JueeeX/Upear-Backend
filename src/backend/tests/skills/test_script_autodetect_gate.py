"""Script auto-whitelist gating — the execution grant is opt-in, not ambient.

Before this gate, any ``.py``/``.js``/``.sh`` file sitting next to SKILL.md
(or shipped as a DB extra_file) was silently whitelisted for sandbox
execution. That turned "a skill carries a file" into "a skill may execute a
file" with no human in the loop — exactly the property an evolution-materialised
(or marketplace-imported) skill must not have by default.
"""

import logging
from pathlib import Path

from core.agent_skills.backends import CompositeBackend, FilesystemBackend
from core.agent_skills.config import auto_detect_scripts_enabled
from core.agent_skills.loader import (
    MultiSourceSkillLoader,
    _auto_detect_scripts,
    _script_candidates,
)

_SKILL_MD = (
    "---\n"
    "name: gated-skill\n"
    "display_name: Gated Skill\n"
    "description: carries scripts\n"
    "---\n\n"
    "## Steps\n\n1. run the helper\n"
)


def _make_skill(
    tmp_path: Path,
    *,
    scripts=("helper.py",),
    scripts_json: str | None = None,
) -> Path:
    skill_dir = tmp_path / "gated-skill"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(_SKILL_MD, encoding="utf-8")
    for name in scripts:
        (skill_dir / name).write_text("print('x')\n", encoding="utf-8")
    if scripts_json is not None:
        (skill_dir / "_scripts.json").write_text(scripts_json, encoding="utf-8")
    return skill_dir


def _loader(tmp_path: Path) -> MultiSourceSkillLoader:
    backend = CompositeBackend(
        [FilesystemBackend(root_dir=tmp_path, source_name="test", priority=10)]
    )
    return MultiSourceSkillLoader(backend=backend)


def test_auto_detect_defaults_to_off(monkeypatch):
    monkeypatch.delenv("SUMC_SKILL_AUTO_DETECT_SCRIPTS", raising=False)
    assert auto_detect_scripts_enabled() is False
    for value in ("1", "true", "yes"):
        monkeypatch.setenv("SUMC_SKILL_AUTO_DETECT_SCRIPTS", value)
        assert auto_detect_scripts_enabled() is True
    monkeypatch.setenv("SUMC_SKILL_AUTO_DETECT_SCRIPTS", "0")
    assert auto_detect_scripts_enabled() is False


def test_script_files_get_no_whitelist_by_default(tmp_path, monkeypatch, caplog):
    monkeypatch.delenv("SUMC_SKILL_AUTO_DETECT_SCRIPTS", raising=False)
    _make_skill(tmp_path, scripts=("helper.py", "side_effect.sh"))
    with caplog.at_level(logging.WARNING, logger="core.agent_skills.loader"):
        spec = _loader(tmp_path).load_skill_full("gated-skill")

    assert spec is not None
    assert spec.executable_scripts == []
    warnings = [r for r in caplog.records if "NOT executable" in r.getMessage()]
    assert warnings, "unapproved scripts must be surfaced in the log"
    assert "helper.py" in warnings[0].getMessage()


def test_opt_in_flag_restores_auto_detection(tmp_path, monkeypatch):
    monkeypatch.setenv("SUMC_SKILL_AUTO_DETECT_SCRIPTS", "true")
    _make_skill(tmp_path)
    spec = _loader(tmp_path).load_skill_full("gated-skill")
    assert [s["name"] for s in spec.executable_scripts] == ["helper.py"]


def test_explicit_scripts_json_is_approved_without_the_flag(tmp_path, monkeypatch):
    """An explicit _scripts.json is the approval path — it needs no opt-in."""
    monkeypatch.delenv("SUMC_SKILL_AUTO_DETECT_SCRIPTS", raising=False)
    _make_skill(
        tmp_path,
        scripts=("helper.py", "unlisted.py"),
        scripts_json='[{"name": "helper.py", "language": "python", "timeout": 30}]',
    )
    spec = _loader(tmp_path).load_skill_full("gated-skill")
    assert [s["name"] for s in spec.executable_scripts] == ["helper.py"]


def _fake_skill_info(file_path: Path):
    from types import SimpleNamespace

    return SimpleNamespace(
        is_database=False,
        content=None,
        file_path=file_path,
    )


def test_candidates_cover_extra_files_and_directory(tmp_path):
    skill_dir = _make_skill(tmp_path, scripts=("a.py", "b.sh"))
    assert _script_candidates({}, _fake_skill_info(skill_dir / "SKILL.md")) == [
        "a.py",
        "b.sh",
    ]
    # DB skills: extra_files only — the directory must not be scanned.
    assert _script_candidates({"c.py": "x", "_meta.json": "{}"}, _fake_skill_info(
        tmp_path / "elsewhere" / "SKILL.md"
    )) == ["c.py"]
    # Non-script files never appear.
    empty_dir = tmp_path / "empty-skill"
    empty_dir.mkdir()
    (empty_dir / "SKILL.md").write_text(_SKILL_MD, encoding="utf-8")
    (empty_dir / "readme.md").write_text("docs", encoding="utf-8")
    assert _script_candidates({}, _fake_skill_info(empty_dir / "SKILL.md")) == []


def test_auto_detect_entries_carry_language_and_timeout(tmp_path):
    skill_dir = _make_skill(tmp_path, scripts=("run.py",))
    entries = _auto_detect_scripts({}, _fake_skill_info(skill_dir / "SKILL.md"))
    assert entries == [
        {"name": "run.py", "language": "python", "timeout": 60}
    ]
