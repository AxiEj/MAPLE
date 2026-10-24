"""Bounded AmberTools selection and isolated child environments."""

import os

import pytest

from maple.function.calculator.extra_correction.implicit import ambertools


def install(prefix):
    for name in ("antechamber", "atomtype", "bondtype", "am1bcc", "sqm"):
        path = prefix / "bin" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/bin/sh\nexit 0\n")
        path.chmod(0o755)
    for name in (
        "CONNECT.TPL",
        "APS.DAT",
        "ATOMTYPE_GFF2.DEF",
        "ATOMTYPE_BCC.DEF",
        "BCCPARM.DAT",
        "ATOMTYPE_ABCG2.DEF",
        "BCCPARM_ABCG2.DAT",
    ):
        path = prefix / "dat" / "antechamber" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("data")
    return prefix / "bin" / "antechamber"


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.delenv("AMBERHOME", raising=False)
    monkeypatch.setattr(ambertools.sys, "prefix", str(tmp_path / "active"))
    return home


def registry(home, *prefixes):
    path = home / ".conda" / "environments.txt"
    path.parent.mkdir()
    path.write_text("\n".join(map(str, prefixes)) + "\n")


def test_explicit_wins_and_env_is_private(tmp_path, monkeypatch, isolated):
    executable = install(tmp_path / "explicit")
    other = install(tmp_path / "other")
    monkeypatch.setenv("AMBERHOME", str(other.parent.parent))
    before = dict(os.environ)
    resolved = ambertools.resolve_antechamber(str(executable))
    assert resolved.executable == str(executable)
    assert resolved.environment["AMBERHOME"] == str(executable.parent.parent)
    assert resolved.environment["PATH"].split(os.pathsep)[0] == str(executable.parent)
    assert resolved.discovery["selection"] == "explicit"
    assert resolved.discovery["helpers"]["sqm"]["sha256"]
    assert dict(os.environ) == before


def test_unusable_explicit_never_falls_back(tmp_path, isolated):
    install(tmp_path / "registered")
    registry(isolated, tmp_path / "registered")
    with pytest.raises(ImportError, match="explicit"):
        ambertools.resolve_antechamber(str(tmp_path / "missing"))


def test_amberhome_precedes_path(tmp_path, monkeypatch, isolated):
    expected = install(tmp_path / "amberhome")
    on_path = install(tmp_path / "on-path")
    monkeypatch.setenv("AMBERHOME", str(expected.parent.parent))
    monkeypatch.setenv("PATH", str(on_path.parent) + os.pathsep + os.environ["PATH"])
    assert ambertools.resolve_antechamber().executable == str(expected)


def test_active_prefix_and_registered_environment_discovery(
    tmp_path, monkeypatch, isolated
):
    active = install(tmp_path / "active")
    assert ambertools.resolve_antechamber().executable == str(active)
    active.unlink()
    registered = install(tmp_path / "registered")
    registry(
        isolated,
        tmp_path / "missing",
        registered.parent.parent,
        registered.parent.parent,
    )
    resolved = ambertools.resolve_antechamber()
    assert resolved.executable == str(registered)
    assert resolved.discovery["selection"] == "conda-registry"


def test_multiple_viable_registered_installs_require_explicit(tmp_path, isolated):
    a = install(tmp_path / "a")
    b = install(tmp_path / "b")
    registry(isolated, a.parent.parent, b.parent.parent)
    with pytest.raises(ImportError, match="[Aa]mbiguous.*executable") as exc:
        ambertools.resolve_antechamber()
    assert str(a) in str(exc.value) and str(b) in str(exc.value)


def test_incomplete_registered_install_is_not_ready(tmp_path, isolated):
    executable = install(tmp_path / "partial")
    (executable.parent / "sqm").unlink()
    registry(isolated, executable.parent.parent)
    with pytest.raises(ImportError, match="AmberTools24"):
        ambertools.resolve_antechamber()


def test_incomplete_explicit_install_fails_instead_of_falling_back(tmp_path, isolated):
    executable = install(tmp_path / "partial")
    (executable.parent / "sqm").unlink()
    ready = install(tmp_path / "ready")
    registry(isolated, ready.parent.parent)
    with pytest.raises(ImportError, match="sqm"):
        ambertools.resolve_antechamber(str(executable))


def test_explicit_symlink_resolves_own_installation_resources(tmp_path, isolated):
    executable = install(tmp_path / "real")
    link = tmp_path / "link-antechamber"
    link.symlink_to(executable)
    resolved = ambertools.resolve_antechamber(str(link))
    assert resolved.executable == str(executable)
    assert resolved.environment["AMBERHOME"] == str(executable.parent.parent)
    assert resolved.discovery["helpers"]["sqm"]["path"] == str(
        executable.parent / "sqm"
    )


def test_missing_method_data_is_not_an_installation_readiness_claim(tmp_path, isolated):
    executable = install(tmp_path / "partial")
    (executable.parent.parent / "dat" / "antechamber" / "ATOMTYPE_ABCG2.DEF").unlink()
    assert ambertools.resolve_antechamber(str(executable), method="am1bcc")
    with pytest.raises(ImportError, match="ATOMTYPE_ABCG2.DEF"):
        ambertools.resolve_antechamber(str(executable), method="abcg2")
