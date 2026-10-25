"""Behavior and structural cost of bound AIMNet source identities."""
from pathlib import Path
from types import SimpleNamespace
import os

import pytest

from maple.function.calculator.aimnet import _aimnet2025_identity as identity
from test_aimnet2025_adapter import copied_source_package as source_package_fixture

copied_source_package = source_package_fixture


def forbidden(*args, **kwargs):
    pytest.fail('unchanged hot source guard repeated full inventory/resolution/hash')


def module_at(path):
    return SimpleNamespace(__file__=str(path), __spec__=SimpleNamespace(origin=str(path)))


def test_unchanged_source_checks_do_not_walk_resolve_or_hash(copied_source_package, monkeypatch):
    module = module_at(copied_source_package / 'ops.py')
    monkeypatch.setattr(identity, '_executing_aimnet_modules', lambda: (('aimnet.ops', module),))
    _, bound = identity.bind_external_runtime()
    identity.require_external_sources(bound)
    monkeypatch.setattr(Path, 'rglob', forbidden)
    monkeypatch.setattr(Path, 'resolve', forbidden)
    monkeypatch.setattr(identity.CheckpointFileIdentity, 'bind', forbidden)
    for _ in range(3):
        identity.require_external_sources(bound)


def test_ordinary_bytecode_churn_refreshes_only_verified_inventory(copied_source_package, monkeypatch):
    _, bound = identity.bind_external_runtime()
    cache = copied_source_package / '__pycache__'
    cache.mkdir()
    pyc = cache / 'ops.cpython-311.pyc'
    pyc.write_bytes(b'ordinary interpreter cache')
    identity.require_external_sources(bound)
    pyc.unlink()
    cache.rmdir()
    identity.require_external_sources(bound)
    monkeypatch.setattr(Path, 'rglob', forbidden)
    monkeypatch.setattr(Path, 'resolve', forbidden)
    identity.require_external_sources(bound)


@pytest.mark.parametrize('mutation', ['nested_python', 'cache_python', 'file_symlink', 'dir_symlink', 'cache_symlink', 'replacement'])
def test_live_inventory_change_never_refreshes_source_authority(copied_source_package, tmp_path, mutation):
    root = copied_source_package
    _, bound = identity.bind_external_runtime()
    if mutation in ('nested_python', 'cache_python'):
        folder = root / ('__pycache__' if mutation == 'cache_python' else 'lazy_dispatch')
        folder.mkdir()
        (folder / 'extra.py').write_text('pass\n')
    elif mutation == 'file_symlink':
        original = root / 'ops.py'
        moved = tmp_path / 'original_ops.py'
        original.rename(moved)
        original.symlink_to(moved)
    elif mutation == 'dir_symlink':
        (root / 'dispatch').symlink_to(tmp_path, target_is_directory=True)
    elif mutation == 'cache_symlink':
        (root / 'harmless.pyc').symlink_to(root / 'ops.py')
    else:
        original = root / 'ops.py'
        replacement = tmp_path / 'replacement.py'
        replacement.write_bytes(original.read_bytes())
        os.replace(replacement, original)
    with pytest.raises(ValueError, match='identity|inventory|changed'):
        identity.require_external_sources(bound)


def test_new_pinned_lazy_import_is_validated_once(copied_source_package, monkeypatch):
    _, bound = identity.bind_external_runtime()
    module = module_at(copied_source_package / 'ops.py')
    monkeypatch.setattr(identity, '_executing_aimnet_modules', lambda: (('aimnet.ops', module),))
    identity.require_external_sources(bound)
    monkeypatch.setattr(Path, 'resolve', forbidden)
    monkeypatch.setattr(Path, 'rglob', forbidden)
    identity.require_external_sources(bound)


def test_directory_entry_check_does_not_rely_on_timestamp_resolution(copied_source_package, monkeypatch):
    # Model a coarse filesystem clock without modifying any real source stat.
    monkeypatch.setattr(identity, '_directory_identity', lambda observed: ('same-clock-tick',))
    _, bound = identity.bind_external_runtime()
    cache = copied_source_package / '__pycache__'
    cache.mkdir()
    (cache / 'extra.py').write_text('pass\n')
    with pytest.raises(ValueError, match='inventory'):
        identity.require_external_sources(bound)


@pytest.mark.parametrize('mutation', ['object', 'removed', 'origin', 'search_path'])
def test_validated_module_identity_cannot_be_replaced(copied_source_package, monkeypatch, mutation):
    module = module_at(copied_source_package / 'ops.py')
    modules = [('aimnet.ops', module)]
    monkeypatch.setattr(identity, '_executing_aimnet_modules', lambda: tuple(modules))
    _, bound = identity.bind_external_runtime()
    if mutation == 'object':
        modules[0] = ('aimnet.ops', module_at(copied_source_package / 'ops.py'))
    elif mutation == 'removed':
        modules.clear()
    elif mutation == 'origin':
        module.__spec__.origin = str(copied_source_package / 'nbops.py')
    else:
        module.__path__ = [str(copied_source_package)]
    with pytest.raises(ValueError, match='executing.*identity'):
        identity.require_external_sources(bound)


def test_module_origin_alias_retarget_is_not_hidden_by_cache(copied_source_package, monkeypatch, tmp_path):
    root = copied_source_package
    alias = tmp_path / 'alias'
    alias.symlink_to(root, target_is_directory=True)
    module = module_at(alias / 'ops.py')
    monkeypatch.setattr(identity, '_executing_aimnet_modules', lambda: (('aimnet.ops', module),))
    _, bound = identity.bind_external_runtime()
    foreign = tmp_path / 'foreign'
    foreign.mkdir()
    # Same inode and bytes do not authorize an origin outside the bound tree.
    os.link(root / 'ops.py', foreign / 'ops.py')
    alias.unlink()
    alias.symlink_to(foreign, target_is_directory=True)
    with pytest.raises(ValueError, match='identity|origin|alias|changed'):
        identity.require_external_sources(bound)


def test_canonical_package_ancestor_cannot_turn_into_a_symlink(copied_source_package, monkeypatch, tmp_path):
    prefix = tmp_path / 'prefix'
    prefix.mkdir()
    root = prefix / 'aimnet'
    copied_source_package.rename(root)
    monkeypatch.setattr(identity.metadata, 'distribution', lambda name: SimpleNamespace(locate_file=lambda p: prefix / p))
    module = module_at(root / 'ops.py')
    monkeypatch.setattr(identity, '_executing_aimnet_modules', lambda: (('aimnet.ops', module),))
    _, bound = identity.bind_external_runtime()
    real = tmp_path / 'moved-prefix'
    prefix.rename(real)
    prefix.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match='identity|origin|alias|changed'):
        identity.require_external_sources(bound)


@pytest.mark.parametrize('kind', ['foreign_origin', 'unlisted_name'])
def test_new_lazy_import_cannot_gain_unbound_source_authority(copied_source_package, monkeypatch, tmp_path, kind):
    _, bound = identity.bind_external_runtime()
    module = module_at(copied_source_package / 'ops.py')
    name = 'aimnet.unlisted_math' if kind == 'unlisted_name' else 'aimnet.ops'
    if kind == 'foreign_origin':
        module.__file__ = module.__spec__.origin = str(tmp_path / 'foreign.py')
    monkeypatch.setattr(identity, '_executing_aimnet_modules', lambda: ((name, module),))
    with pytest.raises(ValueError, match='executing.*identity'):
        identity.require_external_sources(bound)


def test_disappearing_bytecode_during_full_rescan_is_not_a_source_change(copied_source_package, monkeypatch):
    _, bound = identity.bind_external_runtime()
    cache = copied_source_package / '__pycache__'
    cache.mkdir()
    bytecode = cache / 'ops.cpython-311.pyc'
    bytecode.write_bytes(b'ordinary bytecode')
    original = Path.lstat

    def lstat(path, *args, **kwargs):
        if path == bytecode and bytecode.exists():
            bytecode.unlink()
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'lstat', lstat)
    identity.require_external_sources(bound)
    identity.require_external_sources(bound)
