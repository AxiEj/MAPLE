"""Portable pins for the one exact official 2025 potential.

These pins describe capability, not a scientific qualification receipt. Paths
are relative to the installed aimnet package, never a developer's environment.
"""
from __future__ import annotations

from importlib import metadata
from collections.abc import Mapping
from pathlib import Path
import os
import stat
import sys

import numpy as np

from ..analytic_protocol import (
    AnalyticIdentityError, CheckpointFileIdentity, file_stat_identity, named_array_sha256,
)

CHECKPOINT_FILENAME = "aimnet2_2025_b973c_d3_0.pt"
CHECKPOINT_SHA256 = "043ed5418a104e31f79462f8e5ebeca64a2d24422174f5d29f894d32271981b5"
REGISTRY_KEY = "aimnet2-b973c-2025-d3_0"
PROFILE_ID = "aimnet2025-native-f32-analytic-d3-v1"
NEIGHBOR_GRAPH_SCHEME = "row-multiset-with-padding-v1"
PRECISION_EXECUTION_POLICY = "model-scoped-ieee-no-tf32-v1"
PACKAGES = {"aimnet": "0.2.0", "torch": "2.12.0+cu130",
            "nvalchemi-toolkit-ops": "0.3.1", "warp-lang": "1.13.0"}
INSTALL_HINT = (
    "AIMNet2025 requires Python >=3.11 and the pinned optional native runtime: "
    "aimnet==0.2.0, nvalchemi-toolkit-ops==0.3.1, warp-lang==1.13.0, "
    "and torch==2.12.0+cu130 (select the matching official Torch wheel). "
    "The base MAPLE Python 3.10 installation does not require these packages."
)
# Complete package inventory verified against the official commit archive AND
# installed distribution RECORD; no hash is derived by trusting mutable live bytes.
# All 34 Python sources + 5 YAML configurations + the official D3 table.
OFFICIAL_SOURCE_COMMIT = "b1b0fd8c8b8d8acb8811eb646f9ba7b15d72da55"
SOURCE_SUFFIXES = frozenset({".py", ".yaml", ".pt"})
SOURCE_SHA256 = {
    "__init__.py": "bb18b97049cb5f783f0bb9d49f11b050b845fb0c8d214c1c6142260d8867a0a0",
    "calculators/__init__.py": "19ec2247ae975e571c87eaf0a830b63701c9724d5cb2be1897eeb903efe17cb0",
    "calculators/aimnet2ase.py": "8a5731fe9c2e803c3a99c48eb96f2f66c71e8680cad6dff4977a7316d5fbc9cf",
    "calculators/aimnet2pysis.py": "3c2abcf101efe07d10bf7227aa5af3cea373fe0595c7c2d214be292916633cc0",
    "calculators/aimnet2torchsim.py": "76fdbe3e74f4a94cddfaf9b7e5fa5b3bdb0097d98c66a336e878f0952ff3cb0e",
    "calculators/calculator.py": "4f50f2f22cbb1698a00a2328a9d8cb0a430b8a8c672ca88aeed195e924f38841",
    "calculators/hf_hub.py": "7858467c6a9fa172b2c2dd22ca5141459d906cb4c309d7c19dc0a6f045d7e185",
    "calculators/model_registry.py": "9e2f61fcfff3024b7cc20e561061666fd6ffeabbacaa6033b30ad745f83d05d8",
    "calculators/model_registry.yaml": "f0e9d4073793a0ce4793851abb266ef514b9f0e75befc34dadda7ae49115f9ce",
    "cli.py": "b9cb840ab864ff68e5914efdd5a9d1474ddcdfb7844913b7704319f637fade15",
    "config.py": "0232457935c9e08aae689d778de70a872b5ecf409b1d138ace5c3271f63b9360",
    "constants.py": "4fa79bfc21b9764c6e520db65dff16885d22181f892e82cdcc7abd47c491ceb4",
    "data/__init__.py": "ac3f385b6a9ace35d46dd71806e1e4326629e387c6445312dea9efd4e28fb10b",
    "data/sgdataset.py": "815099044279ce74ed4b2a9f06c38e17304d7173a827f93ea55bf35ad4fe9262",
    "dftd3_data.pt": "83d1d0ff69de90cb3ded49c80c8151786106f2aea7eb3425abe625e4b1909ce2",
    "interfaces/__init__.py": "5400c5c456daa2ceb58ad0c9799a923e859357fb18ad0cabad603a63d2defd11",
    "kernels/__init__.py": "7d692e675f0fde3bfea8d1815ea4fee2dceb41e6e6d3e993cb742e2843ef39be",
    "kernels/conv_sv_2d_sp_wp.py": "b67d5a342f602f91514785c13be20a00ab9c5840f2995729d518e5668a069e12",
    "models/__init__.py": "fdc6de2603dfdf9c505bb9e26f5ce7d15784be04f12b8997464c18178ef6b5db",
    "models/aimnet2.py": "a6fcfb549ab3906165f1721a2de1332078f63c5d17dd276fe6d67887e1a9658f",
    "models/aimnet2.yaml": "566234b9bed46e2266f37df3c02b422acac121a5d92a326c097c57c6eb7074b2",
    "models/aimnet2_dftd3_wb97m.yaml": "1a4f3af7fd914085f3947c1135b7ec6a2e2ef703d7de803220d2b523bd5e0452",
    "models/aimnet2_rxn.yaml": "22325abf01cb8c47399dd8764c99666f1d51657cf870eda3e70f3b8fe77a27ba",
    "models/base.py": "f22ed8577cff74cc63e7b51897e231565733732f61f8377db9bf616b8d73bf37",
    "models/convert.py": "bf14b6b087ccfb033604e8255ed886d64d12e5ac864fc098ee747a776cb9c605",
    "models/utils.py": "4670c83ab1c37459526c60b9da0298252a76d2a79feffb571e1d99987c221d0f",
    "modules/__init__.py": "23ef617aa45d9d77b2a4c523cde43a6381004ccffd9f8345c820ba3a6e87d95c",
    "modules/aev.py": "c283deaa5a4cb3f9faccec6b04c457848ab421a07629e639c3b89391458efdd7",
    "modules/core.py": "e95b05a9add7534c3b785ea1743af3f5e269cb20afa228ff5f55d929a5aa239a",
    "modules/lr.py": "fe31354eede1dc4d8042ff5f0faab8d1b719b03a347e2227e2c7cdd912ac076f",
    "nbops.py": "4a6f78489e719aa8cac37d6b7ac09dfdcda6368aa8aec4984f88c299f51c7c50",
    "ops.py": "5e06d7cb2fe7e231ec336282b505bdf1236f082c55389c2ace5f2abc5165ecda",
    "train/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "train/calc_sae.py": "87ecbc8d0554e8eed19b0f312565f70c73a916fd60b63170b59cd9c943a1cfd8",
    "train/default_train.yaml": "1401b77f7a450b6764154b953bf5c4fd407aac5afe4771a400275d9fb147ff9f",
    "train/export_model.py": "0c660911d2d7c4cdde3d68eb510abcd8fdd1d8c9b175dcf2414747758942d5c3",
    "train/loss.py": "1d249a819bcefe346fa2e2305f46472e238c7fdb8b817deb4a6b6b64c09c28e0",
    "train/metrics.py": "80260814082c39f986bf631cd86b5f39c22ad932d4eaae5cbd03d62ee8947523",
    "train/train.py": "5420bc9a3db2d3dc5fb023982be6fd615f914b35b377e541f708a839508e60c9",
    "train/utils.py": "9a8f31c31940864ad835cecdc766d080b807bab3a9c34509a678551c74609b7d",
}
MODEL_METADATA = {
    "format_version": 2, "cutoff": 5.0, "needs_coulomb": True,
    "needs_dispersion": True, "coulomb_mode": "sr_embedded",
    "coulomb_sr_rc": 4.599999904632568, "coulomb_sr_envelope": "exp",
    "d3_params": {"s6": 1.0, "s8": 1.5, "a1": 0.37, "a2": 4.1},
    "has_embedded_lr": True,
    "implemented_species": [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 33, 34, 35, 53],
}


def canonical_neighbor_graph_sha256(graph):
    """Hash connectivity without mistaking within-row packing order for edges.

    Only integer 2D nbmat* columns are sorted, on copies. Shapes, dtypes,
    dummy rows, padding/sentinels and duplicate multiplicities are retained.
    All other named arrays retain their exact order. Model inputs never change.
    """
    if not isinstance(graph, Mapping) or not graph:
        raise ValueError("Neighbor graph requires a nonempty named-array mapping.")
    canonical = {}
    neighbors = False
    for name, value in graph.items():
        if not isinstance(name, str):
            raise TypeError("Neighbor graph array names must be strings.")
        array = np.asarray(value)
        if name.startswith("nbmat"):
            if array.ndim != 2 or array.dtype.kind not in {"i", "u"}:
                raise ValueError("Neighbor nbmat layout must be an integer 2D array.")
            canonical[name] = np.sort(array.copy(), axis=1)
            neighbors = True
        else:
            canonical[name] = array
    if not neighbors:
        raise ValueError("Neighbor graph requires an integer 2D nbmat array.")
    return named_array_sha256(canonical)


def _source_inventory(root):
    paths = []
    root_stat = root.lstat()
    if not stat.S_ISDIR(root_stat.st_mode):
        raise AnalyticIdentityError("AIMNet2025 source inventory root is not a directory.")
    directories = {str(root): _directory_snapshot(str(root))}
    for path in root.rglob("*"):
        try:
            observed = path.lstat()
            if stat.S_ISLNK(observed.st_mode):
                raise AnalyticIdentityError("AIMNet2025 source inventory contains a symlink/foreign origin.")
            if stat.S_ISDIR(observed.st_mode):
                directories[str(path)] = _directory_snapshot(str(path))
            elif stat.S_ISREG(observed.st_mode) and path.suffix in SOURCE_SUFFIXES:
                paths.append(path)
        except FileNotFoundError:
            # A vanished bytecode entry is no longer part of the inventory.
            # Missing authoritative files still fail the exact inventory and
            # original-file checks; no file identity is refreshed here.
            continue
    return frozenset(path.relative_to(root).as_posix() for path in paths), directories


def _directory_identity(observed):
    return (*file_stat_identity(observed), observed.st_mode, observed.st_ctime_ns)


def _directory_snapshot(path):
    observed = os.lstat(path)
    entries = None
    if stat.S_ISDIR(observed.st_mode):
        # Directory timestamps can collide within one filesystem clock tick.
        # Inspect entries as well: new helpers/symlinks must be caught even then.
        with os.scandir(path) as iterator:
            entries = frozenset((entry.name, entry.is_symlink(),
                                 entry.is_dir(follow_symlinks=False),
                                 entry.is_file(follow_symlinks=False)) for entry in iterator)
    return _directory_identity(observed), entries


def _anchor_identity(observed):
    # Unrelated entries in a parent directory must not invalidate a package.
    # Replacing that directory or retargeting a symlink must invalidate it.
    if stat.S_ISDIR(observed.st_mode):
        return observed.st_dev, observed.st_ino, stat.S_IFDIR
    return _directory_identity(observed)


def _require_source_inventory(root, expected):
    actual, directories = _source_inventory(root)
    if actual != frozenset(expected):
        raise AnalyticIdentityError(
            "AIMNet2025 source inventory changed: "
            f"added={sorted(actual - set(expected))}, missing={sorted(set(expected) - actual)}."
        )
    return directories


def _executing_aimnet_modules():
    return tuple((name, module) for name, module in tuple(sys.modules.items())
                 if module is not None and (name == "aimnet" or name.startswith("aimnet.")))


def _require_executing_sources(root, expected, modules=None):
    """Reject shadow packages/modules and foreign package search paths."""
    for name, module in _executing_aimnet_modules() if modules is None else modules:
        origin = getattr(getattr(module, "__spec__", None), "origin", None)
        filename = getattr(module, "__file__", None)
        stem = name.removeprefix("aimnet").lstrip(".").replace(".", "/")
        candidates = {f"{stem}.py", f"{stem}/__init__.py"} if stem else {"__init__.py"}
        if not origin or not filename:
            raise AnalyticIdentityError(f"AIMNet2025 executing source identity is missing: {name}.")
        path = Path(filename).resolve()
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise AnalyticIdentityError(f"AIMNet2025 executing source identity has a foreign origin: {name}.") from exc
        if relative not in expected or relative not in candidates or Path(origin).resolve() != path:
            raise AnalyticIdentityError(f"AIMNet2025 executing source identity mismatch: {name}.")
        search = getattr(module, "__path__", None)
        if search is not None and tuple(Path(value).resolve() for value in search) != (path.parent,):
            raise AnalyticIdentityError(f"AIMNet2025 executing package search-path identity changed: {name}.")


class BoundExternalSources:
    """One calculator's attested sources and incremental filesystem guards.

    File byte identities never refresh. Directory-entry changes trigger a full
    inventory verification before refreshing directory tokens; ordinary pyc
    churn is allowed, new mathematical source and all symlinks are not.
    Existing module metadata is immutable. New lazy imports are validated once.
    No global cache, polling interval, or admission bypass is involved.
    """

    def __init__(self, root, files, directories):
        self.root = root
        self.files = files
        self._file_paths = frozenset(file.path for file in files)
        self._expected = frozenset(Path(path).relative_to(root).as_posix() for path in self._file_paths)
        self._directories = directories
        self._modules = {}
        self._anchors = {}
        for parent in root.parents:
            self._bind_anchor(str(parent))
        self.require_unchanged()

    def _bind_anchor(self, path):
        if path in self._file_paths or path in self._directories:
            return
        current = _anchor_identity(os.lstat(path))
        if path in self._anchors and self._anchors[path] != current:
            raise AnalyticIdentityError("AIMNet2025 executing source origin/alias identity changed.")
        self._anchors[path] = current

    def _require_files(self):
        for file in self.files:
            current = os.lstat(file.path)
            if not stat.S_ISREG(current.st_mode) or file_stat_identity(current) != file.stat_identity:
                raise AnalyticIdentityError("AIMNet2025 bound source file identity changed.")

    def _require_modules(self):
        observed = _executing_aimnet_modules()
        if self._modules.keys() - {name for name, _ in observed}:
            raise AnalyticIdentityError("AIMNet2025 executing module identity was removed.")
        for name, module in observed:
            filename = getattr(module, "__file__", None)
            origin = getattr(getattr(module, "__spec__", None), "origin", None)
            search = getattr(module, "__path__", None)
            search = None if search is None else tuple(search)
            raw_paths = (filename, origin, *(search or ()))
            cwd = os.getcwd() if any(path and not os.path.isabs(path) for path in raw_paths) else None
            token = (filename, origin, search, cwd)
            if name in self._modules:
                previous, previous_token = self._modules[name]
                if module is not previous or token != previous_token:
                    raise AnalyticIdentityError("AIMNet2025 executing module identity changed.")
                continue
            _require_executing_sources(self.root, self._expected, ((name, module),))
            for raw in raw_paths:
                path = Path(raw).absolute()
                for entry in (path, *path.parents):
                    self._bind_anchor(str(entry))
            self._modules[name] = (module, token)

    def require_unchanged(self):
        """Check every access, including cached ASE returns, without rewalking."""
        try:
            self._require_files()
            for path, expected in self._anchors.items():
                if _anchor_identity(os.lstat(path)) != expected:
                    raise AnalyticIdentityError("AIMNet2025 executing source origin/alias identity changed.")
            changed = False
            for path, expected in self._directories.items():
                try:
                    current = _directory_snapshot(path)
                except FileNotFoundError:
                    current = None  # A removed pycache is verified by the full rescan.
                changed |= current != expected
            if changed:
                directories = _require_source_inventory(self.root, self._expected)
                self._require_files()
                self._directories = directories
            self._require_modules()
        except OSError as exc:
            raise AnalyticIdentityError("AIMNet2025 bound source identity is no longer readable.") from exc


def require_external_sources(binding):
    """Keep source checking at every boundary; only unchanged metadata is cached."""
    if not isinstance(binding, BoundExternalSources):
        raise AnalyticIdentityError("AIMNet2025 complete source identity bindings are missing.")
    binding.require_unchanged()


def bind_external_runtime():
    """Hash the full authoritative package inventory, retaining live guards."""
    versions = {}
    try:
        for package, expected in PACKAGES.items():
            actual = metadata.version(package)
            if actual != expected:
                raise RuntimeError(
                    f"AIMNet2025 runtime identity mismatch: {package}=={actual}, "
                    f"expected {expected}. {INSTALL_HINT}"
                )
            versions[package] = actual
        root = Path(metadata.distribution("aimnet").locate_file("aimnet")).resolve()
    except metadata.PackageNotFoundError as exc:
        raise ImportError(INSTALL_HINT) from exc
    directories = _require_source_inventory(root, SOURCE_SHA256)
    files = []
    for relative, expected in sorted(SOURCE_SHA256.items()):
        bound = CheckpointFileIdentity.bind(root / relative)
        if bound.sha256 != expected:
            raise RuntimeError(f"AIMNet2025 external source identity mismatch: aimnet/{relative}.")
        files.append(bound)
    binding = BoundExternalSources(root, tuple(files), directories)
    return {"packages": versions, "source_sha256": dict(SOURCE_SHA256)}, binding
