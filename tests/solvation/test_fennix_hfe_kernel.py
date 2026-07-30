"""Fail-closed tests for the native FeNNix alchemical mechanics kernel.

The fake Hamiltonian below is executable mechanics documentation only.  It is
not HFE sampling, an experimental accuracy panel, GPU admission, or a speed
benchmark.
"""

from __future__ import annotations

from contextlib import nullcontext
import inspect
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from maple.function.solvfe.fennix_hfe import (
    FeNNixAlchemicalKernel,
    FeNNixAlchemicalSystem,
    FeNNixKernelResult,
    FeNNixLambdaState,
    upcast_parameter_tree_float64,
    validate_fennix_alchemical_system,
)
from maple.function.solvfe.fennix_hfe import kernel as kernel_module
from maple.function.solvfe.fennix_hfe.types import (
    FENNIX_KERNEL_SCIENTIFIC_SCOPE,
    FeNNixAlchemicalParameters,
)
from maple.function.calculator.fennol import _fennix_bio1_calculator as adapter_module

_SHA = "a" * 64


def _valid_system(**changes: object) -> FeNNixAlchemicalSystem:
    system = FeNNixAlchemicalSystem(
        atomic_numbers=(6, 8, 1, 1, 8, 1, 1),
        coordinates_angstrom=(
            (2.0, 2.0, 2.0),
            (7.0, 7.0, 7.0),
            (7.9, 7.0, 7.0),
            (6.7, 7.8, 7.0),
            (12.0, 12.0, 12.0),
            (12.9, 12.0, 12.0),
            (11.7, 12.8, 12.0),
        ),
        cell_angstrom=((18.0, 0.0, 0.0), (0.0, 18.0, 0.0), (0.0, 0.0, 18.0)),
        molecule_ids=(0, 1, 1, 1, 2, 2, 2),
        solute_atom_indices=(0,),
    )
    return replace(system, **changes)


@pytest.mark.parametrize(
    ("progress", "lambda_e", "lambda_v", "active"),
    [
        (0.0, 0.0, 0.0, "lambda_v"),
        (0.25, 0.0, 0.5, "lambda_v"),
        (0.5, 0.0, 1.0, "lambda_v"),
        (0.75, 0.5, 1.0, "lambda_e"),
        (1.0, 1.0, 1.0, "lambda_e"),
    ],
)
def test_progress_maps_to_two_paper_lambda_segments(
    progress, lambda_e, lambda_v, active
):
    state = FeNNixLambdaState.from_progress(progress)

    assert (state.lambda_e, state.lambda_v) == (lambda_e, lambda_v)
    assert state.active_derivative == active
    assert state.active_derivative_scale == 2.0


@pytest.mark.parametrize("progress", [-np.inf, -1e-12, 1.0 + 1e-12, np.inf, np.nan])
def test_progress_outside_closed_unit_interval_is_rejected(progress):
    with pytest.raises(ValueError, match="progress"):
        FeNNixLambdaState.from_progress(progress)


def test_half_progress_has_one_canonical_shared_state():
    exact = FeNNixLambdaState.from_progress(0.5)
    left = FeNNixLambdaState.from_progress(np.nextafter(0.5, 0.0))
    right = FeNNixLambdaState.from_progress(np.nextafter(0.5, 1.0))

    assert (exact.lambda_e, exact.lambda_v) == (0.0, 1.0)
    assert np.allclose((left.lambda_e, left.lambda_v), (0.0, 1.0), atol=2e-15)
    assert np.allclose((right.lambda_e, right.lambda_v), (0.0, 1.0), atol=2e-15)


@pytest.mark.parametrize(
    ("progress", "d_e", "d_v", "expected"),
    [(0.25, 11.0, 3.0, 6.0), (0.75, 11.0, 3.0, 22.0)],
)
def test_lambda_state_chain_derivative_selects_only_active_segment(
    progress, d_e, d_v, expected
):
    state = FeNNixLambdaState.from_progress(progress)

    assert state.chain_derivative(d_e, d_v) == expected


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"total_charge_e": 1}, "neutral"),
        ({"solute_charge_e": -1}, "neutral"),
        ({"multiplicity": 2}, "closed-shell"),
        ({"solute_multiplicity": 2}, "closed-shell"),
        ({"pbc": (True, False, True)}, "PBC"),
        ({"cell_angstrom": np.diag([16.9, 18.0, 18.0])}, "cell height"),
        ({"solute_atom_indices": ()}, "solute"),
        ({"solute_atom_indices": (0, 4)}, "one molecule"),
        ({"molecule_ids": np.array([0, 1, 1, 2, 2, 2, 2])}, "H2O"),
        ({"molecule_ids": np.array([0, 1, 1, 1, 2, 2, 0])}, "complete"),
    ],
)
def test_system_validation_rejects_out_of_scope_system(change, message):
    with pytest.raises(ValueError, match=message):
        validate_fennix_alchemical_system(_valid_system(**change))


def test_system_validation_rejects_nonwater_solvent_molecule():
    system = _valid_system(
        atomic_numbers=np.array([6, 8, 1, 1, 7, 1, 1], dtype=np.int32)
    )

    with pytest.raises(ValueError, match="H2O"):
        validate_fennix_alchemical_system(system)


def test_system_validation_rejects_neutral_singlet_with_odd_electron_count():
    system = _valid_system(
        atomic_numbers=np.array([7, 8, 1, 1, 8, 1, 1], dtype=np.int32)
    )

    with pytest.raises(ValueError, match="even electron"):
        validate_fennix_alchemical_system(system)


def test_system_validation_normalizes_the_admitted_periodic_water_scope():
    validated = validate_fennix_alchemical_system(_valid_system())

    assert validated.atomic_numbers == (6, 8, 1, 1, 8, 1, 1)
    assert validated.solute_atom_indices == (0,)
    assert validated.pbc == (True, True, True)
    assert np.asarray(validated.coordinates_angstrom).dtype == np.float64
    assert np.asarray(validated.cell_angstrom).dtype == np.float64


class _TreeUtil:
    @staticmethod
    def tree_map(function, tree):
        if isinstance(tree, dict):
            return {
                key: _TreeUtil.tree_map(function, value) for key, value in tree.items()
            }
        if isinstance(tree, tuple):
            return tuple(_TreeUtil.tree_map(function, value) for value in tree)
        if isinstance(tree, list):
            return [_TreeUtil.tree_map(function, value) for value in tree]
        return function(tree)

    @staticmethod
    def tree_leaves(tree):
        if isinstance(tree, dict):
            return [
                leaf
                for key in sorted(tree)
                for leaf in _TreeUtil.tree_leaves(tree[key])
            ]
        if isinstance(tree, (tuple, list)):
            return [leaf for value in tree for leaf in _TreeUtil.tree_leaves(value)]
        return [tree]

    @staticmethod
    def tree_flatten(tree):
        return _TreeUtil.tree_leaves(tree), "fake-tree-definition"

    @staticmethod
    def tree_flatten_with_path(tree):
        leaves: list[tuple[tuple[str, ...], object]] = []

        def visit(value, path=()):
            if isinstance(value, dict):
                for key in sorted(value):
                    visit(value[key], path + (str(key),))
            elif isinstance(value, (tuple, list)):
                for index, child in enumerate(value):
                    visit(child, path + (str(index),))
            else:
                leaves.append((path, value))

        visit(tree)
        return leaves, "fake-tree-definition"

    @staticmethod
    def keystr(path):
        return "/".join(path)


class _FakeJax:
    tree_util = _TreeUtil()

    @staticmethod
    def devices(platform):
        return [f"{platform}:0"]

    @staticmethod
    def default_device(_device):
        return nullcontext()

    @staticmethod
    def block_until_ready(value):
        return value

    @staticmethod
    def device_put(value, _device):
        return _TreeUtil.tree_map(
            lambda leaf: (
                np.array(leaf, copy=True) if isinstance(leaf, np.ndarray) else leaf
            ),
            value,
        )

    @staticmethod
    def jit(function):
        return function

    @staticmethod
    def make_jaxpr(_function):
        return lambda *_args, **_kwargs: "fake-jaxpr f64[]"


@pytest.fixture
def fake_jax(monkeypatch):
    fake = _FakeJax()
    monkeypatch.setattr(kernel_module, "_jax_modules", lambda: (fake, np))
    return fake


def test_parameter_upcast_is_float64_and_deterministic(fake_jax):
    tree = {
        "weights": np.array([1.25, -2.5], dtype=np.float32),
        "nested": (np.array([3.0], dtype=np.float64), np.array([2], dtype=np.int32)),
    }

    first, first_receipt = upcast_parameter_tree_float64(
        tree, original_checkpoint_sha256=_SHA
    )
    second, second_receipt = upcast_parameter_tree_float64(
        tree, original_checkpoint_sha256=_SHA
    )

    assert first["weights"].dtype == np.float64
    assert first["nested"][0].dtype == np.float64
    assert first["nested"][1].dtype == np.int32
    assert first_receipt.derived_parameter_fingerprint == (
        second_receipt.derived_parameter_fingerprint
    )
    assert first_receipt.original_checkpoint_sha256 == _SHA


def test_host_upcast_preserves_float32_subnormal_exactly(fake_jax):
    subnormal = np.nextafter(np.float32(0.0), np.float32(1.0))
    source = {"subnormal": np.array([subnormal], dtype=np.float32)}

    derived, receipt = upcast_parameter_tree_float64(
        source, original_checkpoint_sha256=_SHA
    )

    assert source["subnormal"].dtype == np.float32
    assert source["subnormal"][0] == subnormal
    assert derived["subnormal"].dtype == np.float64
    assert derived["subnormal"][0] == np.float64(subnormal)
    assert receipt.original_checkpoint_sha256 == _SHA


def test_host_derived_fingerprint_is_device_independent(fake_jax):
    source = {
        "weights": np.array(
            [np.nextafter(np.float32(0.0), np.float32(1.0)), -2.5],
            dtype=np.float32,
        )
    }
    host_tree, receipt = upcast_parameter_tree_float64(
        source, original_checkpoint_sha256=_SHA
    )

    cpu_tree = fake_jax.device_put(host_tree, "cpu:0")
    gpu_tree = fake_jax.device_put(host_tree, "gpu:0")

    assert kernel_module._parameter_tree_fingerprint(cpu_tree) == (
        receipt.derived_parameter_fingerprint
    )
    assert kernel_module._parameter_tree_fingerprint(gpu_tree) == (
        receipt.derived_parameter_fingerprint
    )


def test_parameter_fingerprint_changes_with_parameter_content(fake_jax):
    _, first = upcast_parameter_tree_float64(
        {"weights": np.array([1.25, -2.5], dtype=np.float32)},
        original_checkpoint_sha256=_SHA,
    )
    _, second = upcast_parameter_tree_float64(
        {"weights": np.array([1.25, -2.0], dtype=np.float32)},
        original_checkpoint_sha256=_SHA,
    )

    assert first.derived_parameter_fingerprint != second.derived_parameter_fingerprint


@pytest.mark.parametrize("dtype", [np.float16, "bfloat16"])
def test_parameter_upcast_rejects_reduced_precision_checkpoint_leaf(fake_jax, dtype):
    if dtype == "bfloat16":
        ml_dtypes = pytest.importorskip("ml_dtypes")
        dtype = ml_dtypes.bfloat16

    with pytest.raises((ValueError, RuntimeError), match="float16|bfloat16|reduced"):
        upcast_parameter_tree_float64(
            {"weights": np.array([1.0], dtype=dtype)},
            original_checkpoint_sha256=_SHA,
        )


@pytest.mark.parametrize(
    "field",
    ["graph_softcore_v_angstrom", "repulsion_softcore_angstrom"],
)
@pytest.mark.parametrize(
    "value", [0.0, -0.5, np.nan, np.inf, -np.inf, True, "0.5", None]
)
def test_alchemical_parameters_reject_invalid_softcore_distance(field, value):
    with pytest.raises(ValueError, match="finite positive"):
        FeNNixAlchemicalParameters(**{field: value})


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.5, 2.0, "2", None])
def test_alchemical_parameters_reject_invalid_repulsion_power(value):
    with pytest.raises(ValueError, match="positive integer"):
        FeNNixAlchemicalParameters(repulsion_power_m=value)


class _FakePreprocessing:
    def init_with_output(self, raw):
        return object(), None

    def process(self, _state, raw):
        return raw


class _ToyModel:
    energy_unit = "eV"
    variables = {"weights": np.array([1.25, -2.5], dtype=np.float32)}
    preprocessing = _FakePreprocessing()

    @staticmethod
    def get_gradient_function(*_names, **_options):
        def gradient(_variables, inputs):
            coordinates = np.asarray(inputs["coordinates"], dtype=np.float64)
            cell = np.asarray(inputs["cells"], dtype=np.float64)
            lambda_e = np.asarray(inputs["alch_elambda"], dtype=np.float64)
            lambda_v = np.asarray(inputs["alch_vlambda"], dtype=np.float64)
            graph_softcore = np.asarray(inputs["alch_softcore_v"])
            repulsion_softcore = np.asarray(inputs["alch_softcore_rep"])
            repulsion_power = np.asarray(inputs["alch_m"])
            if (
                graph_softcore.dtype != np.float64
                or repulsion_softcore.dtype != np.float64
            ):
                raise TypeError(
                    "toy softcore distances must be explicit float64 inputs"
                )
            if repulsion_power.dtype.kind not in "iu":
                raise TypeError("toy softcore power must be an explicit integer input")
            energy = (
                0.5 * np.sum(coordinates**2)
                + 0.25 * np.sum(cell**2)
                + 3.0 * lambda_e
                + 5.0 * lambda_v**2
                + lambda_e * lambda_v
                + 0.0 * (graph_softcore + repulsion_softcore + repulsion_power)
            )
            derivatives = {
                "coordinates": coordinates.copy(),
                "cells": 0.5 * cell,
                "alch_elambda": np.asarray(3.0 + lambda_v, dtype=np.float64),
                "alch_vlambda": np.asarray(
                    10.0 * lambda_v + lambda_e, dtype=np.float64
                ),
            }
            return np.asarray(energy, dtype=np.float64), derivatives, {}

        return gradient


def _runtime_receipt(**changes: object) -> dict[str, object]:
    receipt: dict[str, object] = {
        "distribution": "FeNNol",
        "version": kernel_module.FENNOL_DISTRIBUTION_VERSION,
        "ase_sha256": adapter_module.FENNOL_ASE_SHA256,
        "preprocessing_sha256": kernel_module.FENNOL_PREPROCESSING_SHA256,
    }
    receipt.update(changes)
    return receipt


def _runtime_package_versions(platform: str) -> dict[str, str]:
    versions = dict(kernel_module.FENNIX_RUNTIME_PACKAGE_VERSIONS)
    versions.update(
        kernel_module.FENNIX_GPU_RUNTIME_PACKAGE_VERSIONS
        if platform == "gpu"
        else {name: "N/A" for name in kernel_module.FENNIX_GPU_RUNTIME_PACKAGE_VERSIONS}
    )
    return versions


def test_public_kernel_constructor_has_no_runtime_injection_hooks():
    parameters = inspect.signature(FeNNixAlchemicalKernel).parameters

    assert {
        "model_loader",
        "runtime_validator",
        "runtime_package_validator",
    }.isdisjoint(parameters)
    assert set(parameters) == {
        "checkpoint_path",
        "device",
        "allow_unadmitted_gpu_mechanics",
        "alchemical_parameters",
    }


def _configure_toy_kernel_dependencies(monkeypatch) -> None:
    monkeypatch.setattr(
        kernel_module,
        "_sha256",
        lambda _path: kernel_module.FENNIX_BIO1_CHECKPOINTS["medium"]["sha256"],
    )
    monkeypatch.setattr(kernel_module, "_load_card", lambda _path: {})
    monkeypatch.setattr(kernel_module, "_validate_official_card", lambda _card: None)
    monkeypatch.setattr(
        kernel_module, "_official_model_loader", lambda _path: _ToyModel()
    )
    monkeypatch.setattr(kernel_module, "_official_runtime_validator", _runtime_receipt)
    monkeypatch.setattr(
        kernel_module,
        "_official_runtime_package_versions",
        _runtime_package_versions,
    )
    monkeypatch.setattr(
        kernel_module, "_validate_runtime_receipt", lambda value: dict(value)
    )
    monkeypatch.setattr(
        kernel_module,
        "_validate_native_runtime_sources",
        lambda: dict(kernel_module.FENNIX_NATIVE_RUNTIME_SHA256),
    )
    monkeypatch.setattr(
        kernel_module,
        "_validate_fennol_distribution_tree",
        lambda: (
            kernel_module.FENNIX_PACKAGE_TREE_SHA256,
            kernel_module.FENNIX_PACKAGE_TREE_FILE_COUNT,
        ),
    )
    monkeypatch.setattr(
        kernel_module,
        "_promote_fixed_species_encoding_float64",
        lambda _model: "f" * 64,
    )
    original_fingerprint = kernel_module._parameter_tree_fingerprint(
        _ToyModel.variables
    )
    derived_tree, _receipt = upcast_parameter_tree_float64(
        _ToyModel.variables,
        original_checkpoint_sha256=kernel_module.FENNIX_BIO1_CHECKPOINTS["medium"][
            "sha256"
        ],
    )
    monkeypatch.setattr(
        kernel_module,
        "FENNIX_BIO1M_ORIGINAL_PARAMETER_TREE_FINGERPRINT",
        original_fingerprint,
    )
    monkeypatch.setattr(
        kernel_module,
        "FENNIX_BIO1M_DERIVED_PARAMETER_TREE_FINGERPRINT",
        kernel_module._parameter_tree_fingerprint(derived_tree),
    )
    monkeypatch.setattr(
        kernel_module,
        "FENNIX_BIO1M_FIXED_SPECIES_ENCODING_FLOAT64_SHA256",
        "f" * 64,
    )


def _build_toy_kernel(
    tmp_path: Path,
    monkeypatch,
    parameters: FeNNixAlchemicalParameters | None = None,
) -> FeNNixAlchemicalKernel:
    checkpoint = tmp_path / "fennix-bio1M.fnx"
    checkpoint.write_bytes(b"original checkpoint remains unchanged")
    before = checkpoint.read_bytes()
    _configure_toy_kernel_dependencies(monkeypatch)

    kernel = FeNNixAlchemicalKernel(
        checkpoint,
        alchemical_parameters=parameters,
    )

    assert checkpoint.read_bytes() == before
    return kernel


@pytest.fixture
def toy_kernel(tmp_path: Path, monkeypatch, fake_jax) -> FeNNixAlchemicalKernel:
    return _build_toy_kernel(tmp_path, monkeypatch)


def test_constructor_uses_private_official_runtime_helpers(
    tmp_path, monkeypatch, fake_jax
):
    _configure_toy_kernel_dependencies(monkeypatch)
    calls: list[tuple[str, str]] = []

    def load(path):
        calls.append(("model", path))
        return _ToyModel()

    def runtime():
        calls.append(("runtime", "official"))
        return _runtime_receipt()

    def packages(platform):
        calls.append(("packages", platform))
        return _runtime_package_versions(platform)

    monkeypatch.setattr(kernel_module, "_official_model_loader", load)
    monkeypatch.setattr(kernel_module, "_official_runtime_validator", runtime)
    monkeypatch.setattr(kernel_module, "_official_runtime_package_versions", packages)
    checkpoint = tmp_path / "fennix-bio1M.fnx"
    checkpoint.write_bytes(b"checkpoint")

    FeNNixAlchemicalKernel(checkpoint)

    assert calls == [
        ("runtime", "official"),
        ("packages", "cpu"),
        ("model", str(checkpoint)),
    ]


def _materialize_pinned_native_sources(tmp_path: Path, *, omit: str | None = None):
    for relative_path in kernel_module.FENNIX_NATIVE_RUNTIME_SHA256:
        if relative_path == omit:
            continue
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative_path.encode())

    class Distribution:
        @staticmethod
        def locate_file(relative_path):
            return tmp_path / str(relative_path)

    return Distribution()


def test_native_runtime_identity_requires_nlh_coefficients_file(tmp_path, monkeypatch):
    relative = "fennol/models/physics/nlh_coeffs.dat"
    distribution = _materialize_pinned_native_sources(tmp_path, omit=relative)
    monkeypatch.setattr(
        kernel_module.metadata, "distribution", lambda _name: distribution
    )
    monkeypatch.setattr(
        kernel_module,
        "_sha256",
        lambda path: kernel_module.FENNIX_NATIVE_RUNTIME_SHA256[
            Path(path).relative_to(tmp_path).as_posix()
        ],
    )

    with pytest.raises(ImportError, match="nlh_coeffs.dat"):
        kernel_module._validate_native_runtime_sources()


def test_native_runtime_identity_rejects_nlh_coefficients_hash_drift(
    tmp_path, monkeypatch
):
    relative = "fennol/models/physics/nlh_coeffs.dat"
    distribution = _materialize_pinned_native_sources(tmp_path)
    monkeypatch.setattr(
        kernel_module.metadata, "distribution", lambda _name: distribution
    )

    def digest(path):
        observed = Path(path).relative_to(tmp_path).as_posix()
        if observed == relative:
            return "0" * 64
        return kernel_module.FENNIX_NATIVE_RUNTIME_SHA256[observed]

    monkeypatch.setattr(kernel_module, "_sha256", digest)

    with pytest.raises(ValueError, match="nlh_coeffs.dat"):
        kernel_module._validate_native_runtime_sources()


def test_fennol_distribution_tree_rejects_missing_or_extra_file(tmp_path, monkeypatch):
    path = tmp_path / "fennol" / "unexpected.py"
    path.parent.mkdir(parents=True)
    path.write_text("unexpected", encoding="utf-8")

    class Distribution:
        files = (Path("fennol/unexpected.py"),)

        @staticmethod
        def locate_file(relative_path):
            return tmp_path / str(relative_path)

    monkeypatch.setattr(
        kernel_module.metadata, "distribution", lambda _name: Distribution()
    )

    with pytest.raises(ValueError, match="package source/data tree mismatch"):
        kernel_module._validate_fennol_distribution_tree()


@pytest.mark.parametrize(
    ("constant", "message"),
    [
        ("FENNIX_BIO1M_ORIGINAL_PARAMETER_TREE_FINGERPRINT", "original parameter-tree"),
        ("FENNIX_BIO1M_DERIVED_PARAMETER_TREE_FINGERPRINT", "derived.*parameter-tree"),
        ("FENNIX_BIO1M_FIXED_SPECIES_ENCODING_FLOAT64_SHA256", "species-encoding"),
    ],
)
def test_kernel_rejects_pinned_tree_or_species_encoding_constant_drift(
    tmp_path, monkeypatch, fake_jax, constant, message
):
    checkpoint = tmp_path / "fennix-bio1M.fnx"
    checkpoint.write_bytes(b"checkpoint")
    _configure_toy_kernel_dependencies(monkeypatch)
    monkeypatch.setattr(kernel_module, constant, "0" * 64)

    with pytest.raises(ValueError, match=message):
        FeNNixAlchemicalKernel(
            checkpoint,
        )


def test_raw_inputs_bind_explicit_softcore_values_and_dtypes(
    tmp_path, monkeypatch, fake_jax
):
    parameters = FeNNixAlchemicalParameters(
        graph_softcore_v_angstrom=0.75,
        repulsion_softcore_angstrom=0.875,
        repulsion_power_m=3,
    )
    kernel = _build_toy_kernel(tmp_path, monkeypatch, parameters)

    raw = kernel._raw_inputs(validate_fennix_alchemical_system(_valid_system()))

    assert {"alch_softcore_v", "alch_softcore_rep", "alch_m"}.issubset(raw)
    assert raw["alch_softcore_v"].dtype == np.float64
    assert raw["alch_softcore_rep"].dtype == np.float64
    assert raw["alch_m"].dtype.kind in "iu"
    assert raw["alch_softcore_v"].item() == 0.75
    assert raw["alch_softcore_rep"].item() == 0.875
    assert raw["alch_m"].item() == 3


def test_identity_and_result_bind_softcore_values_and_provenance(
    tmp_path, monkeypatch, fake_jax
):
    parameters = FeNNixAlchemicalParameters(
        graph_softcore_v_angstrom=0.75,
        repulsion_softcore_angstrom=0.875,
        repulsion_power_m=3,
    )
    kernel = _build_toy_kernel(tmp_path, monkeypatch, parameters)

    result = kernel.evaluate(_valid_system(), 0.25)

    assert kernel.identity.alchemical_parameters == parameters
    assert result.alchemical_parameters == parameters
    assert parameters.graph_softcore_provenance == "pinned_fennol_source_default"
    assert parameters.repulsion_power_provenance == "pinned_fennol_source_default"
    assert parameters.repulsion_softcore_provenance == (
        "maple_output_blind_reconstruction_not_paper_exact"
    )
    assert parameters.scientific_scope == (
        "maple_owned_softcore_reconstruction_mechanics_not_paper_reproduction"
    )


def test_identity_rejects_softcore_value_or_provenance_drift(toy_kernel):
    expected = toy_kernel.identity
    drifted = FeNNixAlchemicalParameters(
        graph_softcore_v_angstrom=0.75,
        repulsion_softcore_angstrom=0.5,
        repulsion_power_m=2,
    )

    with pytest.raises(ValueError, match="alchemical reconstruction"):
        expected.verify_observed(replace(expected, alchemical_parameters=drifted))


@pytest.mark.parametrize(
    "field",
    [
        "graph_softcore_provenance",
        "repulsion_softcore_provenance",
        "repulsion_power_provenance",
        "scientific_scope",
    ],
)
def test_identity_rejects_softcore_provenance_drift(toy_kernel, field):
    expected = toy_kernel.identity
    drifted = FeNNixAlchemicalParameters()
    object.__setattr__(drifted, field, "wrong")

    with pytest.raises(ValueError, match="alchemical reconstruction"):
        expected.verify_observed(replace(expected, alchemical_parameters=drifted))


def test_kernel_identity_pins_checkpoint_source_runtime_and_float64_tree(toy_kernel):
    identity = toy_kernel.identity

    assert (
        identity.checkpoint_sha256
        == kernel_module.FENNIX_BIO1_CHECKPOINTS["medium"]["sha256"]
    )
    assert identity.source_revision == kernel_module.FENNIX_BIO1_SOURCE_REVISION
    assert identity.checkpoint_source_revision == kernel_module.FENNIX_BIO1_PMC_REVISION
    assert identity.runtime_distribution == "FeNNol"
    assert identity.runtime_version == kernel_module.FENNOL_DISTRIBUTION_VERSION
    assert identity.runtime_source_sha256 == kernel_module.FENNIX_NATIVE_RUNTIME_SHA256
    assert (
        identity.fennol_package_tree_sha256 == kernel_module.FENNIX_PACKAGE_TREE_SHA256
    )
    assert identity.fennol_package_tree_file_count == (
        kernel_module.FENNIX_PACKAGE_TREE_FILE_COUNT
    )
    assert identity.runtime_source_sha256["fennol/models/physics/nlh_coeffs.dat"] == (
        "9f6ad25db062dec6552e6a2132d17484da2e4bfd30e70611a94a8a886cd1e32a"
    )
    assert identity.repulsion_nlh_coefficients_provenance == (
        "pinned_nlh_coeffs_data_float32_origin_promoted_to_float64_for_model_execution"
    )
    assert len(identity.original_parameter_tree_fingerprint) == 64
    assert len(identity.derived_parameter_tree_fingerprint) == 64
    assert identity.runtime_package_versions == _runtime_package_versions("cpu")
    assert identity.fixed_species_encoding_float64_sha256 == "f" * 64
    assert identity.jax_enable_x64 is True
    assert identity.matmul_precision == "highest"
    assert identity.tf32_enabled is False


def test_identity_rejects_repulsion_coefficient_provenance_drift(toy_kernel):
    expected = toy_kernel.identity
    observed = replace(expected)
    object.__setattr__(observed, "repulsion_nlh_coefficients_provenance", "wrong")

    with pytest.raises(ValueError, match="coefficient provenance"):
        expected.verify_observed(observed)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("fennol_package_tree_sha256", "0" * 64),
        ("fennol_package_tree_file_count", 70),
    ],
)
def test_identity_rejects_fennol_package_tree_drift(toy_kernel, field, value):
    observed = replace(toy_kernel.identity)
    object.__setattr__(observed, field, value)

    with pytest.raises(ValueError, match="package tree"):
        toy_kernel.identity.verify_observed(observed)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("checkpoint_sha256", "0" * 64, "checkpoint"),
        ("source_revision", "wrong", "source"),
        ("runtime_version", "wrong", "runtime"),
        ("runtime_source_sha256", {"wrong.py": "0" * 64}, "runtime source"),
        ("runtime_package_versions", {"jax": "wrong"}, "runtime package"),
        ("derived_parameter_tree_fingerprint", "wrong", "fingerprint"),
        ("fixed_species_encoding_float64_sha256", "wrong", "species encoding"),
        ("jax_enable_x64", False, "x64|float64"),
        ("matmul_precision", "default", "highest"),
        ("tf32_enabled", True, "TF32"),
    ],
)
def test_kernel_identity_rejects_any_observed_drift(toy_kernel, field, value, message):
    expected = toy_kernel.identity

    with pytest.raises(ValueError, match=message):
        expected.verify_observed(replace(expected, **{field: value}))


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16", "tf32"])
def test_kernel_identity_rejects_non_float64_compute_dtype(toy_kernel, dtype):
    with pytest.raises(ValueError, match="float64|TF32"):
        toy_kernel.identity.verify_compute_dtypes(("float64", dtype))


def test_kernel_rejects_checkpoint_hash_drift(tmp_path, monkeypatch, fake_jax):
    checkpoint = tmp_path / "fennix-bio1M.fnx"
    checkpoint.write_bytes(b"wrong checkpoint")
    monkeypatch.setattr(kernel_module, "_sha256", lambda _path: "0" * 64)

    with pytest.raises(ValueError, match="checkpoint"):
        FeNNixAlchemicalKernel(checkpoint)


def test_kernel_rejects_gpu_production_use_before_formal_panel(
    tmp_path, monkeypatch, fake_jax
):
    checkpoint = tmp_path / "fennix-bio1M.fnx"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(
        kernel_module,
        "_sha256",
        lambda _path: kernel_module.FENNIX_BIO1_CHECKPOINTS["medium"]["sha256"],
    )
    monkeypatch.setattr(kernel_module, "_load_card", lambda _path: {})
    monkeypatch.setattr(kernel_module, "_validate_official_card", lambda _card: None)
    monkeypatch.setattr(
        kernel_module, "_official_model_loader", lambda _path: _ToyModel()
    )
    monkeypatch.setattr(kernel_module, "_official_runtime_validator", _runtime_receipt)
    monkeypatch.setattr(
        kernel_module,
        "_official_runtime_package_versions",
        _runtime_package_versions,
    )
    monkeypatch.setattr(
        kernel_module, "_validate_runtime_receipt", lambda value: dict(value)
    )
    monkeypatch.setattr(
        kernel_module,
        "_validate_native_runtime_sources",
        lambda: dict(kernel_module.FENNIX_NATIVE_RUNTIME_SHA256),
    )
    monkeypatch.setattr(
        kernel_module,
        "_validate_fennol_distribution_tree",
        lambda: (
            kernel_module.FENNIX_PACKAGE_TREE_SHA256,
            kernel_module.FENNIX_PACKAGE_TREE_FILE_COUNT,
        ),
    )

    with pytest.raises(ValueError, match="ten-functional-group|admission"):
        FeNNixAlchemicalKernel(
            checkpoint,
            device="gpu",
        )


def test_kernel_returns_force_as_negative_coordinate_gradient(toy_kernel):
    system = _valid_system()

    result = toy_kernel.evaluate(system, 0.75)

    np.testing.assert_allclose(
        result.forces_ev_per_angstrom, -np.asarray(system.coordinates_angstrom)
    )


def test_kernel_virial_is_right_strain_derivative(toy_kernel):
    system = _valid_system()

    result = toy_kernel.evaluate(system, 0.75)

    expected_cell_gradient = 0.5 * np.asarray(system.cell_angstrom)
    np.testing.assert_allclose(
        result.cell_gradient_ev_per_angstrom, expected_cell_gradient
    )
    coordinates = np.asarray(system.coordinates_angstrom)
    expected_virial = (
        coordinates.T @ coordinates
        + np.asarray(system.cell_angstrom).T @ expected_cell_gradient
    )
    np.testing.assert_allclose(result.virial_ev, expected_virial)


def test_kernel_lambda_derivatives_apply_two_segment_chain_rule(toy_kernel):
    result = toy_kernel.evaluate(_valid_system(), 0.75)

    assert result.denergy_dlambda_e_ev == 4.0
    assert result.denergy_dlambda_v_ev == 10.5
    assert result.denergy_dprogress_ev == 8.0


def test_kernel_force_and_cell_gradient_match_finite_difference(toy_kernel):
    system = validate_fennix_alchemical_system(_valid_system())
    result = toy_kernel.evaluate(system, 0.75)
    h = 1e-6
    coordinates = np.asarray(system.coordinates_angstrom)
    plus_coordinates = coordinates.copy()
    minus_coordinates = coordinates.copy()
    plus_coordinates[0, 0] += h
    minus_coordinates[0, 0] -= h
    fd_coordinate = (
        toy_kernel.evaluate(
            replace(system, coordinates_angstrom=plus_coordinates), 0.75
        ).energy_ev
        - toy_kernel.evaluate(
            replace(system, coordinates_angstrom=minus_coordinates), 0.75
        ).energy_ev
    ) / (2.0 * h)
    cell = np.asarray(system.cell_angstrom)
    plus_cell = cell.copy()
    minus_cell = cell.copy()
    plus_cell[0, 0] += h
    minus_cell[0, 0] -= h
    fd_cell = (
        toy_kernel.evaluate(replace(system, cell_angstrom=plus_cell), 0.75).energy_ev
        - toy_kernel.evaluate(replace(system, cell_angstrom=minus_cell), 0.75).energy_ev
    ) / (2.0 * h)

    assert -result.forces_ev_per_angstrom[0][0] == pytest.approx(
        fd_coordinate, abs=1e-7
    )
    assert result.cell_gradient_ev_per_angstrom[0][0] == pytest.approx(
        fd_cell, abs=1e-7
    )


def test_kernel_progress_derivative_matches_finite_difference(toy_kernel):
    system = _valid_system()
    progress = 0.75
    h = 1e-6
    result = toy_kernel.evaluate(system, progress)
    finite_difference = (
        toy_kernel.evaluate(system, progress + h).energy_ev
        - toy_kernel.evaluate(system, progress - h).energy_ev
    ) / (2.0 * h)

    assert result.denergy_dprogress_ev == pytest.approx(finite_difference, abs=3e-8)


def test_kernel_individual_lambda_derivatives_match_finite_difference(toy_kernel):
    system = _valid_system()
    result = toy_kernel.evaluate(system, 0.75)
    coordinates = np.asarray(system.coordinates_angstrom)
    cell = np.asarray(system.cell_angstrom)[None, :, :]
    base = 0.5 * np.sum(coordinates**2) + 0.25 * np.sum(cell**2)
    lambda_e = result.lambda_state.lambda_e
    lambda_v = result.lambda_state.lambda_v
    h = 1e-6

    def energy(le, lv):
        return base + 3.0 * le + 5.0 * lv**2 + le * lv

    fd_e = (energy(lambda_e + h, lambda_v) - energy(lambda_e - h, lambda_v)) / (2.0 * h)
    fd_v = (energy(lambda_e, lambda_v + h) - energy(lambda_e, lambda_v - h)) / (2.0 * h)

    assert result.denergy_dlambda_e_ev == pytest.approx(fd_e, abs=5e-7)
    assert result.denergy_dlambda_v_ev == pytest.approx(fd_v, abs=5e-7)


def test_kernel_virial_matches_right_strain_finite_difference(toy_kernel):
    system = validate_fennix_alchemical_system(_valid_system())
    result = toy_kernel.evaluate(system, 0.75)
    coordinates = np.asarray(system.coordinates_angstrom)
    cell = np.asarray(system.cell_angstrom)
    h = 1e-6
    generator = np.zeros((3, 3), dtype=np.float64)
    generator[0, 1] = 1.0

    def strain(sign):
        scaling = np.eye(3) + sign * h * generator
        return replace(
            system,
            coordinates_angstrom=coordinates @ scaling,
            cell_angstrom=cell @ scaling,
        )

    finite_difference = (
        toy_kernel.evaluate(strain(+1.0), 0.75).energy_ev
        - toy_kernel.evaluate(strain(-1.0), 0.75).energy_ev
    ) / (2.0 * h)

    assert result.virial_ev[0][1] == pytest.approx(finite_difference, abs=5e-7)


def test_close_contact_decoupled_endpoint_mechanics_remain_finite(toy_kernel):
    system = _valid_system()
    coordinates = np.asarray(system.coordinates_angstrom, dtype=np.float64).copy()
    coordinates[0] = coordinates[1]

    result = toy_kernel.evaluate(
        replace(system, coordinates_angstrom=coordinates),
        0.0,
    )

    assert np.isfinite(result.energy_ev)
    assert np.isfinite(result.forces_ev_per_angstrom).all()
    assert np.isfinite(result.virial_ev).all()
    assert np.isfinite(result.denergy_dprogress_ev)


def test_half_progress_has_distinct_finite_one_sided_derivatives(toy_kernel):
    system = _valid_system()
    join = 0.5
    h = 1e-7
    at_join = toy_kernel.evaluate(system, join)
    left = (at_join.energy_ev - toy_kernel.evaluate(system, join - h).energy_ev) / h
    right = (toy_kernel.evaluate(system, join + h).energy_ev - at_join.energy_ev) / h

    assert at_join.denergy_dprogress_ev == pytest.approx(left, abs=3e-6)
    assert left == pytest.approx(20.0, abs=3e-6)
    assert right == pytest.approx(8.0, abs=3e-6)


def test_kernel_endpoints_and_shared_half_state_close_exactly(toy_kernel):
    system = _valid_system()
    zero = toy_kernel.evaluate(system, 0.0)
    half = toy_kernel.evaluate(system, 0.5)
    full = toy_kernel.evaluate(system, 1.0)
    coordinates = np.asarray(system.coordinates_angstrom)
    cell = np.asarray(system.cell_angstrom)[None, :, :]
    base = 0.5 * np.sum(coordinates**2) + 0.25 * np.sum(cell**2)

    solute = np.asarray(system.coordinates_angstrom)[[0]]
    solvent = np.asarray(system.coordinates_angstrom)[1:]
    decoupled_sum = (
        0.5 * np.sum(solute**2) + 0.5 * np.sum(solvent**2) + 0.25 * np.sum(cell**2)
    )
    assert zero.energy_ev == decoupled_sum == base
    assert half.energy_ev == base + 5.0
    assert full.energy_ev == base + 9.0


def test_mechanics_result_never_unlocks_hfe_accuracy_gpu_or_performance(toy_kernel):
    result = toy_kernel.evaluate(_valid_system(), 0.75)

    assert isinstance(result, FeNNixKernelResult)
    assert result.identity.gpu_production_admitted is False
    assert result.identity.scientific_scope == FENNIX_KERNEL_SCIENTIFIC_SCOPE
    assert result.scope == FENNIX_KERNEL_SCIENTIFIC_SCOPE
    assert not hasattr(result, "hydration_free_energy")
    assert not hasattr(result, "experimental_error")
    assert not hasattr(result, "speedup")
