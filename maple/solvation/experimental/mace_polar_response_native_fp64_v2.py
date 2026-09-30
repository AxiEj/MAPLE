"""Programmatic-only native-FP64 response v2 composition.

This identity combines the strict native-FP64 MACE boundary, unchanged
structured ddPCM response equations, and native-literal SMD-CDS.  It is not
registered for release and advertises no scientific-admission capability.
"""

from __future__ import annotations

from pathlib import Path

from .mace_polar_response_core import (
    ResponseCompositionPolicy,
    ResponsePESCore,
    _BUILDER_TOKEN,
    _SECOND_ORDER_CONTINUUM_LIMIT_BYTES,
)

NATIVE_FP64_RESPONSE_V2_PROVIDER_ID = (
    "maple.pure-macepolar-native-fp64-ddpcm-response-native-cds.experimental-v2"
)
_NATIVE_FP64_RESPONSE_V2_POLICY_ID = (
    "macepolar-native-fp64-response-native-cds-composition-policy-v2"
)


def native_fp64_response_v2_ids(kind: str) -> dict[str, str]:
    if kind not in {"cpu", "cuda"}:
        raise ValueError("Native-FP64 response identity requires cpu or cuda.")
    normalized = kind
    return {
        "provider_id": NATIVE_FP64_RESPONSE_V2_PROVIDER_ID,
        "profile_id": (
            "pure-macepolar-native-fp64-point-l1-ddpcm-response-native-smd-"
            f"{normalized}-experimental-v2"
        ),
        "scalar_contract_id": (
            "route2-experimental-pure-macepolar-native-fp64-point-l1-ddpcm-"
            f"response-native-smd-{normalized}-v2"
        ),
    }


def _validate_v2_components(model, continuum, solvent_term) -> None:
    from maple.solvation.continuum.ddpcm_response import TorchDDPCMResponse
    from maple.solvation.models.mace_polar_native_fp64 import (
        MACEPolarNativeFP64Adapter,
    )
    from maple.solvation.nonpolar.native_smd_cds import TorchNativeSMDCDS

    if not (
        type(model) is MACEPolarNativeFP64Adapter
        and model.is_production_adapter
        and type(continuum) is TorchDDPCMResponse
        and type(solvent_term) is TorchNativeSMDCDS
    ):
        raise TypeError(
            "Native-FP64 response v2 requires its exact model, structured ddPCM, "
            "and native-CDS components."
        )


def _v2_source_files() -> dict[str, Path]:
    solvation = Path(__file__).resolve().parents[1]
    return {
        "response_native_fp64_v2_composition": Path(__file__),
        "native_fp64_model": solvation / "models" / "mace_polar_native_fp64.py",
        "model_graph_adapter": solvation / "models" / "mace_polar_torch.py",
        "native_smd_cds": solvation / "nonpolar" / "native_smd_cds.py",
        "native_smd_cds_parameters": (
            solvation / "nonpolar" / "native_smd_cds_parameters.py"
        ),
    }


_NATIVE_FP64_RESPONSE_V2_POLICY = ResponseCompositionPolicy(
    policy_id=_NATIVE_FP64_RESPONSE_V2_POLICY_ID,
    provider_id=NATIVE_FP64_RESPONSE_V2_PROVIDER_ID,
    ids_for_device=native_fp64_response_v2_ids,
    validate_components=_validate_v2_components,
    source_files=_v2_source_files,
)


class MACEPolarResponseNativeFP64V2PES(ResponsePESCore):
    """Separately identified native-FP64 v2 total PES."""

    admitted_capabilities: tuple[str, ...] = ()
    scientific_release_admitted = False

    def __init__(
        self,
        model,
        continuum,
        solvent_term,
        symbols,
        device="cpu",
        solvent="water",
        _testing_token=None,
        _builder_token=None,
    ) -> None:
        super().__init__(
            model=model,
            continuum=continuum,
            solvent_term=solvent_term,
            symbols=symbols,
            device=device,
            solvent=solvent,
            _policy=_NATIVE_FP64_RESPONSE_V2_POLICY,
            _testing_token=_testing_token,
            _builder_token=_builder_token,
        )


def build_smd_mace_polar_response_native_fp64_v2_pes(
    symbols,
    solvent,
    device="cpu",
    checkpoint_path=None,
) -> MACEPolarResponseNativeFP64V2PES:
    """Build the closed native-FP64 v2 response composition."""
    from maple.function.calculator.extra_correction.implicit.smd_cds import (
        smd_coulomb_radii,
    )
    from maple.function.route2_solvents import route2_solvent_spec
    from maple.solvation.continuum.ddpcm_response import TorchDDPCMResponse
    from maple.solvation.models.mace_polar_native_fp64 import (
        build_official_mace_polar_native_fp64_adapter,
    )
    from maple.solvation.nonpolar.native_smd_cds import TorchNativeSMDCDS

    specification = route2_solvent_spec(solvent)
    if specification.name not in {"water", "ethanol", "hexane"}:
        raise ValueError(
            "Native-FP64 response v2 initially supports water, ethanol, and "
            "hexane only."
        )
    normalized_device = "cuda:0" if device == "cuda" else str(device)
    if normalized_device.startswith("cuda:"):
        from .cuda_execution import prepare_cuda_execution

        prepare_cuda_execution()
    symbols = tuple(symbols)
    continuum = TorchDDPCMResponse(
        symbols,
        smd_coulomb_radii(symbols, solvent=specification.name),
        dielectric=specification.descriptors.dielectric,
        lmax=15,
        n_lebedev=1202,
        eta=0.1,
        device=normalized_device,
        max_dense_bytes=_SECOND_ORDER_CONTINUUM_LIMIT_BYTES,
    )
    continuum.preflight_resources(
        derivative_order=2,
        atom_count=len(symbols),
        limit_bytes=_SECOND_ORDER_CONTINUUM_LIMIT_BYTES,
    )
    cds = TorchNativeSMDCDS(symbols, specification.name, device=normalized_device)
    model = build_official_mace_polar_native_fp64_adapter(
        normalized_device, checkpoint_path=checkpoint_path
    )
    return MACEPolarResponseNativeFP64V2PES(
        model=model,
        continuum=continuum,
        solvent_term=cds,
        symbols=symbols,
        device=normalized_device,
        solvent=specification.name,
        _builder_token=_BUILDER_TOKEN,
    )


__all__ = [
    "MACEPolarResponseNativeFP64V2PES",
    "NATIVE_FP64_RESPONSE_V2_PROVIDER_ID",
    "build_smd_mace_polar_response_native_fp64_v2_pes",
    "native_fp64_response_v2_ids",
]
