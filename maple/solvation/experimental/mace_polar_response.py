"""Experimental v1 response composition over the identity-neutral core.

The provider/scalar/profile and exact legacy component policy are unchanged.
Moving the mechanics into the shared core changes the implementation/source
manifest epoch only; it does not relabel this composition as native-FP64 v2.
"""

from __future__ import annotations

from pathlib import Path

from .mace_polar_response_core import (
    ResponseCompositionPolicy,
    ResponsePESCore,
    _BUILDER_TOKEN,
    _SECOND_ORDER_CONTINUUM_LIMIT_BYTES,
    _response_resource_preflight,
)

_PROVIDER_ID = "maple.pure-macepolar-frozen-ddpcm-response-legacy-cds.experimental-v1"
_V1_POLICY_ID = "macepolar-response-legacy-components-policy-v1-core-epoch2"


def _v1_ids(kind: str) -> dict[str, str]:
    return {
        "provider_id": _PROVIDER_ID,
        "profile_id": (
            "pure-macepolar-frozen-point-l1-ddpcm-response-smd-"
            f"{kind}-experimental-v1"
        ),
        "scalar_contract_id": (
            "route2-experimental-pure-macepolar-frozen-point-l1-ddpcm-response-"
            f"smd-{kind}-v1"
        ),
    }


def _validate_v1_components(model, continuum, solvent_term) -> None:
    from maple.solvation.continuum.ddpcm_response import TorchDDPCMResponse
    from maple.solvation.models.mace_polar_torch import MACEPolarTorchGraphAdapter
    from maple.solvation.nonpolar.legacy_smd_cds import TorchLegacySMDCDS

    if not (
        isinstance(model, MACEPolarTorchGraphAdapter)
        and isinstance(continuum, TorchDDPCMResponse)
        and isinstance(solvent_term, TorchLegacySMDCDS)
    ):
        raise TypeError("Response PES requires the exact frozen Torch components.")


def _v1_source_files() -> dict[str, Path]:
    solvation = Path(__file__).resolve().parents[1]
    return {
        "response_v1_composition": Path(__file__),
        "model_graph_adapter": solvation / "models" / "mace_polar_torch.py",
        "legacy_smd_cds": solvation / "nonpolar" / "legacy_smd_cds.py",
    }


_V1_POLICY = ResponseCompositionPolicy(
    policy_id=_V1_POLICY_ID,
    provider_id=_PROVIDER_ID,
    ids_for_device=_v1_ids,
    validate_components=_validate_v1_components,
    source_files=_v1_source_files,
)


class MACEPolarResponsePES(ResponsePESCore):
    """Legacy-component v1 identity backed by shared response mechanics."""

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
            _policy=_V1_POLICY,
            _testing_token=_testing_token,
            _builder_token=_builder_token,
        )


def build_smd_mace_polar_response_pes(
    symbols, *, solvent: str, device: str = "cpu", checkpoint_path=None
) -> MACEPolarResponsePES:
    """Build the fixed v1 response identity without scientific overrides."""
    from maple.function.calculator.extra_correction.implicit.smd_cds import (
        smd_coulomb_radii,
    )
    from maple.function.route2_solvents import route2_solvent_spec
    from maple.solvation.continuum.ddpcm_response import TorchDDPCMResponse
    from maple.solvation.models.mace_polar import build_official_mace_polar_1_m_adapter
    from maple.solvation.models.mace_polar_torch import MACEPolarTorchGraphAdapter
    from maple.solvation.nonpolar.legacy_smd_cds import TorchLegacySMDCDS

    specification = route2_solvent_spec(solvent)
    if specification.name not in {"water", "hexane"}:
        raise ValueError("Response PES initially supports water and hexane only.")
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
    cds = TorchLegacySMDCDS(symbols, specification.name, device=normalized_device)
    base = build_official_mace_polar_1_m_adapter(
        device=normalized_device, checkpoint_path=checkpoint_path, torch_graph=True
    )
    return MACEPolarResponsePES(
        model=MACEPolarTorchGraphAdapter(base),
        continuum=continuum,
        solvent_term=cds,
        symbols=symbols,
        device=normalized_device,
        solvent=specification.name,
        _builder_token=_BUILDER_TOKEN,
    )


__all__ = ["MACEPolarResponsePES", "build_smd_mace_polar_response_pes"]
