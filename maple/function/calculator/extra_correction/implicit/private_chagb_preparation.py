"""Transactional frozen-source charge preparation for private Torch CHA SP."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from .charges import ChargeResult, _molecular_charge_and_mult
from .private_torch_chagb import PrivateTorchChaGB, _value


def _array_sha256(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def prepare_private_chagb_frozen_source(
    atoms,
    *,
    prepared_topology: str | Path,
    prepared_topology_sha256: str,
    device: Any = None,
    dtype: str | None = "float32",
) -> tuple[PrivateTorchChaGB, ChargeResult]:
    """Validate once, then select the topology's frozen source-charge vector.

    The charged MOL2 remains the bound atom/type/file identity source. The
    caller's incoming initial-charge array is recorded only as metadata and is
    never selected, normalized, or used to reconstruct the prepared topology.
    """

    _molecular_charge_and_mult(atoms)
    provider = object.__new__(PrivateTorchChaGB)
    provider._initialize_loaded(
        prepared_topology=prepared_topology,
        prepared_topology_sha256=prepared_topology_sha256,
        device=device,
        dtype=dtype,
    )
    source_charges = np.asarray(
        _value(provider._topology, "source_charges_e"), dtype=np.float64
    )
    if source_charges.shape != (len(atoms),) or not np.all(np.isfinite(source_charges)):
        raise ValueError("Prepared topology source-charge vector is invalid.")
    incoming_charges = np.asarray(atoms.get_initial_charges(), dtype=np.float64)
    if incoming_charges.shape != source_charges.shape or not np.all(
        np.isfinite(incoming_charges)
    ):
        raise ValueError("Incoming caller initial-charge vector is invalid.")
    charge_result = ChargeResult(
        charges=source_charges.copy(),
        method="prepared-topology-frozen-source-charges",
        mode="fixed",
        provenance={
            "source": "prepared-topology-frozen-source-charges",
            "generation_performed": False,
            "normalization_performed": False,
            "selected_charge_role": "serialized-source-charges",
            "prepared_topology_path": str(provider._topology_path),
            "prepared_topology_file_sha256": provider._topology_file_sha256,
            "prepared_topology_content_sha256": (
                provider._source_topology_content_sha256
            ),
            "source_charge_manifest_sha256": str(
                _value(provider._topology, "source_charge_manifest_sha256")
            ),
            "source_charges_sha256": str(
                _value(provider._topology, "source_charges_sha256")
            ),
            "source_charges_e_binary64_sha256": _array_sha256(source_charges),
            "original_mol2_path": str(atoms.info["mol2"]["path"]),
            "original_mol2_file_sha256": str(
                _value(provider._topology, "source_mol2_sha256")
            ),
            "incoming_caller_initial_charges_role": "metadata-only-not-selected",
            "incoming_caller_initial_charges_binary64_sha256": _array_sha256(
                incoming_charges
            ),
            "effective_prmtop_sha256": str(
                _value(provider._topology, "prepared_prmtop_sha256")
            ),
            "effective_charges_sha256": str(
                _value(provider._topology, "effective_charges_sha256")
            ),
            "effective_charges_fp32_tensor_sha256": dict(
                _value(provider._topology, "compute_tensor_sha256")
            )["effective_charges_e"],
        },
    )
    provider._bind_source_identity(atoms, charge_result)
    return provider, charge_result


__all__ = ["prepare_private_chagb_frozen_source"]
