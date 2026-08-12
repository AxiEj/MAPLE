"""Autograd coordinate VJP for fixed native MACE-POLAR field features."""

from __future__ import annotations

import numpy as np


def density_position_vjp_features(
    calculator: object,
    atoms: object,
    *,
    model_field_features: object,
    density_cotangent: object,
) -> np.ndarray:
    """Return ``d <cotangent,density> / dR`` with features held fixed.

    This narrow adapter uses the calculator's existing batch/model/projector
    primitives. Keeping it outside the legacy calculator preserves the bytes
    of the historical P0 response-only certificate while binding this new
    vNext derivative implementation through the model-adapter provenance hash.
    """

    import importlib

    torch = importlib.import_module("torch")

    features = np.asarray(model_field_features, dtype=float)
    cotangent = np.asarray(density_cotangent, dtype=float)
    atom_count = len(atoms)  # type: ignore[arg-type]
    expected_density_shape = (atom_count, 4)
    if (
        features.ndim != 2
        or features.shape[0] != atom_count
        or features.shape[1] == 0
        or not np.all(np.isfinite(features))
    ):
        raise ValueError(
            "Model-field features must be finite with shape " "(n_atoms, n_features)."
        )
    if cotangent.shape != expected_density_shape or not np.all(np.isfinite(cotangent)):
        raise ValueError(
            "Density cotangent must be finite with shape "
            f"{expected_density_shape}; received {cotangent.shape}."
        )
    if bool(getattr(calculator, "_route2_jgp94_d2_canonical_mace", False)):
        raise NotImplementedError(
            "JGP94 D2 canonicalisation has no native-feature coordinate proof."
        )

    dtype = getattr(calculator, "dtype")
    device = getattr(calculator, "device")
    feature_tensor = torch.as_tensor(features, dtype=dtype, device=device)
    batch = calculator._batch_dict(atoms)
    positions = batch.get("positions")
    if (
        not torch.is_tensor(positions)
        or positions.shape != (atom_count, 3)
        or not torch.is_floating_point(positions)
    ):
        received = None if positions is None else tuple(positions.shape)
        raise RuntimeError(
            "MACE-POLAR coordinate VJP requires floating-point batch positions "
            f"with shape ({atom_count}, 3); received {received}."
        )
    positions_required_grad = positions.requires_grad
    positions.requires_grad_(True)
    try:
        projector = getattr(calculator, "_reaction_projector")
        with projector.use_model_field_features(feature_tensor):
            output = calculator._model_forward(
                batch,
                compute_force=False,
                compute_stress=False,
                compute_hessian=False,
            )
        density = output.get("density_coefficients")
        if density is None or density.shape != expected_density_shape:
            received = None if density is None else tuple(density.shape)
            raise RuntimeError(
                "MACE-POLAR feature-driven density coordinate response must "
                f"have shape {expected_density_shape}; received {received}."
            )
        if not bool(torch.isfinite(density).all()) or not density.requires_grad:
            raise RuntimeError(
                "MACE-POLAR feature-driven density is non-finite or disconnected "
                "from the coordinate autograd graph."
            )
        cotangent_tensor = torch.as_tensor(cotangent, dtype=dtype, device=device)
        (position_gradient,) = torch.autograd.grad(
            density,
            (positions,),
            grad_outputs=cotangent_tensor,
            create_graph=False,
            allow_unused=True,
        )
        if position_gradient is None:
            raise RuntimeError(
                "MACE-POLAR feature-driven density is disconnected from the "
                "coordinate autograd graph."
            )
        result = np.asarray(position_gradient.detach().cpu(), dtype=float).copy()
        if result.shape != (atom_count, 3) or not np.all(np.isfinite(result)):
            raise RuntimeError(
                "MACE-POLAR feature-driven density position VJP must be finite "
                f"with shape {(atom_count, 3)}; received {result.shape}."
            )
        return calculator._long_range_evaluator.coordinate_vjp(result)
    finally:
        if not positions_required_grad:
            positions.requires_grad_(False)


__all__ = ["density_position_vjp_features"]
