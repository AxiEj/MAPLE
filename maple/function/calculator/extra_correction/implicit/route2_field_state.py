"""Immutable field state passed from a continuum operator to a polar MLIP."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReactionFieldDrive:
    """Separate model-driving features from the density-dual energy field.

    ``density_dual_field_ev`` always has the Route-2 Cartesian
    ``[V,gx,gy,gz]`` shape used by the polarization-energy pairing.
    A local-jet model drive may use either that same field or a separately
    gauge-fixed ``model_local_field_ev``.  An exact GTO path supplies the
    complete checkpoint-native feature tensor explicitly instead.
    """

    density_dual_field_ev: np.ndarray
    model_local_field_ev: np.ndarray | None
    model_field_features: np.ndarray | None
    projector: str
    model_field_gauge: str
    model_field_gauge_reference_ev: float

    def __post_init__(self) -> None:
        field = np.asarray(self.density_dual_field_ev, dtype=float)
        if (
            field.ndim != 2
            or field.shape[1] != 4
            or not np.all(np.isfinite(field))
        ):
            raise ValueError(
                "Density-dual reaction field must be finite with shape "
                "(n_atoms, 4)."
            )
        immutable_field = np.array(field, copy=True)
        immutable_field.setflags(write=False)
        object.__setattr__(
            self,
            "density_dual_field_ev",
            immutable_field,
        )

        local_field = self.model_local_field_ev
        if local_field is not None:
            local_array = np.asarray(local_field, dtype=float)
            if (
                local_array.shape != field.shape
                or not np.all(np.isfinite(local_array))
            ):
                raise ValueError(
                    "The model-driving local field must be finite with the "
                    "same shape as the density-dual field."
                )
            immutable_local = np.array(local_array, copy=True)
            immutable_local.setflags(write=False)
            object.__setattr__(
                self,
                "model_local_field_ev",
                immutable_local,
            )

        features = self.model_field_features
        if features is not None:
            if local_field is not None:
                raise ValueError(
                    "Supply either a model-driving local field or preprojected "
                    "model features, not both."
                )
            feature_array = np.asarray(features, dtype=float)
            if (
                feature_array.ndim != 2
                or feature_array.shape[0] != field.shape[0]
                or feature_array.shape[1] == 0
                or not np.all(np.isfinite(feature_array))
            ):
                raise ValueError(
                    "Model reaction-field features must be finite with shape "
                    "(n_atoms, n_features)."
                )
            immutable_features = np.array(feature_array, copy=True)
            immutable_features.setflags(write=False)
            object.__setattr__(
                self,
                "model_field_features",
                immutable_features,
            )
        projector = str(self.projector).strip()
        if not projector:
            raise ValueError("A reaction-field projector identity is required.")
        model_field_gauge = str(self.model_field_gauge).strip()
        if not model_field_gauge:
            raise ValueError("A model-field gauge identity is required.")
        gauge_reference = float(self.model_field_gauge_reference_ev)
        if not np.isfinite(gauge_reference):
            raise ValueError("The model-field gauge reference must be finite.")
        if (
            model_field_gauge == "continuum-zero-at-infinity"
            and gauge_reference != 0.0
        ):
            raise ValueError(
                "The continuum-zero-at-infinity model gauge requires a zero "
                "reference."
            )
        if model_field_gauge != "continuum-zero-at-infinity" and (
            local_field is None and features is None
        ):
            raise ValueError(
                "A non-default model-field gauge requires a distinct local "
                "field or preprojected feature tensor."
            )
        if projector == "local-jet" and features is not None:
            raise ValueError(
                "The local-jet projector cannot carry preprojected features."
            )
        if projector == "exact-gto-v1" and features is None:
            raise ValueError(
                "The exact-GTO projector requires preprojected features."
            )
        object.__setattr__(
            self,
            "model_field_gauge_reference_ev",
            gauge_reference,
        )

    @classmethod
    def local_jet(
        cls,
        field_values_ev: np.ndarray,
        *,
        model_field_gauge: str = "continuum-zero-at-infinity",
        model_field_gauge_reference_ev: float = 0.0,
    ) -> "ReactionFieldDrive":
        """Wrap a local-jet drive while preserving the energy-dual field."""

        field = np.asarray(field_values_ev, dtype=float)
        if (
            field.ndim != 2
            or field.shape[1] != 4
            or not np.all(np.isfinite(field))
        ):
            raise ValueError(
                "Density-dual reaction field must be finite with shape "
                "(n_atoms, 4)."
            )
        gauge_reference = float(model_field_gauge_reference_ev)
        model_field = None
        if model_field_gauge != "continuum-zero-at-infinity":
            model_field = np.array(field, copy=True)
            model_field[:, 0] -= gauge_reference
        elif gauge_reference != 0.0:
            raise ValueError(
                "The continuum-zero-at-infinity model gauge requires a zero "
                "reference."
            )

        return cls(
            density_dual_field_ev=field,
            model_local_field_ev=model_field,
            model_field_features=None,
            projector="local-jet",
            model_field_gauge=model_field_gauge,
            model_field_gauge_reference_ev=gauge_reference,
        )


__all__ = ["ReactionFieldDrive"]
