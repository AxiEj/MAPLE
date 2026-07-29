"""Torch-only conservative-energy reconstruction of public LSNN-v1.

The public LSNN-v1 repository is MIT licensed, but its executable entrypoint is
incomplete at revision 1768d068dcb1ea65e8585af3f4a0cbf4047d9125.  This module
implements the published/state-dict energy graph without PyTorch-Geometric so
a pinned compatibility probe can be executed through OpenMM-Torch. Unlike the
public runtime, which detaches its electrostatic energy before returning
explicit forces, this module exposes a scalar full energy and lets OpenMM-Torch
obtain conservative forces by differentiation.

This is benchmark code, not a MAPLE production backend. The upstream
copyright and MIT permission notice are retained in ``LICENSE.LSNN-v1``.
"""

# PyTorch's register_buffer attributes are dynamically typed in its stubs.
# pyright: reportArgumentType=false, reportCallIssue=false, reportIndexIssue=false, reportOperatorIssue=false

from __future__ import annotations

import torch
from torch import nn


class _GBNeckInteraction(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("_d0", torch.zeros(100))
        self.register_buffer("_m0", torch.zeros(100))
        self.num_unique = 10

    def forward(
        self,
        features: torch.Tensor,
        source: torch.Tensor,
        target: torch.Tensor,
        distances: torch.Tensor,
    ) -> torch.Tensor:
        target_features = features[target]
        source_features = features[source]
        radius_i = target_features[:, 1]
        scaled_radius_j = source_features[:, 2]
        radius_index_i = target_features[:, 6].to(torch.long)
        radius_index_j = source_features[:, 6].to(torch.long)

        offset = 0.0195141
        neck_cut = 0.68
        neck_scale = 0.826836
        radius_with_offset_i = radius_i + offset
        radius_with_offset_j = source_features[:, 1] + offset

        distance_difference = torch.abs(distances - scaled_radius_j)
        lower = torch.maximum(radius_i, distance_difference)
        upper = distances + scaled_radius_j
        volume_integral = torch.where(
            (distances + scaled_radius_j - radius_i) > 0,
            0.5
            * (
                1 / lower
                - 1 / upper
                + 0.25
                * (distances - scaled_radius_j.square() / distances)
                * (1 / upper.square() - 1 / lower.square())
                + 0.5 * torch.log(lower / upper) / distances
            ),
            torch.zeros_like(distances),
        )

        table_index = self.num_unique * radius_index_j + radius_index_i
        d0 = self._d0[table_index]
        m0 = self._m0[table_index]
        neck_integral = torch.where(
            (radius_with_offset_i + radius_with_offset_j + neck_cut - distances) > 0,
            m0
            / (
                1
                + 100 * (distances - d0).square()
                + 0.3 * 1_000_000 * (distances - d0).pow(6)
            ),
            torch.zeros_like(distances),
        )

        messages = volume_integral + neck_scale * neck_integral
        aggregate = torch.zeros(
            features.shape[0], dtype=features.dtype, device=features.device
        )
        aggregate.index_add_(0, target, messages)

        alpha = features[:, 3]
        beta = features[:, 4]
        gamma = features[:, 5]
        psi = aggregate * features[:, 1]
        born_radius = 1 / (
            1 / features[:, 1]
            - torch.tanh(alpha * psi - beta * psi.square() + gamma * psi.pow(3))
            / (features[:, 1] + offset)
        )
        return torch.stack((born_radius, features[:, 0]), dim=1)


class _GBNeckEnergies(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # These buffers are present in the upstream state dict.  The energy
        # expression itself does not index them.
        self.register_buffer("_d0", torch.zeros(100))
        self.register_buffer("_m0", torch.zeros(100))
        self.register_buffer("_soluteDielectric", torch.tensor(1.0))
        self.register_buffer("_solventDielectric", torch.tensor(78.5))

    def forward(
        self,
        born_and_charge: torch.Tensor,
        source: torch.Tensor,
        target: torch.Tensor,
        distances: torch.Tensor,
    ) -> torch.Tensor:
        target_features = born_and_charge[target]
        source_features = born_and_charge[source]
        born_product = target_features[:, 0] * source_features[:, 0]
        screening_distance = torch.sqrt(
            distances.square()
            + born_product * torch.exp(-distances.square() / (4 * born_product))
        )
        messages = target_features[:, 1] * source_features[:, 1] / screening_distance
        aggregate = torch.zeros(
            born_and_charge.shape[0],
            dtype=born_and_charge.dtype,
            device=born_and_charge.device,
        )
        aggregate.index_add_(0, target, messages)
        self_energy = born_and_charge[:, 1].square() / born_and_charge[:, 0]
        prefactor = (
            -0.5
            * 138.935485
            * (1 / self._soluteDielectric - 1 / self._solventDielectric)
        )
        return (prefactor * (aggregate + self_energy)).unsqueeze(1)


class _InteractionLayer(nn.Module):
    def __init__(
        self, input_channels: int, output_channels: int, hidden: int = 96
    ) -> None:
        super().__init__()
        self.cutoff = 0.4
        self.register_buffer("_FREQUENCIES", torch.zeros(20))
        self.message1 = nn.Linear(input_channels + 20, hidden)
        self.message2 = nn.Linear(hidden, hidden)
        self.lin1 = nn.Linear(hidden, hidden)
        self.lin2 = nn.Linear(hidden, output_channels)
        self.silu = nn.SiLU()

    def forward(
        self,
        features: torch.Tensor,
        source: torch.Tensor,
        target: torch.Tensor,
        distances: torch.Tensor,
    ) -> torch.Tensor:
        scaled_distance = distances.unsqueeze(1) / self.cutoff
        p = 6
        envelope = (
            1 / scaled_distance
            - ((p + 1) * (p + 2) / 2) * scaled_distance.pow(p - 1)
            + (p * (p + 2)) * scaled_distance.pow(p)
            - (p * (p + 1) / 2) * scaled_distance.pow(p + 1)
        )
        radial_features = envelope * torch.sin(self._FREQUENCIES * scaled_distance)
        messages = torch.cat(
            (features[target], features[source], radial_features), dim=1
        )
        messages = self.silu(self.message1(messages))
        messages = self.silu(self.message2(messages))
        aggregate = torch.zeros(
            (features.shape[0], messages.shape[1]),
            dtype=features.dtype,
            device=features.device,
        )
        aggregate.index_add_(0, target, messages)
        return self.lin2(self.silu(self.lin1(aggregate)))


class LSNNV1Energy(nn.Module):
    """Public LSNN-v1 energy graph with molecule-specific GBn2 features."""

    def __init__(self, gbn2_features: torch.Tensor) -> None:
        super().__init__()
        self.aggregate_information = _GBNeckInteraction()
        self.calculate_energies = _GBNeckEnergies()
        self.lin = nn.Linear(1, 1)  # Retained for exact state-dict compatibility.
        self.interaction1 = _InteractionLayer(10, 96)
        self.interaction2 = _InteractionLayer(192, 2)
        self.sterics_ff = nn.Sequential(
            nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, 1), nn.Sigmoid()
        )
        self.electrostatics_ff = nn.Sequential(
            nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, 1), nn.Sigmoid()
        )
        self.register_buffer("gnn_params", gbn2_features.to(torch.float32))
        self.register_buffer("offset", torch.tensor(0.0195141))
        self.register_buffer("gamma", torch.tensor(0.00542))

    @staticmethod
    def _edges(
        positions: torch.Tensor, cutoff: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distance_matrix = torch.cdist(positions, positions)
        pairs = torch.nonzero((distance_matrix < cutoff) & (distance_matrix > 0))
        source = pairs[:, 0]
        target = pairs[:, 1]
        return source, target, distance_matrix[source, target]

    def forward(
        self,
        positions: torch.Tensor,
        lambda_sterics: torch.Tensor,
        lambda_electrostatics: torch.Tensor,
    ) -> torch.Tensor:
        positions_float = positions.to(torch.float32)
        lambda_s = lambda_sterics.to(torch.float32).reshape(1, 1)
        lambda_e = lambda_electrostatics.to(torch.float32).reshape(1, 1)

        source, target, distances = self._edges(positions_float, 13.0)
        gnn_source, gnn_target, gnn_distances = self._edges(positions_float, 0.4)
        sterics_scale = self.sterics_ff(lambda_s) * lambda_s
        electrostatics_scale = self.electrostatics_ff(lambda_e) * lambda_e

        features = torch.cat(
            (
                self.gnn_params,
                lambda_e.expand(positions_float.shape[0], 1),
                lambda_s.expand(positions_float.shape[0], 1),
            ),
            dim=1,
        )
        born_and_charge = self.aggregate_information(
            features, source, target, distances
        )
        interaction_features = torch.cat(
            (
                born_and_charge,
                features[:, 1:2],
                sterics_scale.expand(positions_float.shape[0], 1),
                electrostatics_scale.expand(positions_float.shape[0], 1),
            ),
            dim=1,
        )
        interaction_features = torch.nn.functional.silu(
            self.interaction1(
                interaction_features, gnn_source, gnn_target, gnn_distances
            )
        )
        corrections = self.interaction2(
            interaction_features, gnn_source, gnn_target, gnn_distances
        )
        born_scale = corrections[:, 0]
        surface_scale = corrections[:, 1]

        radius = (features[:, 1] + self.offset).unsqueeze(1)
        surface_energy = (
            4.184
            * self.gamma
            * surface_scale.unsqueeze(1)
            * (radius + 0.14).square()
            * 100
        )
        corrected_born = born_and_charge[:, 0:1] * (
            0.5 + torch.sigmoid(born_scale.unsqueeze(1))
        )
        electrostatic_energy = self.calculate_energies(
            torch.cat((corrected_born, born_and_charge[:, 1:2]), dim=1),
            source,
            target,
            distances,
        )
        return (
            electrostatic_energy * electrostatics_scale + surface_energy * sterics_scale
        ).sum()


def load_lsnn_v1(state_dict_path: str, gbn2_features: torch.Tensor) -> LSNNV1Energy:
    model = LSNNV1Energy(gbn2_features)
    state_dict = torch.load(state_dict_path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    allowed_missing = {"gnn_params", "offset", "gamma"}
    if set(missing).difference(allowed_missing) or unexpected:
        raise RuntimeError(
            f"LSNN state-dict mismatch: missing={missing}, unexpected={unexpected}"
        )
    return model.eval()
