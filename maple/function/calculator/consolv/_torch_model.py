"""Pure-PyTorch inference graph for the frozen ConSolv parameterization.

This is a model-specific implementation of the published MACE_CONSOL equations,
not a subclass of the standard MACE forward path.  Checkpoints are plain
``weights_only=True`` dictionaries produced by the repository conversion tool.
"""

from __future__ import annotations

import math
import hashlib
from pathlib import Path
from typing import Iterable

import torch
from e3nn import o3

FORMAT = "consolv-torch-v1"
SOURCE_REVISION = "06801dd5f53182ed86c31eee1fa03cc011c05b5a"
CHECKPOINT_IDENTITIES = {
    "solution": "534114cd0a45d97308eb28506ccfe731df143ae5fb0c6d611b6c6c0cb1656767",
    "vacuum": "d0369560cd34312604d8f4df3a002c7544c8867236ce50a6808c75794a5195f0",
}
NORMALIZED_SILU_SCALE = 1.6765648126602173
GROUPS_0 = ((64, 0),)
GROUPS_HIDDEN = ((64, 0), (32, 1))
GROUPS_INTERACTION = ((64, 0), (64, 1), (64, 2), (64, 3))
GROUPS_MESSAGES = (
    ((64, 0), (64, 1), (64, 2), (64, 3)),
    ((96, 0), (128, 1), (128, 2), (96, 3)),
)
GROUPS_DOWN = ((32, 0), (32, 1), (32, 2), (32, 3))


def _slices(groups: Iterable[tuple[int, int]]):
    start = 0
    result = {}
    for mul, ell in groups:
        stop = start + mul * (2 * ell + 1)
        result[ell] = (slice(start, stop), mul, 2 * ell + 1)
        start = stop
    return result


def _linear(x, groups_in, groups_out, weights):
    """Apply an e3nn-JAX Linear with default element/path normalization."""
    si, so = _slices(groups_in), _slices(groups_out)
    chunks = []
    for ell, (_, out_mul, dim) in so.items():
        xs, in_mul, _ = si[ell]
        block = x[..., xs].reshape(*x.shape[:-1], in_mul, dim)
        weight = weights[ell]
        out = torch.einsum("...mi,mo->...oi", block, weight)
        chunks.append((out / math.sqrt(in_mul)).reshape(*x.shape[:-1], out_mul * dim))
    return torch.cat(chunks, dim=-1)


def _indexed_linear(x, groups_in, groups_out, weights, species):
    si, so = _slices(groups_in), _slices(groups_out)
    chunks = []
    for ell, (_, out_mul, dim) in so.items():
        xs, in_mul, _ = si[ell]
        block = x[..., xs].reshape(x.shape[0], in_mul, dim)
        weight = weights[ell][species]
        out = torch.einsum("nmi,nmo->noi", block, weight)
        chunks.append((out / math.sqrt(in_mul)).reshape(x.shape[0], out_mul * dim))
    return torch.cat(chunks, dim=-1)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _ConSolvTorchModel(torch.nn.Module):
    """Frozen two-interaction ConSolv graph with energy-derived forces."""

    def __init__(self, checkpoint: dict):
        super().__init__()
        if checkpoint.get("format") != FORMAT:
            raise ValueError(
                f"Unsupported ConSolv checkpoint format: {checkpoint.get('format')!r}"
            )
        self.metadata = dict(checkpoint["metadata"])
        self.parameter_map = dict(checkpoint["parameter_map"])
        for name, value in checkpoint["state_dict"].items():
            self.register_buffer(name, value.detach().clone())
        self.tp0 = o3.FullTensorProduct(
            "64x0e",
            "1x1o+1x2e+1x3o",
            filter_ir_out=[o3.Irrep(x) for x in ("0e", "1o", "2e", "3o")],
        )
        self.tp1 = o3.FullTensorProduct(
            "64x0e+32x1o",
            "1x1o+1x2e+1x3o",
            filter_ir_out=[o3.Irrep(x) for x in ("0e", "1o", "2e", "3o")],
        )

    def _p(self, source_name: str):
        return getattr(self, self.parameter_map[source_name])

    def _attention_embedding(self, species, solvent_index):
        atom = self._p("mace_consol/~/linear_node_embedding_block/~/embed::embeddings")[
            species
        ]
        atom = atom @ self._p("mace_consol/~/linear_node_embedding_block/~/linear::w")
        descriptor = self.descriptor[solvent_index]
        projected = descriptor[:, None] @ self._p(
            "mace_consol/~/linear_node_embedding_block/~/non_linear_solvent_embedding_block4/~/linear::w"
        ) + self._p(
            "mace_consol/~/linear_node_embedding_block/~/non_linear_solvent_embedding_block4/~/linear::b"
        )
        prefix = "mace_consol/~/linear_node_embedding_block/~/non_linear_solvent_embedding_block4/~/attention/~/multi_head_attention"
        q = atom @ self._p(prefix + "/query::w") + self._p(prefix + "/query::b")
        k = projected @ self._p(prefix + "/key::w") + self._p(prefix + "/key::b")
        v = projected @ self._p(prefix + "/value::w") + self._p(prefix + "/value::b")
        q = q.reshape(len(species), 8, 64)
        k = k.reshape(15, 8, 64)
        v = v.reshape(15, 8, 64)
        logits = torch.einsum("nhd,shd->hns", q, k) / 8.0
        attended = torch.einsum("hns,shd->nhd", logits.softmax(-1), v).reshape(
            len(species), 512
        )
        attended = attended @ self._p(prefix + "/linear::w") + self._p(
            prefix + "/linear::b"
        )
        return atom + attended, descriptor

    def _rbf(self, normalized_distance):
        x = normalized_distance[:, None]
        p = 6
        envelope = torch.where(
            x < 1,
            1
            - (p + 1) * (p + 2) / 2 * x**p
            + p * (p + 2) * x ** (p + 1)
            - p * (p + 1) / 2 * x ** (p + 2),
            torch.zeros_like(x),
        )
        freq = self._p("mace_consol/~/BesselRadial::RBF_Frequencies")
        return envelope * math.sqrt(2.0) * torch.sin(freq * x) / x.clamp_min(1e-15)

    @staticmethod
    def _valid_edges(positions_angstrom):
        n = positions_angstrom.shape[0]
        receiver, sender = torch.meshgrid(
            torch.arange(n, device=positions_angstrom.device),
            torch.arange(n, device=positions_angstrom.device),
            indexing="ij",
        )
        sender, receiver = sender.reshape(-1), receiver.reshape(-1)
        vector_angstrom = positions_angstrom[sender] - positions_angstrom[receiver]
        keep = (sender != receiver) & (vector_angstrom.square().sum(-1) < 25.0)
        return sender[keep], receiver[keep], vector_angstrom[keep] / 5.0

    def _message(self, layer, node, rbf, vectors, sender, receiver, descriptor):
        in_groups = GROUPS_0 if layer == 0 else GROUPS_HIDDEN
        up = _linear(
            node,
            in_groups,
            in_groups,
            {
                ell: self._p(
                    f"mace_consol/layer_{layer}/interaction_layer/linear_up::w[{ell},{ell}] {mul}x{o3.Irrep((ell, 1 if ell % 2 == 0 else -1))},{mul}x{o3.Irrep((ell, 1 if ell % 2 == 0 else -1))}"
                )
                for mul, ell in in_groups
            },
        )
        sh = o3.spherical_harmonics(
            [1, 2, 3], vectors, normalize=True, normalization="component"
        )
        tp_module = self.tp0 if layer == 0 else self.tp1
        tp = tp_module(up[sender], sh)
        tp_slices = []
        start = 0
        for mul, ir in tp_module.irreps_out:
            stop = start + mul * ir.dim
            tp_slices.append(
                (ir.l, tp[:, start:stop].reshape(tp.shape[0], mul, ir.dim))
            )
            start = stop
        source_slices = _slices(in_groups)
        messages = []
        for ell in range(4):
            parts = []
            if ell in source_slices:
                sl, mul, dim = source_slices[ell]
                parts.append(up[sender, sl].reshape(len(sender), mul, dim))
            parts.extend(value for degree, value in tp_slices if degree == ell)
            messages.append(torch.cat(parts, dim=1))

        prefix = f"mace_consol/layer_{layer}/interaction_layer/message_passing_convolution_layer"
        y = torch.nn.functional.silu(
            descriptor
            @ (self._p(prefix + "/linear::w[0,0] 15x0e,64x0e") / math.sqrt(15.0))
        )
        y = y @ (self._p(prefix + "/linear_1::w[0,0] 64x0e,8x0e") / 8.0)
        radial = rbf * (1 + y)
        for i in range(4):
            input_size = radial.shape[-1]
            radial = radial @ self._p(prefix + f"/multi_layer_perceptron/linear_{i}::w")
            radial = radial / math.sqrt(input_size)
            if i < 3:
                radial = NORMALIZED_SILU_SCALE * torch.nn.functional.silu(radial)
        mixed, offset = [], 0
        for part in messages:
            mul = part.shape[1]
            mixed.append(
                part
                * radial[:, offset : offset + mul, None]
                / sum(m for m, _ in in_groups)
            )
            offset += mul
        zeros = [
            torch.zeros(
                (node.shape[0], *part.shape[1:]), dtype=node.dtype, device=node.device
            )
            for part in mixed
        ]
        for out, part in zip(zeros, mixed):
            out.index_add_(0, receiver, part)
        epsilon = 1 / torch.sqrt(1 + self._p("mace_consol::varepsilon") ** 2)
        flat = (
            torch.cat([part.flatten(1) for part in zeros], dim=-1) * epsilon * epsilon
        )
        down_weights = {}
        for out_mul, ell in GROUPS_DOWN:
            in_mul = GROUPS_MESSAGES[layer][ell][0]
            parity = "e" if ell % 2 == 0 else "o"
            down_weights[ell] = self._p(
                f"mace_consol/layer_{layer}/interaction_layer/linear_down::w[{ell},{ell}] "
                f"{in_mul}x{ell}{parity},{out_mul}x{ell}{parity}"
            )
        return _linear(flat, GROUPS_MESSAGES[layer], GROUPS_DOWN, down_weights)

    def _symmetric(self, layer, node, species):
        slices = _slices(GROUPS_DOWN)
        x = torch.cat(
            [
                node[:, sl].reshape(node.shape[0], mul, dim).transpose(1, 2)
                for sl, mul, dim in slices.values()
            ],
            dim=1,
        ).transpose(
            1, 2
        )  # [N, 32, 16]
        outputs = []
        irreps = ("0e", "1o") if layer == 0 else ("0e",)
        base = f"mace_consol/layer_{layer}/equivariant_product_basis_layer/~/symmetric_contraction_layer"
        for ir in irreps:
            u3, u2, u1 = (getattr(self, f"basis_order{o}_{ir}") for o in (3, 2, 1))
            w3, w2, w1 = (self._p(f"{base}::w{o}_{ir}")[species] for o in (3, 2, 1))
            value = torch.einsum("abjki,nkf,nfj->nfabi", u3, w3, x)
            value = value + torch.einsum("abki,nkf->nfabi", u2, w2)
            value = torch.einsum("nfaji,nfj->nfai", value, x)
            value = value + torch.einsum("aki,nkf->nfai", u1, w1)
            value = torch.einsum("nfji,nfj->nfi", value, x)
            outputs.append(value.flatten(1))
        contracted_groups = ((32, 0), (32, 1)) if layer == 0 else ((32, 0),)
        target_groups = GROUPS_HIDDEN if layer == 0 else ((64, 0),)
        weights = {}
        for out_mul, ell in target_groups:
            parity = "e" if ell % 2 == 0 else "o"
            weights[ell] = self._p(
                f"mace_consol/layer_{layer}/equivariant_product_basis_layer/linear::w[{ell},{ell}] "
                f"32x{ell}{parity},{out_mul}x{ell}{parity}"
            )
        return _linear(
            torch.cat(outputs, -1), contracted_groups, target_groups, weights
        )

    def energy_native(self, numbers, positions_angstrom, solvent_index):
        if positions_angstrom.shape != (numbers.numel(), 3):
            raise ValueError("positions must have shape (N, 3)")
        species = numbers.to(torch.long) - 1
        if torch.any(species < 0) or torch.any(species >= 55):
            raise ValueError("atomic number outside ConSolv tensor capacity")
        solvent_index = int(solvent_index)
        if not 0 <= solvent_index < 148:
            raise ValueError("solvent index outside ConSolv tensor capacity")
        node, descriptor = self._attention_embedding(species, solvent_index)
        sender, receiver, vectors = self._valid_edges(positions_angstrom)
        rbf = self._rbf(torch.linalg.vector_norm(vectors, dim=-1))
        for layer in range(2):
            old = node
            node = self._message(
                layer, node, rbf, vectors, sender, receiver, descriptor
            )
            if layer == 0:
                skip_weights = {}
                for mul, ell in GROUPS_DOWN:
                    parity = "e" if ell % 2 == 0 else "o"
                    skip_weights[ell] = self._p(
                        f"mace_consol/layer_0/skip_tp_first::w[{ell},{ell}] "
                        f"32x{ell}{parity},32x{ell}{parity}"
                    )
                node = _indexed_linear(
                    node, GROUPS_DOWN, GROUPS_DOWN, skip_weights, species
                )
            node = self._symmetric(layer, node, species)
            if layer == 1:
                skip = self._p("mace_consol/layer_1/skip_tp::w[0,0] 64x0e,64x0e")
                node = (
                    node + torch.einsum("ni,nio->no", old[:, :64], skip[species]) / 8.0
                )
        w0 = self._p(
            "mace_consol/layer_1/non_linear_readout_layer/linear::w[0,0] 64x0e,16x0e"
        )
        w1 = self._p(
            "mace_consol/layer_1/non_linear_readout_layer/linear_1::w[0,0] 16x0e,1x0e"
        )
        per_atom = (
            NORMALIZED_SILU_SCALE * torch.nn.functional.silu(node @ (w0 / 8.0))
        ) @ (w1 / 4.0)
        per_atom = per_atom[:, 0]
        scale = self._p("atomic_energy_layer/~/scale_shift_layer::scale")
        shift = self._p("atomic_energy_layer/~/scale_shift_layer::shift")
        atomic = self._p("atomic_energy_layer/~/embed::embeddings")[species, 0]
        return (atomic + scale * per_atom + shift).sum(dtype=torch.float64)

    def energy_ev(self, numbers, positions_angstrom, solvent_index):
        return self.energy_native(numbers, positions_angstrom, solvent_index) / 96.485

    def energy_and_forces_ev(self, numbers, positions_angstrom, solvent_index):
        positions = positions_angstrom.detach().clone().requires_grad_(True)
        energy = self.energy_ev(numbers, positions, solvent_index)
        forces = -torch.autograd.grad(energy, positions, create_graph=False)[0]
        return energy, forces


def load_consolv_torch_checkpoint(
    path, *, device="cpu", dtype=torch.float64, expected_state="solution"
):
    """Internal CPU/FP64 diagnostic loader, including the paired vacuum state.

    This raw graph does not enforce the calculator's chemistry/task domain.
    Supported calculations use ConSolvCalculator.
    """
    if str(device).strip().lower() != "cpu":
        raise ValueError("ConSolv diagnostic loading requires CPU.")
    if dtype is not torch.float64:
        raise ValueError("ConSolv diagnostic loading requires FP64 (torch.float64).")
    if expected_state not in CHECKPOINT_IDENTITIES:
        raise ValueError(f"Unsupported ConSolv potential state: {expected_state!r}")
    checkpoint = Path(path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"ConSolv Torch checkpoint not found: {checkpoint}")
    actual = _sha256(checkpoint)
    expected = CHECKPOINT_IDENTITIES[expected_state]
    if actual != expected:
        identified_state = next(
            (
                state
                for state, digest in CHECKPOINT_IDENTITIES.items()
                if digest == actual
            ),
            None,
        )
        if identified_state is not None:
            raise ValueError(
                f"ConSolv checkpoint is {identified_state!r}, not requested "
                f"{expected_state!r}."
            )
        raise ValueError(
            "Unrecognized ConSolv Torch checkpoint identity: "
            f"expected {expected}, got {actual}."
        )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    metadata = payload.get("metadata", {})
    if metadata.get("source_revision") != SOURCE_REVISION:
        raise ValueError("ConSolv converted checkpoint source revision mismatch.")
    recorded_state = metadata.get("potential_state")
    # The frozen solution artifact predates the explicit state field; its exact
    # SHA-256 above is the state identity. All newer artifacts must self-label.
    if recorded_state not in {
        expected_state,
        None if expected_state == "solution" else "",
    }:
        raise ValueError(
            f"ConSolv checkpoint metadata state mismatch: {recorded_state!r}."
        )
    model = _ConSolvTorchModel(payload).to(device=device, dtype=dtype)
    model.eval()
    return model
