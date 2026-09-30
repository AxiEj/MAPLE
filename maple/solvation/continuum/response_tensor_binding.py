"""Cheap fail-closed identity/version bindings for cached Torch tensors."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Iterable, Mapping


def iter_named_tensors(value: Any, prefix: str) -> tuple[tuple[str, Any], ...]:
    """Recursively enumerate tensors without copying tensor storage to the host."""

    torch = __import__("torch")
    found: list[tuple[str, Any]] = []
    visited: set[int] = set()

    def visit(item: Any, name: str) -> None:
        if torch.is_tensor(item):
            found.append((name, item))
            return
        if item is None or isinstance(item, (str, bytes, int, float, bool)):
            return
        identity = id(item)
        if identity in visited:
            return
        visited.add(identity)
        if is_dataclass(item) and not isinstance(item, type):
            for field in fields(item):
                visit(getattr(item, field.name), f"{name}.{field.name}")
        elif isinstance(item, Mapping):
            for key in sorted(item, key=str):
                visit(item[key], f"{name}[{key!r}]")
        elif isinstance(item, (tuple, list)):
            for index, child in enumerate(item):
                visit(child, f"{name}[{index}]")
        else:
            slots = getattr(type(item), "__slots__", ())
            if isinstance(slots, str):
                slots = (slots,)
            for slot in slots:
                if slot not in {"__dict__", "__weakref__"} and hasattr(item, slot):
                    visit(getattr(item, slot), f"{name}.{slot}")
            values = getattr(item, "__dict__", None)
            if values is not None:
                for key in sorted(values):
                    visit(values[key], f"{name}.{key}")

    visit(value, prefix)
    return tuple(found)


@dataclass(frozen=True, slots=True)
class _TensorSnapshot:
    name: str
    tensor: Any
    identity: int
    version: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: Any
    device: Any


@dataclass(frozen=True, slots=True)
class TensorStateBinding:
    """Immutable binding to the exact tensor objects of one response phase."""

    snapshots: tuple[_TensorSnapshot, ...]

    @classmethod
    def capture(cls, named_tensors: Iterable[tuple[str, Any]]) -> "TensorStateBinding":
        snapshots = tuple(
            _TensorSnapshot(
                name=name,
                tensor=tensor,
                identity=id(tensor),
                version=int(tensor._version),
                shape=tuple(tensor.shape),
                stride=tuple(tensor.stride()),
                dtype=tensor.dtype,
                device=tensor.device,
            )
            for name, tensor in named_tensors
        )
        names = tuple(snapshot.name for snapshot in snapshots)
        if len(names) != len(set(names)):
            raise RuntimeError("response tensor binding contains duplicate names.")
        return cls(snapshots)

    @property
    def tensors(self) -> tuple[Any, ...]:
        """Tensor tuple suitable for ``ctx.save_for_backward``."""

        return tuple(snapshot.tensor for snapshot in self.snapshots)

    def validate(self, named_tensors: Iterable[tuple[str, Any]]) -> None:
        current = tuple(named_tensors)
        if tuple(name for name, _ in current) != tuple(
            snapshot.name for snapshot in self.snapshots
        ):
            raise RuntimeError("response tensor set changed after binding.")
        for snapshot, (_, tensor) in zip(self.snapshots, current):
            if id(tensor) != snapshot.identity:
                raise RuntimeError(
                    f"response tensor {snapshot.name} was reassigned after binding."
                )
            if (
                int(tensor._version) != snapshot.version
                or tuple(tensor.shape) != snapshot.shape
                or tuple(tensor.stride()) != snapshot.stride
                or tensor.dtype != snapshot.dtype
                or tensor.device != snapshot.device
            ):
                raise RuntimeError(
                    f"response tensor {snapshot.name} was mutated after binding."
                )


__all__ = ["TensorStateBinding", "iter_named_tensors"]
