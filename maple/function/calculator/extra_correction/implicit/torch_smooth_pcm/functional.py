"""Sealed same-scalar autograd API for Torch smooth PCM."""
from __future__ import annotations

from typing import Any
import numpy as np

from ..electrostatic_pairing import MACE_POLAR_L1_PAIRING
from ..route2_plugin_spaces import ATOMIC_L1_PLUGIN_FIELD_DUAL_SPACE, ATOMIC_L1_PLUGIN_SOURCE_SPACE

_FIELD_TO_RAW = MACE_POLAR_L1_PAIRING.field_to_density_indices
Q_RAW_FROM_CARTESIAN = tuple(
    tuple(float(column == selected) for column in range(4))
    for selected in _FIELD_TO_RAW
)

def block_permutation(atom_count: int) -> np.ndarray:
    if isinstance(atom_count, bool) or not isinstance(atom_count, int) or atom_count < 1:
        raise ValueError("atom_count must be a positive integer")
    return np.kron(np.eye(atom_count), np.asarray(Q_RAW_FROM_CARTESIAN))

class SealedTorchScalar:
    """All public derivatives are final and generated from ``_energy_torch``."""
    source_space = ATOMIC_L1_PLUGIN_SOURCE_SPACE
    field_dual_space = ATOMIC_L1_PLUGIN_FIELD_DUAL_SPACE
    pairing = MACE_POLAR_L1_PAIRING
    _FINAL = frozenset({"energy","drive_cartesian","coordinate_gradient","energy_drive_gradient","source_jvp","source_vjp","debug_source_hvp_raw","mixed_coordinate_vjp","joint_hvp"})

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        inherited = getattr(cls, "_FINAL", frozenset())
        illegal = sorted(name for name in inherited if name in cls.__dict__)
        if illegal:
            raise TypeError("same-scalar derivative methods are final: " + ", ".join(illegal))

    def _validated_tensors(self, geometry: object, source: object, *, grad_r: bool, grad_c: bool):
        import torch
        r_np=np.asarray(geometry)
        c_np=np.asarray(source)
        if r_np.dtype != np.float64 or c_np.dtype != np.float64:
            raise TypeError("geometry and source must be NumPy float64 arrays")
        if r_np.shape != (self.atom_count,3) or c_np.shape != (self.atom_count,4):
            raise ValueError("geometry/source shape mismatch; coincident-centre inputs are also invalid")
        if not np.all(np.isfinite(r_np)) or not np.all(np.isfinite(c_np)):
            raise ValueError("geometry and source must be finite")
        r=torch.tensor(r_np, dtype=torch.float64, device="cpu", requires_grad=grad_r)
        c=torch.tensor(c_np, dtype=torch.float64, device="cpu", requires_grad=grad_c)
        return r,c

    @staticmethod
    def _strict_array(values: object, shape: tuple[int, ...], name: str) -> np.ndarray:
        array=np.asarray(values)
        if array.dtype != np.float64: raise TypeError(f"{name} must be NumPy float64")
        if array.shape != shape: raise ValueError(f"{name} must have shape {shape}")
        if not np.all(np.isfinite(array)): raise ValueError(f"{name} must be finite")
        return array

    @staticmethod
    def _finite_output(value: Any, name: str) -> Any:
        import torch
        if not bool(torch.isfinite(value).all().detach()): raise FloatingPointError(f"{name} is non-finite")
        return value

    def _q_torch(self, reference: Any):
        import torch
        return torch.kron(torch.eye(self.atom_count,dtype=torch.float64,device="cpu"), reference.new_tensor(Q_RAW_FROM_CARTESIAN))

    def energy(self, geometry: object, source: object) -> float:
        r,c=self._validated_tensors(geometry,source,grad_r=False,grad_c=False)
        return float(self._finite_output(self._energy_torch(r,c),"energy").detach())

    def drive_cartesian(self, geometry: object, source: object) -> np.ndarray:
        import torch
        r,c=self._validated_tensors(geometry,source,grad_r=False,grad_c=True)
        raw=torch.autograd.grad(self._energy_torch(r,c),c)[0].reshape(-1)
        result=(self._q_torch(raw).T@raw).reshape(self.atom_count,4)
        return self._finite_output(result,"Cartesian drive").detach().numpy().copy()

    def coordinate_gradient(self, geometry: object, source: object) -> np.ndarray:
        import torch
        r,c=self._validated_tensors(geometry,source,grad_r=True,grad_c=False)
        gradient=torch.autograd.grad(self._energy_torch(r,c),r,allow_unused=True)[0]
        if gradient is None:
            if self.atom_count != 1: raise RuntimeError("multi-sphere coordinate graph is disconnected")
            gradient=torch.zeros_like(r)
        return self._finite_output(gradient,"coordinate gradient").detach().numpy().copy()

    def energy_drive_gradient(self, geometry: object, source: object) -> tuple[float,np.ndarray,np.ndarray]:
        import torch
        r,c=self._validated_tensors(geometry,source,grad_r=True,grad_c=True)
        e=self._energy_torch(r,c)
        gr,gc=torch.autograd.grad(e,(r,c),allow_unused=True)
        if gr is None:
            if self.atom_count != 1: raise RuntimeError("multi-sphere coordinate graph is disconnected")
            gr=torch.zeros_like(r)
        drive=(self._q_torch(gc).T@gc.reshape(-1)).reshape(self.atom_count,4)
        return float(self._finite_output(e,"energy").detach()),self._finite_output(drive,"Cartesian drive").detach().numpy().copy(),self._finite_output(gr,"coordinate gradient").detach().numpy().copy()

    def source_jvp(self, geometry: object, source: object, direction_raw: object) -> np.ndarray:
        import torch
        r,c=self._validated_tensors(geometry,source,grad_r=False,grad_c=False)
        v=self._strict_array(direction_raw,(self.atom_count,4),"raw direction")
        def f(x): return self._energy_torch(r,x)
        _,hv=torch.autograd.functional.hvp(f,c,torch.tensor(v,dtype=torch.float64))
        result=(self._q_torch(hv).T@hv.reshape(-1)).reshape(self.atom_count,4)
        return self._finite_output(result,"source JVP").detach().numpy().copy()

    def debug_source_hvp_raw(self, geometry: object, source: object, direction_raw: object) -> np.ndarray:
        import torch
        r,c=self._validated_tensors(geometry,source,grad_r=False,grad_c=False)
        v=torch.tensor(self._strict_array(direction_raw,(self.atom_count,4),"raw direction"),dtype=torch.float64)
        _,hv=torch.autograd.functional.hvp(lambda x:self._energy_torch(r,x),c,v)
        return self._finite_output(hv,"raw source HVP").detach().numpy().copy()

    def source_vjp(self, geometry: object, source: object, cotangent_cartesian: object) -> np.ndarray:
        import torch
        r,c=self._validated_tensors(geometry,source,grad_r=False,grad_c=True)
        w=self._strict_array(cotangent_cartesian,(self.atom_count,4),"Cartesian cotangent")
        raw=torch.autograd.grad(self._energy_torch(r,c),c,create_graph=True)[0].reshape(-1)
        drive=self._q_torch(raw).T@raw
        result=torch.autograd.grad(drive,c,grad_outputs=torch.tensor(w.reshape(-1),dtype=torch.float64))[0]
        return self._finite_output(result,"source VJP").detach().numpy().copy()

    def mixed_coordinate_vjp(self, geometry: object, source: object, cotangent_cartesian: object) -> np.ndarray:
        import torch
        r,c=self._validated_tensors(geometry,source,grad_r=True,grad_c=True)
        w=self._strict_array(cotangent_cartesian,(self.atom_count,4),"Cartesian cotangent")
        raw=torch.autograd.grad(self._energy_torch(r,c),c,create_graph=True)[0].reshape(-1)
        drive=self._q_torch(raw).T@raw
        result=torch.autograd.grad(drive,r,grad_outputs=torch.tensor(w.reshape(-1),dtype=torch.float64),allow_unused=True)[0]
        if result is None:
            if self.atom_count != 1: raise RuntimeError("multi-sphere mixed coordinate graph is disconnected")
            result=torch.zeros_like(r)
        return self._finite_output(result,"mixed coordinate VJP").detach().numpy().copy()

    def joint_hvp(self, geometry: object, source: object, coordinate_direction: object, source_direction_raw: object) -> tuple[np.ndarray,np.ndarray]:
        import torch
        r,c=self._validated_tensors(geometry,source,grad_r=False,grad_c=False)
        dr=torch.tensor(self._strict_array(coordinate_direction,(self.atom_count,3),"coordinate direction"),dtype=torch.float64)
        dc=torch.tensor(self._strict_array(source_direction_raw,(self.atom_count,4),"raw source direction"),dtype=torch.float64)
        _,hvp=torch.autograd.functional.hvp(lambda rr,cc:self._energy_torch(rr,cc),(r,c),(dr,dc))
        return self._finite_output(hvp[0],"coordinate HVP").detach().numpy().copy(),self._finite_output(hvp[1],"source HVP").detach().numpy().copy()
