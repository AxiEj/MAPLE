"""Private finite-dielectric smooth-partition Schwarz PCM scalar."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any
import numpy as np

from .ddpcm_operators import _assemble_point_ddpcm
from .functional import SealedTorchScalar, block_permutation
from .identity import (CAPABILITIES, CONFIGURATION_CONTRACT_ID, LINEAGE_EVIDENCE_TRANSFERABLE, MODEL_ID, PROVIDER_ID, SCALAR_ID, SOURCE_AUTHOR, SOURCE_COMMIT, execution_record, implementation_hashes, lineage_hashes, sha256_payload)
from .point_l1 import POINT_SOURCE_SHELL_EVENT_TOLERANCE_ANGSTROM

@dataclass(frozen=True)
class SolveDiagnostic:
    name: str
    sigma_min: float
    relative_sigma_min: float
    kappa2: float
    residual_norm: float
    relative_residual: float
    backward_error: float
    conditioning_amplification_indicator: float

class TorchSmoothPCM(SealedTorchScalar):
    """Dense CPU float64 oracle; internal direct construction only."""
    model_id=MODEL_ID; provider_id=PROVIDER_ID; scalar_id=SCALAR_ID; configuration_contract_id=CONFIGURATION_CONTRACT_ID
    capabilities=CAPABILITIES
    source_commit=SOURCE_COMMIT
    historical_evidence_transferable=LINEAGE_EVIDENCE_TRANSFERABLE
    fixed_dimensions=True; smooth_partition=True; finite_dielectric=True

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed_configuration", False):
            raise AttributeError("torch-smooth-pcm-v1 configuration is immutable")
        object.__setattr__(self, name, value)

    def __init__(self, *, atomic_numbers: tuple[int,...], radii_angstrom: tuple[float,...], transition_width_angstrom2: float=0.08, surface_lmax: int=3, partition_lmax: int=6, partition_radial_quadrature_order: int=96, source_radial_quadrature_order: int=128, double_layer_radial_quadrature_order: int=128, dielectric: float=80.0, source_shell_clearance_angstrom: float=0.05, dtype: object=None, device: object=None, tau_accuracy: float=1e-8) -> None:
        import torch
        if dtype not in (None,torch.float64,"float64") or device not in (None,"cpu",torch.device("cpu")):
            raise ValueError("torch-smooth-pcm-v1 is sealed to CPU torch.float64")
        if not isinstance(atomic_numbers,tuple) or not atomic_numbers: raise ValueError("atomic_numbers must be a non-empty tuple")
        if any(isinstance(z,bool) or not isinstance(z,int) or z<=0 for z in atomic_numbers): raise ValueError("atomic numbers must be positive integers")
        if not isinstance(radii_angstrom,tuple) or len(radii_angstrom)!=len(atomic_numbers): raise ValueError("one explicit radius is required per atom")
        radii=tuple(float(x) for x in radii_angstrom)
        if any(not np.isfinite(x) or x<=0 for x in radii): raise ValueError("radii must be finite and positive")
        def posint(x: object,name: str) -> int:
            if isinstance(x,bool) or not isinstance(x,int) or x<1 or x>512: raise ValueError(f"{name} must be a bounded positive integer no greater than 512")
            return x
        transition=float(transition_width_angstrom2); eps=float(dielectric); tau=float(tau_accuracy)
        if not np.isfinite(transition) or transition<=0: raise ValueError("transition width must be finite and positive")
        if not np.isfinite(eps) or eps<=1: raise ValueError("dielectric must be finite and > 1")
        clearance=float(source_shell_clearance_angstrom)
        if not np.isfinite(clearance) or clearance < POINT_SOURCE_SHELL_EVENT_TOLERANCE_ANGSTROM: raise ValueError("source-shell clearance must be at least 0.05 Angstrom")
        if not np.isfinite(tau) or tau<=0: raise ValueError("tau_accuracy must be positive")
        self.atomic_numbers=atomic_numbers; self.radii_angstrom=radii; self.atom_count=len(radii)
        self.transition_width_angstrom2=transition; self.surface_lmax=posint(surface_lmax,"surface_lmax"); self.partition_lmax=posint(partition_lmax,"partition_lmax")
        self.partition_radial_quadrature_order=posint(partition_radial_quadrature_order,"partition radial order")
        self.source_radial_quadrature_order=posint(source_radial_quadrature_order,"source radial order")
        self.double_layer_radial_quadrature_order=posint(double_layer_radial_quadrature_order,"double-layer radial order")
        if self.surface_lmax > 8 or self.partition_lmax > 16: raise ValueError("harmonic lmax exceeds the dense reference budget")
        if self.atom_count * (self.surface_lmax + 1) ** 2 > 4096: raise ValueError("total dense harmonic dimension exceeds 4096")
        if (self.atom_count + 1) * self.partition_lmax > 64: raise ValueError("partition algebraic degree exceeds 64")
        self.dielectric=eps; self.source_shell_clearance_angstrom=clearance; self.tau_accuracy=tau
        self._configuration_payload_at_construction=self._configuration_payload()
        self._configuration_sha256=sha256_payload(self._configuration_payload_at_construction)
        self._sealed_configuration=True

    def _contract_hashes(self) -> dict[str,str]:
        root=Path(__file__).resolve().parent.parent
        names=("route2_plugin_spaces.py","electrostatic_pairing.py","gto_density.py")
        return {name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}

    def _configuration_payload(self) -> dict[str,object]:
        return {"identity_hierarchy":"family IDs -> configuration contract -> per-instance configuration_sha256","configuration_contract_id":self.configuration_contract_id,"model_id":self.model_id,"provider_id":self.provider_id,"scalar_id":self.scalar_id,"atomic_numbers":self.atomic_numbers,"radii_angstrom":self.radii_angstrom,"transition_width_angstrom2":self.transition_width_angstrom2,"surface_lmax":self.surface_lmax,"partition_lmax":self.partition_lmax,"partition_radial_quadrature_order":self.partition_radial_quadrature_order,"source_radial_quadrature_order":self.source_radial_quadrature_order,"double_layer_radial_quadrature_order":self.double_layer_radial_quadrature_order,"dielectric":self.dielectric,"source_shell_clearance_angstrom":self.source_shell_clearance_angstrom,"dtype":"torch.float64","device":"cpu","source_space":self.source_space.name,"field_dual_space":self.field_dual_space.name,"pairing":self.pairing.name,"raw_order":"[q,p_y,p_z,p_x]","cartesian_field_order":"[V,gx,gy,gz]","Q":block_permutation(1).tolist(),"equations":"F=-B c; A_eps G=A_inf F; L X=G; U=1/2 c.T C_raw X","source_commit":self.source_commit,"source_author":SOURCE_AUTHOR,"lineage_hashes":lineage_hashes(),"contract_hashes":self._contract_hashes(),"implementation_hashes":implementation_hashes(),"conditioning":{"sigma_min":1e-12,"relative_sigma_min":1e-10,"kappa_catastrophic":1e12,"tau_accuracy":self.tau_accuracy,"backward_error":1e-12},"capabilities":[]}

    def configuration_sha256(self) -> str:
        if self._configuration_payload()!=self._configuration_payload_at_construction: raise RuntimeError("torch smooth PCM configuration/provenance drifted")
        return self._configuration_sha256

    def execution_provenance(self) -> dict[str,object]:
        nested=execution_record(self.configuration_sha256())
        versions=nested.pop("versions")
        record={"configuration_sha256":self.configuration_sha256(),"torch_version":versions["torch"],"numpy_version":versions["numpy"],"scipy_version":versions["scipy"],"ase_version":versions["ase"],"cpu_platform":nested.pop("platform"),"cpu_processor":nested.pop("processor"),"cpu_model_features":nested.pop("cpu_model_features"),"cpu_count":nested.pop("cpu_count"),"torch_backend":nested.pop("torch_build"),"torch_parallel_backend":nested.pop("torch_parallel"),"torch_threads":nested.pop("torch_threads"),"numpy_blas":nested.pop("numpy_blas")}
        record["execution_provenance_sha256"]=sha256_payload(record)
        return record

    def _assemble_torch(self, positions: Any):
        self.configuration_sha256()
        if positions.dtype != __import__("torch").float64 or positions.device.type != "cpu": raise TypeError("positions must be CPU torch.float64")
        if self.atom_count>1:
            distances=__import__("torch").cdist(positions,positions)
            mask=~__import__("torch").eye(self.atom_count,dtype=bool)
            if bool((distances[mask]<=1e-12).any().detach()): raise ValueError("atomic centres must be distinct")
            radii=positions.new_tensor(self.radii_angstrom)[:,None]
            shell_margin=__import__("torch").abs(distances-radii)
            if bool((shell_margin[mask] < self.source_shell_clearance_angstrom).any().detach()):
                raise ValueError("point source violates the target-shell clearance")
        assembly=_assemble_point_ddpcm(positions,radii=self.radii_angstrom,transition_width=self.transition_width_angstrom2,lmax=self.surface_lmax,partition_lmax=self.partition_lmax,partition_radial_order=self.partition_radial_quadrature_order,source_radial_order=self.source_radial_quadrature_order,double_layer_radial_order=self.double_layer_radial_quadrature_order,dielectric=self.dielectric,point_monopoles_only=False)
        for name,value in assembly._asdict().items():
            if name != "partition" and not bool(__import__("torch").isfinite(value).all().detach()): raise FloatingPointError(f"assembled {name} is non-finite")
        self._matrix_gate(assembly.dielectric_operator,"A_eps")
        self._matrix_gate(assembly.schwarz_operator,"L")
        return assembly

    def _matrix_gate(self, matrix: Any, name: str) -> tuple[float,float,float]:
        torch=__import__("torch")
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not bool(torch.isfinite(matrix).all().detach()): raise ValueError(f"{name} matrix is invalid or non-finite")
        s=torch.linalg.svdvals(matrix); smax=float(s[0].detach()); smin=float(s[-1].detach()); rel=smin/smax; kappa=smax/smin
        cap=min(1e12,self.tau_accuracy/(10*np.finfo(float).eps))
        if not np.isfinite(kappa) or smin<1e-12 or rel<1e-10 or kappa>cap: raise ValueError(f"{name} failed singular-value/conditioning gate")
        return smin,rel,kappa

    def _checked_solve(self, matrix: Any, rhs: Any, name: str):
        import torch
        if not bool(torch.isfinite(rhs).all().detach()): raise FloatingPointError(f"{name} right-hand side is non-finite")
        smin,rel,kappa=self._matrix_gate(matrix,name)
        solution=torch.linalg.solve(matrix,rhs)
        if not bool(torch.isfinite(solution).all().detach()): raise FloatingPointError(f"{name} solution is non-finite")
        residual=matrix@solution-rhs
        rn=torch.linalg.vector_norm(residual); hn=torch.linalg.matrix_norm(matrix,ord=2); yn=torch.linalg.vector_norm(solution); bn=torch.linalg.vector_norm(rhs)
        denom=hn*yn+bn
        if not bool(torch.isfinite(torch.stack((rn,hn,yn,bn,denom))).all().detach()): raise FloatingPointError(f"{name} solve diagnostics are non-finite")
        eta=rn/denom if float(denom.detach()) else rn
        amplification=kappa*float(eta.detach())
        if not bool(torch.isfinite(residual).all().detach()) or float(eta.detach()) > 1e-12 or amplification > self.tau_accuracy: raise ValueError(f"{name} failed residual/backward-error/amplification gate")
        return solution

    def _solve_primal(self, a: Any, vector: Any):
        import torch
        if not bool(torch.isfinite(vector).all().detach()): raise FloatingPointError("source is non-finite")
        F=-(a.source_operator@vector)
        if not bool(torch.isfinite(F).all().detach()): raise FloatingPointError("localized solute potential is non-finite")
        dielectric_rhs=a.conductor_operator@F
        G=self._checked_solve(a.dielectric_operator,dielectric_rhs,"A_eps")
        X=self._checked_solve(a.schwarz_operator,G,"L")
        return F,G,X

    def _energy_torch(self, positions: Any, source: Any):
        a=self._assemble_torch(positions); c=source.reshape(-1); _,_,X=self._solve_primal(a,c)
        energy=0.5*(c@(a.receiver@X))
        if not bool(__import__("torch").isfinite(energy).detach()): raise FloatingPointError("PCM energy is non-finite")
        return energy

    def _diagnostic(self,H: Any,y: Any,rhs: Any,name: str) -> SolveDiagnostic:
        import torch
        residual=H@y-rhs; rn=float(torch.linalg.vector_norm(residual).detach()); hn=float(torch.linalg.matrix_norm(H,ord=2).detach()); yn=float(torch.linalg.vector_norm(y).detach()); bn=float(torch.linalg.vector_norm(rhs).detach())
        denom=hn*yn+bn; eta=rn/denom if denom else rn; smin,rel,kappa=self._matrix_gate(H,name); relative=rn/max(bn,np.finfo(float).tiny); amp=kappa*eta
        if eta>1e-12 or amp>self.tau_accuracy: raise ValueError(f"{name} failed backward-error/amplification gate")
        return SolveDiagnostic(name,smin,rel,kappa,rn,relative,eta,amp)

    def debug_geometry_matrices(self, geometry: object) -> dict[str,np.ndarray]:
        r,c=self._validated_tensors(geometry,np.zeros((self.atom_count,4),dtype=np.float64),grad_r=False,grad_c=False); a=self._assemble_torch(r)
        Q=r.new_tensor(block_permutation(self.atom_count)); C_raw=a.receiver; C_field=Q.T@C_raw
        exposed=__import__("torch").stack(tuple(item[0] for item in a.partition))
        values={"B":a.source_operator,"C_field":C_field,"C_raw":C_raw,"A_eps":a.dielectric_operator,"A_inf":a.conductor_operator,"L":a.schwarz_operator,"localized_double_layer":a.localized_double_layer,"centered_exposure_coefficients":a.centered_exposure_coefficients,"exposed_coefficients":exposed}
        result={k:v.detach().numpy().copy() for k,v in values.items()}
        if any(not np.all(np.isfinite(value)) for value in result.values()): raise FloatingPointError("debug geometry output is non-finite")
        return result

    def debug_primal_state(self, geometry: object, source: object) -> dict[str,object]:
        import torch
        r,c=self._validated_tensors(geometry,source,grad_r=False,grad_c=False); a=self._assemble_torch(r); cv=c.reshape(-1); F,G,X=self._solve_primal(a,cv); C_raw=a.receiver
        lambda_X=self._checked_solve(a.schwarz_operator.T,-0.5*(C_raw.T@cv),"L transpose"); lambda_G=self._checked_solve(a.dielectric_operator.T,lambda_X,"A_eps transpose"); lambda_F=a.conductor_operator.T@lambda_G
        if not bool(torch.isfinite(lambda_F).all()): raise FloatingPointError("lambda_F is non-finite")
        raw_grad=0.5*(C_raw@X)+a.source_operator.T@lambda_F; autograd_c=c.detach().requires_grad_(True); auto=torch.autograd.grad(self._energy_torch(r,autograd_c),autograd_c)[0].reshape(-1)
        d1=self._diagnostic(a.dielectric_operator,G,a.conductor_operator@F,"A_eps"); d2=self._diagnostic(a.schwarz_operator,X,G,"L")
        result={"F":F.detach().numpy().copy(),"G":G.detach().numpy().copy(),"X":X.detach().numpy().copy(),"U":float((0.5*cv@(C_raw@X)).detach()),"lambda_F":lambda_F.detach().numpy().copy(),"lambda_G":lambda_G.detach().numpy().copy(),"lambda_X":lambda_X.detach().numpy().copy(),"raw_gradient_kkt":raw_grad.detach().numpy().copy(),"raw_gradient_autograd":auto.detach().numpy().copy(),"kkt_gradient_max_error":float(torch.max(torch.abs(raw_grad-auto)).detach()),"conditioning":{"A_eps":d1.__dict__,"L":d2.__dict__}}
        numeric=(result["F"],result["G"],result["X"],result["lambda_F"],result["lambda_G"],result["lambda_X"],result["raw_gradient_kkt"],result["raw_gradient_autograd"],np.asarray((result["U"],result["kkt_gradient_max_error"])))
        if any(not np.all(np.isfinite(value)) for value in numeric): raise FloatingPointError("debug primal/KKT output is non-finite")
        return result

    def audit(self, geometry: object, source: object) -> dict[str,object]:
        state=self.debug_primal_state(geometry,source)
        return {"configuration_sha256":self.configuration_sha256(),"model_id":self.model_id,"provider_id":self.provider_id,"scalar_id":self.scalar_id,"configuration_contract_id":self.configuration_contract_id,"capabilities":[],"historical_evidence_transferable":False,"kkt_gradient_max_error":state["kkt_gradient_max_error"],"conditioning":state["conditioning"]}

__all__=["SolveDiagnostic","TorchSmoothPCM"]
