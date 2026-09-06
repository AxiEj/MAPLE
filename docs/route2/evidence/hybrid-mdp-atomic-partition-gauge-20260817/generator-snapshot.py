from __future__ import annotations
import argparse, hashlib, json, os, pathlib, sys, time
import numpy as np
from ase import Atoms
import torch

p=argparse.ArgumentParser()
p.add_argument('--source-root',type=pathlib.Path,required=True)
p.add_argument('--input-root',type=pathlib.Path,required=True)
p.add_argument('--selection-index',type=int,required=True)
p.add_argument('--output',type=pathlib.Path,required=True)
p.add_argument('--device',default='cuda')
a=p.parse_args()
source=a.source_root.resolve(strict=True); inputs=a.input_root.resolve(strict=True)
sys.path.insert(0,str(source)); sys.path.insert(1,str(source/'docs/implicit-solvation/benchmarks'))
from mnsol_dataset import load_mnsol_protocol, load_mnsol_v2012
from mnsol_partition import validate_frozen_mnsol_partition_selection
from maple.solvation.api.profiles import MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_EVALUATOR_ID
from maple.solvation.continuum.separated_source_ddx import embed_atomic_l1_in_first_radial_channel
from maple.solvation.experimental import build_smd_mace_mdp_polar_hybrid_ddx_pes
from maple.solvation.models import MACEPolarOriginalSourceNativeFieldAdapter, build_mace_mdp_anchored_mace_polar_hybrid, build_mace_mdp_moment_adapter, build_official_mace_polar_1_m_radial_gto_adapter
from maple.function.calculator.extra_correction.implicit.smd_cds import HARTREE_TO_KCAL_MOL
from maple.solvation.api.units import HARTREE_TO_EV
from maple.function.calculator.extra_correction.implicit.gto_galerkin import AtomCenteredL1GTOBasis

mdp=pathlib.Path('/home/axie/.cache/mace/MACE-MDP.model').resolve(strict=True)
polar=pathlib.Path('/home/axie/.cache/mace/MACEPOLAR1Mmodel').resolve(strict=True)
def sh(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def stats(x):
 x=np.asarray(x,dtype=float); return {'shape':list(x.shape),'norm':float(np.linalg.norm(x)),'rms':float(np.sqrt(np.mean(x*x))),'max_abs':float(np.max(np.abs(x)))}

torch.set_num_threads(1); torch.set_num_interop_threads(1)
protocol=load_mnsol_protocol(inputs/'route2-mnsol-protocol-v1.json')
dataset=load_mnsol_v2012(inputs/'MNSolDatabase_v2012.zip',protocol)
manifest=json.loads((inputs/'route2-mnsol-development-selection-v1.private.json').read_text())
pilot=json.loads((inputs/'route2-mnsol-pilot-selection-v1.json').read_text())
selected=validate_frozen_mnsol_partition_selection(manifest,dataset,protocol,pilot)
item=selected[a.selection_index]; eligible=item.eligible_record; geom=eligible.geometry
atoms=Atoms(numbers=geom.atomic_numbers,positions=geom.coordinates_angstrom,info={'charge':geom.charge,'mult':geom.multiplicity})
started=time.perf_counter()
mdp_model=build_mace_mdp_moment_adapter(checkpoint_path=mdp,device=a.device)
radial=build_official_mace_polar_1_m_radial_gto_adapter(checkpoint_path=polar,device=a.device,long_range_evaluator_profile=MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_EVALUATOR_ID)
hybrid=build_mace_mdp_anchored_mace_polar_hybrid(permanent=mdp_model,response=MACEPolarOriginalSourceNativeFieldAdapter(radial))
pes=build_smd_mace_mdp_polar_hybrid_ddx_pes(hybrid,tuple(atoms.get_chemical_symbols()),solvent=eligible.canonical_solvent,continuum_model='pcm',lmax=15,n_lebedev=1202,solver_tolerance=1e-12,eta=0.1,n_proc=1)
electro=pes._electrostatic_pes
evaluator=electro._evaluator(atoms)
state=evaluator.solve(atoms)
anchor=evaluator.anchor
radial_source=embed_atomic_l1_in_first_radial_channel(state.induced_source4)
prepared=evaluator._prepared
zero_radial=np.zeros_like(radial_source); zero_perm=np.zeros_like(anchor.permanent_source4)
perm_state=prepared.solve(zero_radial)
total_state=prepared.solve(radial_source)
ind_prepared=evaluator._continuum.prepare(atoms,zero_perm)
ind_state=ind_prepared.solve(radial_source)
zero_state=ind_prepared.solve(zero_radial)
energies={'permanent_only_ev':perm_state.polarization_energy_ev,'induced_only_ev':ind_state.polarization_energy_ev,'total_ev':total_state.polarization_energy_ev,'zero_ev':zero_state.polarization_energy_ev}
energies['cross_ev']=energies['total_ev']-energies['permanent_only_ev']-energies['induced_only_ev']
ev_to_kcal=HARTREE_TO_KCAL_MOL/HARTREE_TO_EV
energies['permanent_only_kcal_mol']=energies['permanent_only_ev']*ev_to_kcal
energies['induced_only_kcal_mol']=energies['induced_only_ev']*ev_to_kcal
energies['cross_kcal_mol']=energies['cross_ev']*ev_to_kcal
energies['total_kcal_mol']=energies['total_ev']*ev_to_kcal

# Target-independent atomic-partition gauge witness.  MACE-MDP is supervised
# on the total molecular dipole, so +delta/-delta redistribution of two local
# atomic dipoles is invisible to that public observable.  Select the most
# continuum-sensitive such direction without consulting the experimental
# solvation target, then exploit the fixed-cavity quadratic identity to obtain
# the exact energy change with one homogeneous ddX solve.
g_perm=np.asarray(perm_state.permanent_energy_gradient,dtype=float)
best=None
for raw_component in (1,2,3):
    hi=int(np.argmax(g_perm[:,raw_component]))
    lo=int(np.argmin(g_perm[:,raw_component]))
    sensitivity=abs(float(g_perm[hi,raw_component]-g_perm[lo,raw_component]))
    candidate=(sensitivity,raw_component,hi,lo)
    if best is None or candidate>best:
        best=candidate
assert best is not None
_,gauge_component,gauge_i,gauge_j=best
gauge_step=0.01
gauge_direction=np.zeros_like(anchor.permanent_source4)
gauge_direction[gauge_i,gauge_component]=gauge_step
gauge_direction[gauge_j,gauge_component]=-gauge_step
constraints=AtomCenteredL1GTOBasis((1.5,)).molecular_charge_dipole_constraints(np.asarray(atoms.positions,dtype=float))
constraint_residual=constraints@gauge_direction.reshape(-1)
psi_delta=(np.asarray(prepared._permanent_psi_matrix)@gauge_direction.reshape(-1)).reshape(np.asarray(prepared._permanent_psi).shape)
phi_delta=np.asarray(prepared._permanent_phi_matrix)@gauge_direction.reshape(-1)
_,gauge_self_ev,_=prepared._problem.solve_general(psi_delta,phi_delta)
gauge_linear_ev=float(np.vdot(g_perm,gauge_direction))
gauge_plus_ev=energies['permanent_only_ev']+gauge_linear_ev+gauge_self_ev
gauge_minus_ev=energies['permanent_only_ev']-gauge_linear_ev+gauge_self_ev
_,gauge_plus_direct_ev,_=prepared._problem.solve_general(np.asarray(prepared._permanent_psi)+psi_delta,np.asarray(prepared._permanent_phi)+phi_delta)
_,gauge_minus_direct_ev,_=prepared._problem.solve_general(np.asarray(prepared._permanent_psi)-psi_delta,np.asarray(prepared._permanent_phi)-phi_delta)
psi_i,phi_i=prepared._radial_problem_data(radial_source)
phi_p=np.asarray(prepared._permanent_phi); phi_t=phi_p+phi_i
first_induced=hybrid.induced_source(atoms,anchor,perm_state.model_field)
record={
 'artifact':'route2-hybrid-development-tail-component-diagnostic-v1',
 'claim_boundary':{'development_target_opened':True,'confirmation_opened':False,'fitting_or_calibration_performed':False,'diagnostic_only':True,'private_attributes_used_for_component_attribution':True,'public_capability_admitted':False},
 'source_root':str(source),'source_git_head':os.popen(f"git -C {source} rev-parse HEAD").read().strip(),
 'source_files':{str(path.relative_to(source)):sh(path) for path in [source/'maple/solvation/continuum/separated_source_ddx.py',source/'maple/solvation/experimental/mace_mdp_polar_ddx.py',source/'maple/solvation/models/mace_mdp_polar_hybrid.py']},
 'checkpoints':{'mdp_sha256':sh(mdp),'polar_sha256':sh(polar)},'device':a.device,
 'identity':{'selection_index':a.selection_index,'opaque_record_id':item.opaque_record_id,'dataset_row_sha256':eligible.record.raw_row_sha256,'name':eligible.record.solute_name,'formula':eligible.record.formula,'solvent':eligible.canonical_solvent,'experimental_delta_g_kcal_mol':eligible.record.delta_g_kcal_mol,'geometry_sha256':geom.sha256,'atom_count':len(atoms)},
 'root':{'residual_ev':state.primal_residual_ev,'cold_iterations':state.cold_iterations,'wide_iterations':state.wide_iterations,'polarization_energy_kcal_mol':state.polarization_energy_ev*ev_to_kcal},
 'sources':{'permanent_total_charge_e':float(np.sum(anchor.permanent_source4[:,0])),'induced_total_charge_e':float(np.sum(state.induced_source4[:,0])),'permanent':stats(anchor.permanent_source4),'induced':stats(state.induced_source4),'first_induced':stats(first_induced),'converged_minus_first_induced':stats(state.induced_source4-first_induced)},
 'surface_potential':{'permanent':stats(phi_p),'induced':stats(phi_i),'total':stats(phi_t)},
 'model_field':{'permanent':stats(perm_state.model_field),'induced':stats(ind_state.model_field),'total':stats(total_state.model_field),'linearity_residual':stats(total_state.model_field-perm_state.model_field-ind_state.model_field)},
 'continuum_components':energies,
 'atomic_partition_gauge_witness':{
   'claim':'fixed-cavity target-independent redistribution preserving total charge and molecular dipole exactly',
   'step_e_angstrom':gauge_step,
   'raw_component':int(gauge_component),
   'atom_i':int(gauge_i),'atom_j':int(gauge_j),
   'constraint_residual_q_mu':constraint_residual.tolist(),
   'linear_energy_change_plus_ev':gauge_linear_ev,
   'direction_self_energy_ev':float(gauge_self_ev),
   'plus_energy_ev':float(gauge_plus_ev),'minus_energy_ev':float(gauge_minus_ev),
   'plus_direct_replay_ev':float(gauge_plus_direct_ev),'minus_direct_replay_ev':float(gauge_minus_direct_ev),
   'plus_change_kcal_mol':float((gauge_plus_ev-energies['permanent_only_ev'])*ev_to_kcal),
   'minus_change_kcal_mol':float((gauge_minus_ev-energies['permanent_only_ev'])*ev_to_kcal),
 },
 'wall_seconds':time.perf_counter()-started,
}
record['computed_total_minus_root_ev']=energies['total_ev']-state.polarization_energy_ev
out=a.output.resolve(); out.parent.mkdir(parents=True,exist_ok=True)
with out.open('x') as f: json.dump(record,f,indent=2,sort_keys=True,allow_nan=False); f.write('\n')
print(json.dumps({'output':str(out),'components':energies,'wall_seconds':record['wall_seconds']},sort_keys=True),flush=True)
