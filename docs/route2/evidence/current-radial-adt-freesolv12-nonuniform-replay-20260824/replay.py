import json,glob,hashlib,time
from pathlib import Path
import numpy as np
from pyscf import lib
from ase import Atoms
from ase.units import Bohr,Hartree
from maple.function.calculator.extra_correction.implicit.smd_cds import route2_coulomb_radii
from maple.function.calculator.extra_correction.implicit.route2_static_surface_mep import build_route2_smd_exterior_probe_surface
from maple.function.calculator.extra_correction.implicit.route2_nonuniform_response import weighted_response_matrix_discrepancy,molecular_dipole_response_from_density_coefficients
from maple.function.route2_smd_profiles import DDPCM_MULTISOLVENT_SMD_DIRECT_PCM_V2_PROFILE
from maple.solvation.api.profiles import MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_EVALUATOR_ID
from maple.solvation.models import MACEPolarOriginalSourceNativeFieldAdapter,build_mace_mdp_moment_adapter,build_mdp_polar_role_separated_adt_response,build_official_mace_polar_1_m_radial_gto_adapter
from maple.solvation.coupling.exact_gto import MACEPolarRadialGTOCoupling,FixedSurfaceGeometry,embed_mace_polar_learned_source
pub=json.load(open('docs/implicit-solvation/benchmarks/route2-v0-freesolv12-mace-localized-response-execution-62a8e413.json'))
static_root=Path('/home/axie/MAPLE/MAPLE-implicitsolv-route2/.omx/benchmarks/route2-v0-freesolv12-zero-field-static-mep-edfae78e/records')
resp_root=Path('/home/axie/MAPLE/MAPLE-implicitsolv-route2/.omx/benchmarks/route2-v0-freesolv12-localized-response-62a8e413/records')
mdp=build_mace_mdp_moment_adapter(checkpoint_path='/home/axie/.cache/mace/MACE-MDP.model',device='cpu')
radial=build_official_mace_polar_1_m_radial_gto_adapter(checkpoint_path='/home/axie/.cache/mace/MACEPOLAR1Mmodel',device='cuda',long_range_evaluator_profile=MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_EVALUATOR_ID)
response=build_mdp_polar_role_separated_adt_response(mdp=mdp,base=MACEPolarOriginalSourceNativeFieldAdapter(radial),source_root=Path.cwd())
coupling=MACEPolarRadialGTOCoupling();records=[]
for old in pub['records']:
 cid=old['compound_id'];t=time.time();sd=static_root/cid/'qm-attempt-001';rd=resp_root/cid/'attempt-001';mol=lib.chkfile.load_mol(str(sd/'gas/gas.chk'));atoms=Atoms(numbers=mol.atom_charges(),positions=mol.atom_coords()*Bohr,info={'charge':int(mol.charge),'multiplicity':int(mol.spin+1)})
 with np.load(sd/'surface.npz') as z:points=np.asarray(z['surface_points_bohr'],float)
 radii=route2_coulomb_radii(atoms.get_chemical_symbols(),solvent='water',profile=DDPCM_MULTISOLVENT_SMD_DIRECT_PCM_V2_PROFILE);surf=build_route2_smd_exterior_probe_surface(atoms.positions,radii,clearance_angstrom=1.0)
 if surf.surface_points_bohr.shape!=points.shape or np.max(abs(surf.surface_points_bohr-points))>1e-7:raise RuntimeError((cid,'surface'))
 with np.load(rd/'localized-modes.npz') as z:sources=np.asarray(z['source_points_bohr'],float)
 with np.load(rd/'qm-response.npz') as z:
  steps=np.asarray(z['field_steps_e'],float);idx=int(np.argmin(abs(steps-3e-4)));qm_mep=np.asarray(z['induced_surface_mep_hartree_per_e_per_source_e'][idx],float).T;qm_dip=np.asarray(z['induced_dipole_e_bohr_per_source_e'][idx],float)
 geom=FixedSurfaceGeometry(atoms.positions,sources);Bsrc=coupling.surface_operator(geom);zero=np.zeros((len(atoms),8));chart=response.chart_for_geometry(atoms)
 matrices={k:[] for k in ['original','radial_residual','adt','total']};dips={k:[] for k in matrices}
 eval_geom=FixedSurfaceGeometry(atoms.positions,points)
 for mode in range(len(sources)):
  field=(Bsrc.T[:,mode]).reshape(len(atoms),8)
  comp=response.field_jvp_components(atoms,zero,field)
  uniform=chart.polar_uniform_tangent(field);original=comp.radial_residual_source4+uniform
  for name,source4 in [('original',original),('radial_residual',comp.radial_residual_source4)]:
   mep=coupling.apply_source(eval_geom,embed_mace_polar_learned_source(source4))/Hartree
   matrices[name].append(mep);dips[name].append(molecular_dipole_response_from_density_coefficients(atoms.positions,source4)/Bohr)
  adt=chart.adt_lift.tangent_potential(points_bohr=points,centers_bohr=atoms.positions/Bohr,atomic_dipoles_ebohr=comp.adt_atomic_dipoles_eangstrom/Bohr)
  matrices['adt'].append(adt);dips['adt'].append(np.sum(comp.adt_atomic_dipoles_eangstrom,axis=0)/Bohr)
  matrices['total'].append(matrices['radial_residual'][-1]+adt);dips['total'].append(dips['radial_residual'][-1]+dips['adt'][-1])
 rec={'compound_id':cid,'name':old['name'],'atom_count':len(atoms),'surface_rebuild_max_abs_bohr':float(np.max(abs(surf.surface_points_bohr-points))),'wall_seconds':time.time()-t,'branches':{}}
 for name in matrices:
  mat=np.asarray(matrices[name]).T;dip=np.asarray(dips[name]);rec['branches'][name]={'mep':weighted_response_matrix_discrepancy(mat,qm_mep,surf.quadrature_weights),'dipole_relative_frobenius':float(np.linalg.norm(dip-qm_dip)/np.linalg.norm(qm_dip))}
 records.append(rec);print(json.dumps({'id':cid,'branches':rec['branches']},sort_keys=True),flush=True)
agg={}
for name in ['original','radial_residual','adt','total']:
 vals=[r['branches'][name]['mep']['weighted_relative_frobenius'] for r in records];d=[r['branches'][name]['dipole_relative_frobenius'] for r in records]
 agg[name]={'mep_mean':float(np.mean(vals)),'mep_max':float(np.max(vals)),'mep_pass_count_le_0p2':sum(v<=.2 for v in vals),'dipole_mean':float(np.mean(d)),'dipole_max':float(np.max(d))}
payload={'artifact':'route2-current-radial-adt-freesolv12-nonuniform-replay-unbound-v1','claim':'reuses frozen no-solvation-target QM response; current profile replay, not preregistered admission','records':records,'aggregate':agg}
out=Path('/home/axie/.cache/maple-route2-hybrid-runs/current-radial-adt-freesolv12-nonuniform-replay-20260824.json');out.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n');print(json.dumps(agg,indent=2,sort_keys=True))
