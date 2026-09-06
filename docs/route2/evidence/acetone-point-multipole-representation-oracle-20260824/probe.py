import json, itertools
from pathlib import Path
import numpy as np
from scipy.linalg import null_space
from ase.units import Bohr
from pyscf import lib
from maple.solvation.release.cartesian_multipole_mep import cartesian_atomic_multipole_potential

P=Path('/home/axie/.cache/maple-route2-hybrid-runs/qm-boundary-response-v3-20260823/preregistration.json')
pre=json.load(open(P)); files=pre['inputs']['case']['files']
mol=lib.chkfile.load_mol(files['qm_checkpoint']['path'])
Cmo=np.asarray(lib.chkfile.load(files['qm_checkpoint']['path'],'scf/mo_coeff'))
occ=np.asarray(lib.chkfile.load(files['qm_checkpoint']['path'],'scf/mo_occ'))
D=np.einsum('pi,i,qi->pq',Cmo,occ,Cmo,optimize=True)
with np.load('/home/axie/.cache/maple-route2-hybrid-runs/qm-boundary-response-v3-20260823/result/arrays.npz') as z:
 p0=np.asarray(z['surface_points_bohr'],float); a0=np.asarray(z['surface_areas_bohr2'],float)
centers_bohr=np.asarray(mol.atom_coords(),float); centers_A=centers_bohr*Bohr
origin=centers_bohr.mean(axis=0)
scales=(0.95,1.0,1.05)
point_sets=[origin+s*(p0-origin) for s in scales]
areas=[a0*s*s for s in scales]

def qm_mep(points):
 ints=np.asarray(mol.intor('int1e_grids',hermi=1,grids=points),float)
 electronic=np.einsum('xij,ji->x',ints,D,optimize=True)
 nuc=sum(z/np.linalg.norm(points-r,axis=1) for z,r in zip(mol.atom_charges(),centers_bohr))
 return nuc-electronic
refs=[qm_mep(p) for p in point_sets]

# Orthonormal symmetric tensor basis arrays.
q=[]; p=[]; Q=[]; O=[]; labels=[]
N=mol.natm
zeros=lambda: (np.zeros(N),np.zeros((N,3)),np.zeros((N,3,3)),np.zeros((N,3,3,3)))
for ia in range(N):
 z=zeros();z[0][ia]=1;q.append(z);labels.append((ia,0,'q'))
 for i in range(3):
  z=zeros();z[1][ia,i]=1;p.append(z);labels.append((ia,1,(i,)))
 for i in range(3):
  for j in range(i,3):
   z=zeros();perms=set(itertools.permutations((i,j)));v=1/np.sqrt(len(perms))
   for u in perms:z[2][(ia,)+u]=v
   Q.append(z);labels.append((ia,2,(i,j)))
 for i in range(3):
  for j in range(i,3):
   for k in range(j,3):
    z=zeros();perms=set(itertools.permutations((i,j,k)));v=1/np.sqrt(len(perms))
    for u in perms:z[3][(ia,)+u]=v
    O.append(z);labels.append((ia,3,(i,j,k)))
basis=q+p+Q+O
allpts=np.concatenate(point_sets)
def column(z):
 return cartesian_atomic_multipole_potential(points_bohr=allpts,centers_angstrom=centers_A,charges_e=z[0],dipoles_eangstrom=z[1],quadrupoles_eangstrom2=z[2],octupoles_eangstrom3=z[3])
A=np.column_stack([column(z) for z in basis])
# Exact molecular Q/mu target in same origin as coordinates.
rint=np.asarray(mol.intor_symmetric('int1e_r',comp=3),float)
mu_nuc=np.einsum('i,ix->x',mol.atom_charges(),centers_bohr)*Bohr
mu_ele=-np.einsum('xij,ji->x',rint,D)*Bohr
mu=mu_nuc+mu_ele
b=np.r_[float(mol.charge),mu]
# Constraint columns by directly measuring q and dipole of each basis.
C=np.zeros((4,len(basis)))
for k,z in enumerate(basis):
 C[0,k]=z[0].sum()
 C[1:,k]=np.sum(z[0][:,None]*centers_A+z[1],axis=0)

def solve(ncol,fit_shells=(0,1,2)):
 AA=A[:,:ncol]; CC=C[:,:ncol]
 x0=CC.T@np.linalg.solve(CC@CC.T,b)
 Nul=null_space(CC,rcond=1e-12)
 idx=np.concatenate([np.arange(si*len(p0),(si+1)*len(p0)) for si in fit_shells])
 w=np.concatenate([areas[si] for si in fit_shells]);w=w/w.mean()
 X=np.sqrt(w)[:,None]*(AA[idx]@Nul); y=np.sqrt(w)*(np.concatenate([refs[si] for si in fit_shells])-AA[idx]@x0)
 z=np.linalg.lstsq(X,y,rcond=1e-12)[0];coef=x0+Nul@z
 pred=AA@coef
 out=[]
 off=0
 for s,ref,area in zip(scales,refs,areas):
  pp=pred[off:off+len(ref)];off+=len(ref)
  err=pp-ref
  out.append({'scale':s,'area_relative_error':float(np.sqrt(np.sum(area*err**2)/np.sum(area*ref**2))),'correlation':float(np.corrcoef(pp,ref)[0,1]),'max_abs_hartree_per_e':float(np.max(abs(err)))})
 return {'coefficient_count':ncol,'constraint_error':float(np.max(abs(CC@coef-b))),'coefficient_norm':float(np.linalg.norm(coef)),'shells':out}
res={
 'artifact':'acetone-point-multipole-representation-oracle-unbound-v1',
 'claim':'local no-fit representation probe; not preregistered or admission evidence',
 'target_charge_e':float(mol.charge),'target_dipole_eangstrom':mu.tolist(),
 'fit_all':{
  'l1':solve(len(q)+len(p)),
  'l2':solve(len(q)+len(p)+len(Q)),
  'l3':solve(len(basis)),
 },
 'fit_outer_validate_middle':{
  'l1':solve(len(q)+len(p),(0,2)),
  'l2':solve(len(q)+len(p)+len(Q),(0,2)),
  'l3':solve(len(basis),(0,2)),
 }
}
out=Path('/home/axie/.cache/maple-route2-hybrid-runs/acetone-point-l3-representation-oracle-20260824.json')
out.write_text(json.dumps(res,indent=2,sort_keys=True)+'\n')
print(json.dumps(res,indent=2,sort_keys=True))
