import json,time
from pathlib import Path
import numpy as np
from pyscf import lib,df,gto,dft
root=Path('/home/axie/MAPLE/MAPLE-implicitsolv-route2/.omx/benchmarks/route2-gto-pcm-energy-projection-four-v1')
checkpoints={
'mobley_3053621':root/'mobley_3053621/qm-gas/gas.chk',
'mobley_3867265':Path('/home/axie/MAPLE/MAPLE-implicitsolv-route2/.omx/benchmarks/route2-qm-ddx-acetone-center-20260725/pyscf_smd_acetone_wb97mv_def2tzvpd_g3_n50x194sg1.gas.chk'),
'mobley_3034976':root/'mobley_3034976/qm-gas/gas.chk',
'mobley_352111':Path('/home/axie/MAPLE/MAPLE-implicitsolv-route2/.omx/benchmarks/route2-flexible-multihetero-20260725/pyscf_smd_2acetoxyethyl_acetate_wb97mv_def2tzvpd_g3_n50x194sg1.gas.chk'),
}
records=[]
for cid,chk in checkpoints.items():
 t=time.time();panel=root/cid/'cutoff-1e-10/work';mol=lib.chkfile.load_mol(str(chk));C=np.asarray(lib.chkfile.load(str(chk),'scf/mo_coeff'));o=np.asarray(lib.chkfile.load(str(chk),'scf/mo_occ'));D=np.einsum('pi,i,qi->pq',C,o,C,optimize=True)
 with np.load(panel/'surface.npz') as z:pts=np.asarray(z['surface_points_bohr'],float)
 with np.load(panel/'qm-surface-mep.npz') as z:ref=np.asarray(z['surface_potential_hartree_per_e'],float)
 with np.load(panel/'cavity.npz') as z:weights=np.asarray(z['weights'],float).reshape(-1)
 if weights.shape!=ref.shape: raise RuntimeError((cid,weights.shape,ref.shape))
 auxbasis=df.make_auxbasis(mol);aux=df.addons.make_auxmol(mol,auxbasis=auxbasis);J=aux.intor('int2c2e');w,v=np.linalg.eigh(J);keep=w>w.max()*1e-12;Jinv=v[:,keep]@((v[:,keep].T)/w[keep,None]);ints3=df.incore.aux_e2(mol,aux,intor='int3c2e',aosym='s1');rhs=np.einsum('ijp,ij->p',ints3,D,optimize=True);coef=Jinv@rhs
 G=dft.gen_grid.Grids(aux);G.level=4;G.build(with_non0tab=False);m0=np.zeros(aux.nao_nr());m1=np.zeros((3,aux.nao_nr()))
 for i0 in range(0,len(G.coords),30000):
  c=G.coords[i0:i0+30000];wt=G.weights[i0:i0+30000];ao=dft.numint.eval_ao(aux,c);m0+=np.einsum('g,gp->p',wt,ao);m1+=np.einsum('g,gx,gp->xp',wt,c,ao)
 ne=float(np.einsum('ij,ji',D,mol.intor_symmetric('int1e_ovlp')));em=np.einsum('xij,ji->x',mol.intor_symmetric('int1e_r',comp=3),D);A=np.vstack([m0,m1]);target=np.r_[ne,em];res=target-A@coef;M=A@Jinv@A.T;coefc=coef+Jinv@A.T@np.linalg.solve(M,res)
 cross=gto.mole.intor_cross('int2c2e',aux,gto.fakemol_for_charges(pts));nuc=sum(z/np.linalg.norm(pts-r,axis=1) for z,r in zip(mol.atom_charges(),mol.atom_coords()))
 rec={'case_id':cid,'atom_count':mol.natm,'naux':aux.nao_nr(),'metric_rank':int(keep.sum()),'metric_condition':float(w[keep].max()/w[keep].min()),'moment_raw_max_error':float(np.max(abs(A@coef-target))),'moment_constrained_max_error':float(np.max(abs(A@coefc-target))),'wall_seconds':time.time()-t}
 for label,x in [('raw',coef),('constrained',coefc)]:
  pred=nuc-x@cross;err=pred-ref;rec[label]={'area_relative_error':float(np.sqrt(np.sum(weights*err**2)/np.sum(weights*ref**2))),'correlation':float(np.corrcoef(pred,ref)[0,1]),'max_abs_hartree_per_e':float(np.max(abs(err)))}
 records.append(rec);print(json.dumps(rec,sort_keys=True),flush=True)
payload={'artifact':'route2-four-case-make-auxbasis-density-representation-oracle-unbound-v1','claim':'local no-solvation-target no-fit representation probe; not preregistered admission evidence','basis':'pyscf.df.make_auxbasis on frozen orbital basis','metric_eigenvalue_relative_cutoff':1e-12,'moment_grid_level':4,'records':records,'aggregate':{'mean_raw_area_relative_error':float(np.mean([r['raw']['area_relative_error'] for r in records])),'mean_constrained_area_relative_error':float(np.mean([r['constrained']['area_relative_error'] for r in records])),'maximum_constrained_area_relative_error':float(np.max([r['constrained']['area_relative_error'] for r in records]))}}
out=Path('/home/axie/.cache/maple-route2-hybrid-runs/four-case-make-auxbasis-oracle-20260824.json');out.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n');print(json.dumps(payload['aggregate'],indent=2,sort_keys=True))
