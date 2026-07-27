from __future__ import annotations
import hashlib, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch
ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "maple").is_dir() and (parent / "pyproject.toml").is_file()
)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from maple.function.calculator.set_calculator import SetCalculator
from maple.function.calculator.extra_correction.implicit.correction import ImplicitSolvationCorrection
from maple.function.read.command_control import CommandControl
from maple.function.read.filereader.mol2_reader import MOL2Reader
from maple.function.route2_smd_profiles import PCMSOLVER_INTRINSIC_CAVITY_PROFILE
OUT = Path(os.environ['OUT_DIR']).resolve(); BASE=ROOT/'.omx/benchmarks/route2-macepolar-smd'; SEL=Path(__file__).with_name('selection.json')
selection=json.loads(SEL.read_text()); prepared=json.loads((BASE/'prepared.json').read_text()); byid={x['compound_id']:x for x in prepared['candidates']}; device='cuda' if torch.cuda.is_available() else 'cpu'
def params(): return CommandControl.from_settings(['#model=macepol-m','#sp','#solv(implicit=water,method=smd,provider=pcmsolver,'+f'profile={PCMSOLVER_INTRINSIC_CAVITY_PROFILE},response=frozen,standard_state=1m,experimental=true)']).as_dict()
def sync():
    if device=='cuda': torch.cuda.synchronize()
first=byid[selection['records'][0]['compound_id']]; atoms=MOL2Reader(str(BASE/first['mol2_relative_path']),charge=0,mult=1); p=params(); t=time.perf_counter(); calc=SetCalculator(device,p['model'],str(OUT/'model.out'),atoms=atoms,implicit='smd',solvent='water',model_options=p.get('model_options'),solvation_options=p['solv'],charge_options={}).set_calculator(); sync(); load=time.perf_counter()-t
records=[]; panel=time.perf_counter()
for index,item in enumerate(selection['records']):
 c=byid[item['compound_id']]; atoms=MOL2Reader(str(BASE/c['mol2_relative_path']),charge=0,mult=1); target=OUT/c['compound_id']/'maple.out'; target.parent.mkdir(parents=True)
 calc.solvent_correction=ImplicitSolvationCorrection(atoms,{},p['solv'],output=str(target)); calc.reset(); atoms.calc=calc; sync(); t=time.perf_counter(); atoms.get_potential_energy(); sync(); elapsed=time.perf_counter()-t
 s=calc.results['solvation']; pred=float(s['delta_g_solv_hartree'])*627.5094740631; exp=float(c['experimental_kcal_mol']); rec={'selection_index':index,'compound_id':c['compound_id'],'name':c['name'],'class':item['class'],'experimental_kcal_mol':exp,'predicted_kcal_mol':pred,'signed_error_kcal_mol':pred-exp,'absolute_error_kcal_mol':abs(pred-exp),'wall_seconds':elapsed,'components_kcal_mol':{k:float(v)*627.5094740631 for k,v in s['components_hartree'].items()}}; records.append(rec); print(f"{index+1:02d}/10 {c['name']:<15} fixed_l1 pred={pred:8.3f} exp={exp:7.3f} abs={abs(pred-exp):6.3f} t={elapsed:5.2f}s",flush=True)
errors=np.array([r['signed_error_kcal_mol'] for r in records]); times=np.array([r['wall_seconds'] for r in records]); metrics={'record_count':10,'mean_absolute_error_kcal_mol':float(np.mean(np.abs(errors))),'root_mean_square_error_kcal_mol':float(np.sqrt(np.mean(errors**2))),'mean_signed_error_kcal_mol':float(np.mean(errors)),'maximum_absolute_error_kcal_mol':float(np.max(np.abs(errors))),'mean_wall_seconds':float(np.mean(times)),'total_wall_seconds':float(np.sum(times))}
diff=subprocess.check_output(['git','diff','--binary'],cwd=ROOT); payload={'artifact':'route2-intrinsic-fixed-l1-ten-panel-v1','git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'git_diff_sha256':hashlib.sha256(diff).hexdigest(),'selection_sha256':hashlib.sha256(SEL.read_bytes()).hexdigest(),'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'device':device,'model_load_seconds':load,'actual_panel_wall_seconds':time.perf_counter()-panel,'records':records,'aggregate':metrics,'claim_boundary':selection['claim_boundary']}
(OUT/'results.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n'); print(json.dumps(metrics,indent=2)); print('RESULT_PATH='+str(OUT/'results.json'))
