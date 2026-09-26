#!/usr/bin/env python3
"""Generate and score the fixed Physics-IQ Solid Mechanics task set."""
import argparse,os,subprocess,sys,json,csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['assets-root','base-model','checkpoint','real-folder','real-masks','output-dir']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--seed',type=int,choices=[43278311,56382197,68491523]);p.add_argument('--dry-run',action='store_true')
    a=p.parse_args();out=a.output_dir.resolve();commands=[]
    rows=list(csv.DictReader((ROOT/'physicsiq/data/generation_manifest.csv').open()))
    rows=[r for r in rows if a.seed is None or int(r['seed'])==a.seed]
    for r in rows:
        for key in ['video','condition_path']:
            if r.get(key) and not (a.assets_root/r[key]).is_file():raise FileNotFoundError(a.assets_root/r[key])
    for directory in [a.base_model,a.checkpoint,a.real_folder,a.real_masks]:
        if not directory.is_dir():raise FileNotFoundError(directory)
    from evaluate_physicsiq import build_jobs
    build_jobs(out,a.real_folder.resolve(),a.real_masks.resolve(),out/'scores',a.seed,require_videos=False)
    for f in [*[a.base_model/f'diffusion_pytorch_model-{i:05d}-of-00003.safetensors' for i in (1,2,3)],a.base_model/'Wan2.2_VAE.pth',a.base_model/'models_t5_umt5-xxl-enc-bf16.pth',a.checkpoint/'trainable_model.safetensors']:
        if not f.is_file():raise FileNotFoundError(f)
    for route in ['adapter','base']:
        cmd=[sys.executable,str(ROOT/'scripts/generate.py'),'--route',route,'--assets-root',str(a.assets_root.resolve()),'--base-model',str(a.base_model.resolve()),'--output-dir',str(out/route)]
        if route=='adapter':cmd+=['--checkpoint',str(a.checkpoint.resolve())]
        if a.seed is not None:cmd+=['--seed',str(a.seed)]
        commands.append(cmd)
    cmd=[sys.executable,str(ROOT/'scripts/evaluate_physicsiq.py'),'--videos',str(out),'--real-folder',str(a.real_folder.resolve()),'--real-masks',str(a.real_masks.resolve()),'--output-dir',str(out/'scores')]
    if a.seed is not None:cmd+=['--seed',str(a.seed)]
    commands.append(cmd)
    if a.dry_run:print(json.dumps({'validated_generation_inputs':len(rows),'commands':commands},indent=2));return
    out.mkdir(parents=True,exist_ok=False)
    for cmd in commands:subprocess.run(cmd,check=True,cwd=ROOT)
if __name__=='__main__':main()
