#!/usr/bin/env python3
"""Encode raw videos with the released compact Wan preprocessing path."""
import argparse,csv,json,os,subprocess,sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True)
    p.add_argument('--data-root',required=True)
    p.add_argument('--model-root',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--gpu',default='0')
    p.add_argument('--shard-index',type=int,default=0)
    p.add_argument('--shard-count',type=int,default=1)
    p.add_argument('--prompt-column',required=True,choices=['prompt','neutral_prompt','text_prompt'])
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    if not 0<=a.shard_index<a.shard_count:raise ValueError('Invalid shard assignment')
    root=Path(a.data_root).resolve();model=Path(a.model_root).resolve();out=Path(a.output).resolve()
    code=Path(__file__).resolve().parents[1]/'runtime/training'
    rows=list(csv.DictReader(open(a.manifest)))[a.shard_index::a.shard_count]
    if not rows:raise ValueError('Empty shard')
    prepared=[];missing=[]
    for r in rows:
        video=root/r.get('video','')
        if not r.get('video') or not video.is_file():missing.append(r['clip_id']);continue
        prompt=r.get(a.prompt_column,'')
        if not prompt.strip():raise ValueError(f"Empty {a.prompt_column} for {r['clip_id']}")
        prepared.append({'clip_id':r['clip_id'],'video':str(video),'prompt':prompt})
    if missing:raise ValueError(f'{len(missing)} original videos unavailable in this shard; first: {missing[0]}')
    paths=[[str(model/f'diffusion_pytorch_model-{i:05d}-of-00003.safetensors') for i in (1,2,3)],str(model/'models_t5_umt5-xxl-enc-bf16.pth'),str(model/'Wan2.2_VAE.pth')]
    for f in paths[0]+paths[1:]:
        if not Path(f).is_file():raise FileNotFoundError(f)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Use a new preprocessing output directory')
    if a.dry_run:
        print(json.dumps({'rows':len(prepared),'height':448,'width':768,'frames':49,'resize':'pad','prompt_column':a.prompt_column,'gpu_run':False}));return
    out.mkdir(parents=True,exist_ok=True);temp=out/'tmp';temp.mkdir()
    manifest=out/'inputs.csv'
    with manifest.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(prepared[0]));w.writeheader();w.writerows(prepared)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpu,PYTHONPATH=str(code),OMP_NUM_THREADS='2',TMPDIR=str(temp),TMP=str(temp),TEMP=str(temp),IMAGEIO_FFMPEG_TEMP_DIR=str(temp),WANDB_MODE='disabled',ENABLE_WANDB_LOG='0',WANDB_DISABLED='true',PHYSICAL_WM_OBJECT_MASS='0',PHYSICAL_WM_THREE_CONTACT_GATES='0',PHYSICAL_WM_TEXT_ONLY='0')
    cmd=[sys.executable,'-u',str(code/'train.py'),'--task','sft:data_process','--dataset_base_path','/','--dataset_metadata_path',str(manifest),'--data_file_keys','video','--output_path',str(out/'cache'),'--height','448','--width','768','--num_frames','49','--video_resize_mode','pad','--dataset_repeat','1','--dataset_num_workers','0','--model_paths',json.dumps(paths),'--tokenizer_path',str(model/'google/umt5-xxl'),'--seed','42','--extra_inputs','input_image','--compact_data_process_cache']
    subprocess.run(cmd,env=env,check=True)
    records=[]
    for f in (out/'cache').rglob('*.pth.provenance.json'):
        d=json.loads(f.read_text());records.append({'clip_id':d['clip_id'],'cache_source':str(f).removesuffix('.provenance.json')})
    if len(records)!=len(prepared):raise RuntimeError('Cache output count differs from input count')
    (out/'cache_index.json').write_text(json.dumps(records,indent=2))
    print(f'Encoded {len(records)} videos; physical-condition tensors remain in the raw dataset.')
if __name__=='__main__':main()
