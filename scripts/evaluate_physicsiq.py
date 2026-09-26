#!/usr/bin/env python3
"""Score fixed Solid Mechanics videos against the official 24 FPS references."""
import argparse,csv,json,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'physicsiq/vendor')]

def build_jobs(videos,real_folder,real_masks,output,seed=None,require_videos=True):
    descriptions=list(csv.DictReader((ROOT/'physicsiq/data/solid_descriptions.csv').open()))
    takes={};first={}
    for row in descriptions:
        ident,view,take,scenario=row['scenario'].split('_',3)
        takes[(scenario,view,take)]=ident
        if take=='take-1':first[ident]=(view,scenario)
    manifest=list(csv.DictReader((ROOT/'physicsiq/data/generation_manifest.csv').open()))
    jobs=[]
    for row in manifest:
        if seed is not None and int(row['seed'])!=seed:continue
        ident=row['benchmark_id'];view,scenario=first[ident]
        roots=[videos,videos/'adapter/videos',videos/'base/videos']
        matches=[p/(row['task_id']+'.mp4') for p in roots if (p/(row['task_id']+'.mp4')).is_file()]
        if require_videos and len(matches)!=1:raise ValueError(f"Expected one generated video for {row['task_id']}, found {len(matches)}")
        job=dict(task_id=row['task_id'],benchmark_id=ident,search_variant='fixed',parameter_variant='',scenario=scenario,view=view,take1_id=ident,take2_id=takes[(scenario,view,'take-2')],output_video_name=f'{ident}_{view}_{scenario}',expected_mask_name=f'{ident}_video-masks_24FPS_{view}_take-1_{scenario}',generated_video=str(matches[0] if matches else videos/(row['task_id']+'.mp4')),benchmark_code_root=str(ROOT/'physicsiq/vendor'),real_folder=str(real_folder),real_masks=str(real_masks),media_root=str(output/'media'),condition_summary_json=row['condition_summary_json'],prompt_source=row['prompt_source'],generation_seed=int(row['seed']))
        for take in [1,2]:
            rid=job[f'take{take}_id']
            for base,kind in [(real_folder,'testing-videos'),(real_masks,'video-masks')]:
                f=base/f'{rid}_{kind}_24FPS_{view}_take-{take}_{scenario}'
                if not f.is_file():raise FileNotFoundError(f)
        jobs.append(job)
    if len(jobs) not in [114,342]:raise ValueError('Incomplete fixed benchmark task selection')
    return jobs

def aggregate(results):
    import pandas as pd
    from physiq.calculate_iq_score_stable import IQTable
    from physiq.calculate_iq_score import VIEWS
    groups={};seen=set()
    for r in results:
        key=(int(r['generation_seed']),r['scenario'],r['view'])
        if key in seen:raise ValueError('Duplicate scenario/view/seed')
        seen.add(key);groups.setdefault(key[0],{}).setdefault(key[1],{}).update(r['metrics'])
    per_seed=[]
    for seed,scenarios in sorted(groups.items()):
        if len(scenarios)!=38:raise ValueError(f'Expected 38 scenarios for seed {seed}')
        for scenario in scenarios:
            if any((seed,scenario,v) not in seen for v in VIEWS):raise ValueError('Missing benchmark view')
        table=IQTable(pd.DataFrame([dict(scenario=s,**m) for s,m in sorted(scenarios.items())]))
        per_seed.append({'seed':seed,'verified_view':100*float(table.get_output_dict()['final_score_view'])})
    return {'videos':len(results),'per_seed':per_seed,'verified_view':statistics.mean(x['verified_view'] for x in per_seed)}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['videos','real-folder','real-masks','output-dir']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--seed',type=int,choices=[43278311,56382197,68491523]);p.add_argument('--dry-run',action='store_true')
    a=p.parse_args();jobs=build_jobs(a.videos.resolve(),a.real_folder.resolve(),a.real_masks.resolve(),a.output_dir.resolve(),a.seed)
    if a.dry_run:print(json.dumps({'validated_video_and_reference_inputs':len(jobs),'frames':120,'fps':24}));return
    a.output_dir.mkdir(parents=True,exist_ok=False)
    from evaluation.physicsiq import score_worker
    results=[]
    for i,job in enumerate(jobs):
        results.append(score_worker(job));print(f'{i+1}/{len(jobs)}',flush=True)
    summary=aggregate(results);(a.output_dir/'RESULT.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
