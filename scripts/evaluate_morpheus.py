#!/usr/bin/env python3
"""Track and score supplied MORPHEUS videos with the released evaluation settings."""
import argparse,csv,json,os,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
VENDOR=ROOT/'runtime/morpheus'
SEEDS=[937,5318,1888]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True,help='CSV: model,task_id,case,seed,prompt_type,video')
    p.add_argument('--video-root',type=Path,required=True)
    p.add_argument('--sam2-checkpoint',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args();rows=list(csv.DictReader(a.manifest.open()));seen=set()
    if not rows:raise ValueError('Empty manifest')
    for r in rows:
        for key in ['model','task_id','case','prompt_type']:
            if not re.fullmatch(r'[A-Za-z0-9_.-]+',r[key]):raise ValueError('Invalid '+key)
        key=(r['model'],r['task_id'])
        if key in seen:raise ValueError('Duplicate model/task ID')
        seen.add(key)
        if int(r['seed']) not in SEEDS:raise ValueError('Seed absent from fixed protocol')
        r['video']=str((a.video_root/r['video']).resolve())
        if not Path(r['video']).is_file():raise FileNotFoundError(r['video'])
    for f in [a.sam2_checkpoint,VENDOR/'labels.json',VENDOR/'morpheus/scoring/collision_mass_mlp.pt']:
        if not f.is_file():raise FileNotFoundError(f)
    if a.dry_run:print(json.dumps({'validated_inputs':len(rows),'falling_min_frames':15,'pinn_seed':3407}));return
    a.output_dir.mkdir(parents=True,exist_ok=False);temp=a.output_dir/'tmp';temp.mkdir()
    os.environ.update(MORPHEUS_PINN_SEED='3407',MPLBACKEND='Agg',TMPDIR=str(temp.resolve()),TMP=str(temp.resolve()),TEMP=str(temp.resolve()),IMAGEIO_FFMPEG_TEMP_DIR=str(temp.resolve()))
    sys.path[:0]=[str(VENDOR),str(VENDOR/'sam2')]
    import cv2
    from morpheus.resizing.transforms import process_mp4_file
    from morpheus.tracking.sam2_tracker import LazyTracker
    import morpheus.tracking.trajectory_cropping as cropping
    original=cropping.find_crop_frame
    def find(y_coords,experiment_type,min_frames=20,x_coords=None):
        minimum=15 if experiment_type in {'falling_marker','falling_apple','falling_ball','falling_tape'} else min_frames
        return original(y_coords,experiment_type,min_frames=minimum,x_coords=x_coords)
    cropping.find_crop_frame=find
    from morpheus.pipeline import process_all_videos
    from morpheus.cli.track_and_score import build_parser
    cv2.setNumThreads(1)
    tracker=LazyTracker(checkpoint=str(a.sam2_checkpoint.resolve()),config_dir=str(VENDOR/'configs'),model_cfg='sam2.1_hiera_l.yaml',device='cuda')
    results=[]
    for i,r in enumerate(rows):
        work=a.output_dir/(r['model']+'--'+r['task_id']);processed=work/'processed';scores=work/'scores'
        cap=cv2.VideoCapture(r['video']);shape=(int(cap.get(3)),int(cap.get(4)),int(cap.get(7)));fps=cap.get(5);cap.release()
        if shape!=(768,448,49) or abs(fps-24)>1e-3:raise ValueError(f"Expected 768x448, 49 frames, 24 FPS: {r['task_id']}")
        number=SEEDS.index(int(r['seed']));rel=Path('single_frame_conditioning/WAN-2.1')/r['prompt_type']/r['case']/str(number)
        process_mp4_file(r['video'],f"single_frame_conditioning/wan/{r['prompt_type']}/{r['case']}/seed_{r['seed']}",str(processed),number,True)
        if len(list((processed/rel/'frames_for_tracking').glob('*.jpg')))!=49:raise ValueError('Preprocessed frame count mismatch')
        args=build_parser().parse_args(['--input-dir',str(processed),'--output-dir',str(scores),'--calculate-scores','--conditioning','single_frame_conditioning','--methods','WAN-2.1','--experiments',r['case'],'--labels-json',str(VENDOR/'labels.json'),'--device','cuda'])
        args.depth_processor_v2=None
        process_all_videos(tracker,args,conditioning=tuple(args.conditioning),methods=tuple(args.methods),experiments=tuple(args.experiments))
        file=scores/rel/'combined_scores.json'
        if not file.is_file():raise RuntimeError(f"Scoring did not produce measurements for {r['task_id']}")
        result=json.loads(file.read_text())
        if 'statistical_score' not in result:raise ValueError('Missing dynamical score')
        result['evaluation_settings']={'falling_crop_min_frames':15,'pinn_seed':3407}
        file.write_text(json.dumps(result,indent=2))
        results.append({'model':r['model'],'task_id':r['task_id'],'case':r['case'],'seed':int(r['seed']),'raw_scores':result})
        (a.output_dir/'raw_measurements.json').write_text(json.dumps(results,indent=2))
        print(f'{i+1}/{len(rows)}',flush=True)
    print(json.dumps({'scored_videos':len(results),'measurements':str(a.output_dir/'raw_measurements.json')}))
if __name__=='__main__':main()
