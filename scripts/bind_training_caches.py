#!/usr/bin/env python3
"""Bind freshly encoded caches to the frozen sample order and physical conditions."""
import argparse,csv,json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--encoded',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--split',choices=['train','validation'],required=True)
    a=p.parse_args();rows=list(csv.DictReader(a.manifest.open()));cache={}
    for root in a.encoded:
        for f in root.rglob('*.pth.provenance.json'):
            record=json.loads(f.read_text());key=record['clip_id'];path=Path(str(f).removesuffix('.provenance.json')).resolve()
            if key in cache:raise ValueError(f'Duplicate encoded clip ID: {key}')
            if not path.is_file():raise FileNotFoundError(path)
            cache[key]=path
    for r in rows:
        if r['clip_id'] not in cache:raise ValueError('Missing encoded clip: '+r['clip_id'])
        condition=(a.data_root/r['condition_path']).resolve()
        if not condition.is_file():raise FileNotFoundError(condition)
    a.output.mkdir(parents=True,exist_ok=False);view=a.output/'cache'/a.split;view.mkdir(parents=True)
    output=[]
    for i,r in enumerate(rows):
        r=dict(r);name=f'{a.split}_{i:06d}.pth';target=view/name;target.symlink_to(cache[r['clip_id']])
        r.update(cache_file=name,cache_source=str(cache[r['clip_id']]),override_path=str((a.data_root/r['condition_path']).resolve()),condition_path=str((a.data_root/r['condition_path']).resolve()),prompt_context_path='')
        if r.get('video'):r['video']=str((a.data_root/r['video']).resolve())
        output.append(r)
    with (a.output/f'{a.split}.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(output[0]));writer.writeheader();writer.writerows(output)
    print(json.dumps({'rows':len(output),'cache':str(view.resolve()),'manifest':str((a.output/f'{a.split}.csv').resolve())}))
if __name__=='__main__':main()
