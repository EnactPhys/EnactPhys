#!/usr/bin/env python3
"""Download, extract and validate the raw training-data release."""
import argparse,csv,json,tarfile
from pathlib import Path

def extract(archive, destination):
    with tarfile.open(archive) as tar:
        for member in tar:
            target=(destination/member.name).resolve()
            if not target.is_relative_to(destination.resolve()) or not member.isfile():
                raise ValueError(f'Unsupported archive member: {member.name}')
            if target.exists():
                if target.stat().st_size!=member.size:raise ValueError(f'Existing file size differs: {target}')
                continue
            target.parent.mkdir(parents=True,exist_ok=True)
            temporary=target.with_name(target.name+'.partial')
            with tar.extractfile(member) as src,temporary.open('wb') as dst:
                import shutil
                shutil.copyfileobj(src,dst)
            temporary.replace(target)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--download-dir',default='data/PhysDelta')
    p.add_argument('--output',default='data/training_raw')
    p.add_argument('--download',action='store_true')
    a=p.parse_args();download=Path(a.download_dir).resolve();out=Path(a.output).resolve()
    if a.download:
        from huggingface_hub import snapshot_download
        snapshot_download('EnactPhys/PhysDelta',repo_type='dataset',allow_patterns=['training_raw/*'],local_dir=download)
    source=download/'training_raw';index=json.loads((source/'index.json').read_text())
    incomplete=[s['file'] for s in index['shards'] if not s['uploaded'] or not (download/s['file']).is_file()]
    if incomplete:raise RuntimeError(f'{len(incomplete)} shards are not available yet; first: {incomplete[0]}')
    out.mkdir(parents=True,exist_ok=True)
    for shard in index['shards']:extract(download/shard['file'],out)
    report={}
    for split in ('train','validation'):
        rows=list(csv.DictReader((source/'manifests'/f'{split}.csv').open()))
        missing=[]
        for row in rows:
            for field in ('video','condition_path'):
                value=row.get(field,'')
                if not value or not (out/value).is_file():missing.append({'clip_id':row['clip_id'],'field':field})
        with (out/f'{split}.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        report[split]={'rows':len(rows),'missing_inputs':len(missing),'first_missing':missing[:5]}
    (out/'coverage.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    if any(v['missing_inputs'] for v in report.values()):raise SystemExit('Raw data coverage is incomplete; see coverage.json. Do not start a full-data training run.')
if __name__=='__main__':main()
