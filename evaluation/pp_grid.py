#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',required=True,help='JSONL; each row needs id and video_path')
    p.add_argument('--out-dir',required=True)
    p.add_argument('--overwrite',action='store_true')
    return p.parse_args()


def main():
    import cv2
    from PIL import Image, ImageDraw, ImageFont
    a=parse_args(); out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    rows=[json.loads(x) for x in Path(a.manifest).read_text().splitlines() if x.strip()]
    font=ImageFont.load_default(); made=skipped=failed=0
    for r in rows:
        dst=out/f"{r['id']}.jpg"
        if dst.exists() and not a.overwrite: skipped+=1; continue
        cap=cv2.VideoCapture(r['video_path']); n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n < 1: cap.release(); failed+=1; continue
        frame_ids=[round(k*(n-1)/15) for k in range(16)]
        canvas=Image.new('RGB',(1280,800),(12,14,18)); draw=ImageDraw.Draw(canvas); ok_all=True
        for j,idx in enumerate(frame_ids):
            cap.set(cv2.CAP_PROP_POS_FRAMES,idx); ok,fr=cap.read()
            if not ok: ok_all=False; break
            fr=cv2.cvtColor(fr,cv2.COLOR_BGR2RGB); im=Image.fromarray(fr); im.thumbnail((320,180))
            tile=Image.new('RGB',(320,180),(0,0,0)); tile.paste(im,((320-im.width)//2,(180-im.height)//2))
            x=(j%4)*320; y=(j//4)*200
            canvas.paste(tile,(x,y+20)); draw.text((x+5,y+3),f't{j+1:02d} / frame {idx}',fill='white',font=font)
        cap.release()
        if not ok_all: failed+=1; continue
        canvas.save(dst,quality=94,subsampling=0); made+=1
    print(json.dumps({'n':len(rows),'made':made,'skipped':skipped,'failed':failed,'out_dir':str(out)},ensure_ascii=False))
    if failed:
        raise SystemExit(f'failed to build {failed}/{len(rows)} grid16 inputs')

if __name__=='__main__': main()
