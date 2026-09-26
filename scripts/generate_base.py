#!/usr/bin/env python3
"""Generate a flat task bank with the unmodified Wan base model."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image
import torch
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    args = parser.parse_args()
    with args.manifest.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))[args.shard_index::args.shard_count]
    if not rows:
        return
    if any(row["model_route"] != "WAN_BASE" or row.get("condition_path") for row in rows):
        raise RuntimeError("Wan route must be adapter-off and conditions-off")
    model = args.model_root
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16, device="cuda",
        model_configs=[ModelConfig(path=[str(model / f"diffusion_pytorch_model-{i:05d}-of-00003.safetensors") for i in (1, 2, 3)]),
                       ModelConfig(path=str(model / "models_t5_umt5-xxl-enc-bf16.pth")),
                       ModelConfig(path=str(model / "Wan2.2_VAE.pth"))],
        tokenizer_config=ModelConfig(path=str(model / "google/umt5-xxl")), redirect_common_files=False,
        enable_sparse_object_interaction_adapter=False,
    )
    if getattr(pipe.dit, "sparse_object_adapter", None) is not None:
        raise RuntimeError("adapter unexpectedly present")
    args.output_root.mkdir(parents=True, exist_ok=True)
    for row in rows:
        task = row["task_id"]
        video_path = args.output_root / f"{task}.mp4"
        record_path = args.output_root / f"{task}.json"
        identity = {"route": "WAN_BASE", "row": row}
        if record_path.is_file():
            if json.loads(record_path.read_text())["identity"] != identity or not video_path.is_file():
                raise RuntimeError(f"identity mismatch: {task}")
            continue
        source = Path(row.get("video") or row["source_video"])
        if source.suffix.lower() in (".png", ".jpg", ".jpeg"):
            image = Image.open(source).convert("RGB")
        else:
            reader = imageio.get_reader(str(source)); image = Image.fromarray(reader.get_data(0)).convert("RGB"); reader.close()
        scale = min(768 / image.width, 448 / image.height)
        size = (round(image.width * scale), round(image.height * scale))
        resized = image.resize(size, Image.Resampling.LANCZOS)
        image = Image.new("RGB", (768, 448)); image.paste(resized, ((768-size[0])//2, (448-size[1])//2))
        frames = pipe(prompt=row["prompt"], negative_prompt=row.get("negative_prompt", ""), input_image=image,
                      sparse_object_condition=None, seed=int(row["seed"]), rand_device="cpu",
                      height=448, width=768, num_frames=49, cfg_scale=float(row.get("cfg_scale") or 1.0),
                      num_inference_steps=int(row.get("num_inference_steps") or 30), sigma_shift=float(row.get("sigma_shift") or 5.0),
                      tiled=True, tile_size=(30, 52), tile_stride=(15, 26))
        temporary = args.output_root / f"{task}.incomplete.mp4"
        with imageio.get_writer(str(temporary), fps=int(row.get("fps") or 24), codec="libx264", pixelformat="yuv420p", quality=7) as writer:
            for frame in frames:
                writer.append_data(np.asarray(frame))
        temporary.rename(video_path)
        record_path.write_text(json.dumps({"status": "complete", "identity": identity}, ensure_ascii=False))
        print(json.dumps({"task_id": task, "status": "complete"}), flush=True)


if __name__ == "__main__":
    main()
