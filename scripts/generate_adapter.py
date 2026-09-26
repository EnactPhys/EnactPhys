#!/usr/bin/env python3
"""Matched ID-only video generation for H2+m and text-only LoRA."""
import argparse
import csv
import json
import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image
import torch
from safetensors.torch import load_file
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig


FORWARD_KEYS = ("first_frame_masks", "object_valid_mask", "force", "force_present", "force_schedule",
                "gravity", "gravity_present", "mu", "mu_present", "restitution", "restitution_present",
                "platform_mask", "edge_mu", "edge_mu_present", "edge_restitution", "edge_restitution_present",
                "mass", "mass_present")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mass-encoder", choices=("linear", "mlp"), default="linear")
    p.add_argument("--cfg-scale", type=float, default=1.0)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--architecture", choices=("h2_mass", "textonly_lora"), required=True)
    p.add_argument("--model-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--shard-index", type=int, required=True)
    p.add_argument("--shard-count", type=int, required=True)
    args = p.parse_args()
    assert 0 <= args.shard_index < args.shard_count
    complete = json.loads((args.checkpoint / "complete.json").read_text())
    assert complete["status"] == "complete"
    with args.manifest.open() as f:
        rows = list(csv.DictReader(f))[args.shard_index::args.shard_count]
    if not rows:
        return
    assert all(r["split"] == "PhysDelta2010_subset" for r in rows)
    args.output_root.mkdir(parents=True, exist_ok=True)
    h2 = args.architecture == "h2_mass"
    os.environ.update(PHYSICAL_WM_OBJECT_MASS="1", PHYSICAL_WM_MASS_ENCODER=args.mass_encoder, PHYSICAL_WM_THREE_CONTACT_GATES="1",
                      PHYSICAL_WM_WRITER_SUPPORT_MASS="1.0", PHYSICAL_WM_TEXT_ONLY="0" if h2 else "1",
                      PHYSICAL_WM_CORRECTABLE_ROUTE_CARRY="1", PHYSICAL_WM_ROUTE_PRIOR_ALPHA="1.0",
                      PHYSICAL_WM_HARD_SUPPORT="0", PHYSICAL_WM_POSITION_DEPENDENT_WRITER_VALUE="0",
                      PHYSICAL_WM_RELATIVE_POSITION_WRITER_VALUE="0", PHYSICAL_WM_SHARED_CONTACT_GATE="0",
                      PHYSICAL_WM_PAIR_EVENT_SUPERVISION="0", PHYSICAL_WM_CONTACT_SUPERVISION="0",
                      PHYSICAL_WM_DISABLE_GRAVITY="0", PHYSICAL_WM_WRITE_GATE_MODE="legacy",
                      PHYSICAL_WM_CAUSAL_OBJECT_TEMPORAL="0", PHYSICAL_WM_EVENT_AFTER_EFFECT="0")
    model = args.model_root
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16, device="cuda",
        model_configs=[ModelConfig(path=[str(model / f"diffusion_pytorch_model-{i:05d}-of-00003.safetensors") for i in (1, 2, 3)]),
                       ModelConfig(path=str(model / "models_t5_umt5-xxl-enc-bf16.pth")),
                       ModelConfig(path=str(model / "Wan2.2_VAE.pth"))],
        tokenizer_config=ModelConfig(path=str(model / "google/umt5-xxl")), redirect_common_files=False,
        enable_sparse_object_interaction_adapter=h2,
        sparse_object_adapter_architecture="temporal_independent_edge_schedule_decoupled_writer_continuous",
        sparse_object_injection_blocks=tuple(range(10, 18)),
    )
    state = load_file(str(args.checkpoint / "trainable_model.safetensors"))
    if h2:
        prefix = "pipe.dit.sparse_object_adapter."
        assert state and all(k.startswith(prefix) for k in state)
        state = {k[len(prefix):]: v for k, v in state.items()}
        incompatible = pipe.dit.sparse_object_adapter.load_state_dict(state, strict=False)
        permitted = {f"groups.{i}.interaction.{q}.{s}" for i in range(10, 18)
                     for q in ("query_projection", "key_projection") for s in ("weight", "bias")}
        assert set(incompatible.missing_keys) <= permitted and not incompatible.unexpected_keys
    else:
        prefix = "pipe.dit."
        assert state and all(k.startswith(prefix) for k in state)
        state = {k[len(prefix):]: v for k, v in state.items()}
        converted = pipe.lora_loader(torch_dtype=torch.float32, device="cpu").convert_state_dict(state)
        targets = ("q", "k", "v", "o", "ffn.0", "ffn.2")
        expected = {name for name, module in pipe.dit.named_modules()
                    if isinstance(module, torch.nn.Linear) and any(name.endswith("." + target) for target in targets)}
        aa = {key[:-len(".lora_A.weight")]: value for key, value in converted.items() if key.endswith(".lora_A.weight")}
        bb = {key[:-len(".lora_B.weight")]: value for key, value in converted.items() if key.endswith(".lora_B.weight")}
        assert expected and set(aa) == expected == set(bb)
        assert all(aa[k].shape[0] == 64 and bb[k].shape[1] == 64 for k in expected)
        pipe.load_lora(pipe.dit, state_dict=state, alpha=1.0)
    for row in rows:
        task = row["task_id"]
        if not task or Path(task).name != task:
            raise ValueError("invalid task id")
        video_path = args.output_root / f"{task}.mp4"
        record_path = args.output_root / f"{task}.json"
        effective_cfg = float(row.get("cfg_scale") or args.cfg_scale)
        identity = dict(checkpoint=str(args.checkpoint), architecture=args.architecture, mass_encoder=args.mass_encoder, cfg_scale=effective_cfg, row=row)
        if record_path.exists():
            assert json.loads(record_path.read_text())["identity"] == identity and video_path.is_file()
            continue
        if video_path.exists():
            raise FileExistsError(video_path)
        source = Path(row["video"])
        if source.suffix.lower() in (".png", ".jpg", ".jpeg"):
            image = Image.open(source).convert("RGB")
        else:
            reader = imageio.get_reader(str(source))
            image = Image.fromarray(reader.get_data(0)).convert("RGB")
            reader.close()
        scale = min(768 / image.width, 448 / image.height)
        size = (round(image.width * scale), round(image.height * scale))
        resized = image.resize(size, Image.Resampling.LANCZOS)
        image = Image.new("RGB", (768, 448))
        image.paste(resized, ((768 - size[0]) // 2, (448 - size[1]) // 2))
        condition = None
        if h2:
            raw = torch.load(row["condition_path"], map_location="cpu", weights_only=False)
            condition = {k: raw[k].to("cuda") for k in FORWARD_KEYS}
            condition["contract"] = raw["contract"]
        frames = pipe(prompt=row["h2_prompt"] if h2 else row["text_prompt"], negative_prompt="",
                      input_image=image, sparse_object_condition=condition, seed=int(row["seed"]), rand_device="cpu",
                      height=448, width=768, num_frames=49, cfg_scale=effective_cfg, num_inference_steps=int(row.get("num_inference_steps") or 30),
                      sigma_shift=float(row.get("sigma_shift") or 5.0), tiled=True, tile_size=(30, 52), tile_stride=(15, 26))
        temporary = args.output_root / f"{task}.incomplete.mp4"
        with imageio.get_writer(str(temporary), fps=int(row.get("fps",16)), codec="libx264", pixelformat="yuv420p", quality=7) as writer:
            for frame in frames:
                writer.append_data(np.asarray(frame))
        assert len(frames) == 49
        temporary.rename(video_path)
        record_path.write_text(json.dumps(dict(status="complete", identity=identity), ensure_ascii=False))
        print(json.dumps(dict(task_id=task, status="complete")), flush=True)


if __name__ == "__main__":
    main()
