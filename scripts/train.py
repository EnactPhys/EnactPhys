#!/usr/bin/env python3
"""Launch one node of the matched 16-GPU mass experiment from a resolved JSON."""
import argparse
import csv
import json
import os
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--node-rank", type=int, choices=(0, 1), required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    c = json.loads(Path(args.config).read_text())
    repository = Path(__file__).resolve().parents[1]
    for key in ("code_root", "output_path", "temp_root", "train_cache", "validation_cache", "train_override", "validation_override", "model_root"):
        value = Path(c[key])
        c[key] = str((repository / value).resolve() if not value.is_absolute() else value.resolve())
    executable = shutil.which(c["python"])
    if executable is None:
        raise FileNotFoundError("Configured Python executable is unavailable")
    c["python"] = executable
    assert c["architecture"] in ("h2_mass", "textonly_lora")
    assert c["world_size"] == 16 and c["microbatch_per_rank"] == 2
    assert c["gradient_accumulation_steps"] == 1
    root = Path(c["code_root"])
    output = Path(c["output_path"])
    temp = Path(c["temp_root"]) / f"n{args.node_rank}"
    for key in ("python", "train_cache", "validation_cache", "train_override", "validation_override", "model_root"):
        if not args.dry_run and not Path(c[key]).exists():
            raise FileNotFoundError(c[key])
    if args.node_rank == 0 and output.exists() and any(output.iterdir()) and not c.get("resume_from_checkpoint"):
        raise ValueError(f"fresh output is not empty: {output}")
    if not args.dry_run:
        temp.mkdir(parents=True, exist_ok=True)
        for key in ("train_override", "validation_override"):
            original = Path(c[key])
            with original.open() as f:
                reader = csv.DictReader(f)
                fields = reader.fieldnames
                rows = list(reader)
            for row in rows:
                for field in ("override_path", "condition_path", "cache_source", "original_cache_path", "prompt_context_path", "video"):
                    if row.get(field) and not Path(row[field]).is_absolute():
                        row[field] = str((original.parent / row[field]).resolve())
            resolved = temp / (key + ".csv")
            with resolved.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            c[key] = str(resolved)
    model = Path(c["model_root"])
    model_paths = [[str(model / f"diffusion_pytorch_model-{i:05d}-of-00003.safetensors") for i in (1, 2, 3)],
                   str(model / "models_t5_umt5-xxl-enc-bf16.pth"), str(model / "Wan2.2_VAE.pth")]
    env = dict(os.environ)
    env.update(TMPDIR=str(temp), TMP=str(temp), TEMP=str(temp), IMAGEIO_FFMPEG_TEMP_DIR=str(temp),
               PYTHONPATH=str(root), CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7", OMP_NUM_THREADS="4",
               TOKENIZERS_PARALLELISM="false", NCCL_SOCKET_IFNAME=c.get("network_interface", "eth0"), GLOO_SOCKET_IFNAME=c.get("network_interface", "eth0"),
               NCCL_IB_DISABLE="1", NCCL_DEBUG="WARN", TORCH_NCCL_ASYNC_ERROR_HANDLING="1",
               PHYSICAL_WM_OBJECT_MASS="1", PHYSICAL_WM_MASS_ENCODER=c.get("mass_encoder", "linear"),
               PHYSICAL_WM_TEXT_ONLY="1" if c["architecture"] == "textonly_lora" else "0",
               PHYSICAL_WM_MAX_GRAD_NORM="1" if c["architecture"] == "textonly_lora" else "0",
               PHYSICAL_WM_THREE_CONTACT_GATES="1" if c["architecture"] == "h2_mass" else "0", PHYSICAL_WM_SHARED_CONTACT_GATE="0",
               PHYSICAL_WM_CONTACT_SUPERVISION="0", PHYSICAL_WM_PAIR_EVENT_SUPERVISION="0",
               PHYSICAL_WM_DISABLE_GRAVITY="0", PHYSICAL_WM_WRITE_GATE_MODE="legacy",
               PHYSICAL_WM_CAUSAL_OBJECT_TEMPORAL="0", PHYSICAL_WM_EVENT_AFTER_EFFECT="0",
               PHYSICAL_WM_STRICT_CACHE_OVERRIDE_HASHES="0", PHYSICAL_WM_WRITER_SUPPORT_MASS="1.0",
               PHYSICAL_WM_HARD_SUPPORT="0", PHYSICAL_WM_HARD_SUPPORT_LOSS_WEIGHT="0.0",
               PHYSICAL_WM_CORRECTABLE_ROUTE_CARRY="1", PHYSICAL_WM_ROUTE_PRIOR_ALPHA="1.0",
               PHYSICAL_WM_POSITION_DEPENDENT_WRITER_VALUE="0", PHYSICAL_WM_RELATIVE_POSITION_WRITER_VALUE="0")
    cmd = [c["python"], "-m", "accelerate.commands.launch", "--multi_gpu", "--num_machines", "2",
           "--num_processes", "16", "--machine_rank", str(args.node_rank), "--main_process_ip", c["master_addr"],
           "--main_process_port", str(c["master_port"]), "--same_network", "--rdzv_backend", "static",
           str(root / "train.py"), "--task", "sft:train",
           "--dataset_base_path", c["train_cache"], "--cache_override_manifest", c["train_override"],
           "--enable_prompt_context_override", "--validation_cache_path", c["validation_cache"],
           "--validation_cache_override_manifest", c["validation_override"],
           "--validation_steps", str(c["validation_steps"]), "--validation_seed", "424242",
           "--data_file_keys", "video", "--output_path", str(output), "--height", "448", "--width", "768",
           "--num_frames", "49", "--video_resize_mode", "pad", "--dataset_repeat", "1",
           "--dataset_num_workers", "0", "--train_batch_size", "2", "--drop_incomplete_global_batch",
           "--model_paths", json.dumps(model_paths), "--tokenizer_path", str(model / "google/umt5-xxl"),
           "--learning_rate", str(c["learning_rate"]), "--weight_decay", "0.01", "--num_epochs", "100",
           "--seed", "42", "--remove_prefix_in_ckpt", "pipe.dit.", "--use_gradient_checkpointing",
           "--gradient_accumulation_steps", "1", "--max_train_steps", str(c["max_train_steps"]),
           "--save_steps", str(c["save_steps"]), "--skip_step0_equivalence"]
    if c["architecture"] == "h2_mass":
        cmd += ["--trainable_models", "dit", "--enable_sparse_object_interaction_adapter",
                "--sparse_object_adapter_architecture", "temporal_independent_edge_schedule_decoupled_writer_continuous",
                "--sparse_object_injection_blocks", "10,11,12,13,14,15,16,17",
                "--object_attention_loss_weight", "0.3", "--writer_loss_weight", "0.3",
                "--reader_supervision_mode", "mask_route_read", "--dynamic_loss_weight", "0.2",
                "--temporal_loss_weight", "0.2", "--e_pair_event_loss_weight", "0.0",
                "--mu_pair_event_loss_weight", "0.0", "--contact_loss_weight", "0.0",
                "--base_contact_loss_weight", "0.1", "--e_contact_loss_weight", "0.1",
                "--mu_contact_loss_weight", "0.1", "--extra_inputs", "input_image,sparse_object_condition"]
    else:
        cmd += ["--lora_base_model", "dit", "--lora_target_modules", "q,k,v,o,ffn.0,ffn.2",
                "--lora_rank", "64", "--extra_inputs", "input_image",
                "--object_attention_loss_weight", "0", "--writer_loss_weight", "0",
                "--dynamic_loss_weight", "0", "--temporal_loss_weight", "0", "--impact_loss_weight", "0",
                "--e_pair_event_loss_weight", "0", "--mu_pair_event_loss_weight", "0",
                "--contact_loss_weight", "0", "--base_contact_loss_weight", "0",
                "--e_contact_loss_weight", "0", "--mu_contact_loss_weight", "0"]
    if c.get("resume_from_checkpoint"):
        cmd += ["--resume_from_checkpoint", c["resume_from_checkpoint"]]
    if c.get("smoke"):
        env.update(WANDB_MODE="disabled", WANDB_DISABLED="true", ENABLE_WANDB_LOG="0")
    else:
        project = "physical-wm" if c["architecture"] == "h2_mass" else "official"
        env.update(WANDB_MODE=c.get("wandb_mode", "offline"), WANDB_RUN_ID=c["run_id"],
                   WANDB_NAME=c["run_id"], WANDB_PROJECT=project, WANDB_RESUME="allow", ENABLE_WANDB_LOG="1")
        cmd += ["--enable_wandb_log", "--wandb_project", project]
    if c.get("single_gpu_smoke"):
        assert c.get("smoke") and args.node_rank == 0 and c["max_train_steps"] <= 3
        cmd = [c["python"]] + cmd[cmd.index(str(root / "train.py")):]
        env["CUDA_VISIBLE_DEVICES"] = str(c["smoke_gpu"])
    if args.dry_run:
        print(json.dumps({"mode": "configuration_only", "command": cmd,
                          "training_environment": {k: v for k, v in env.items() if k.startswith("PHYSICAL_WM_")}}, indent=2))
        return
    os.chdir(root)
    os.execvpe(cmd[0], cmd, env)


if __name__ == "__main__":
    main()
