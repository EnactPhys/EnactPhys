"""Run EnactPhys generation, SAM2 tracking and Real parameter-control scoring."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sam2-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=[3407, 424242, 918273])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    common = ["--dataset", str(args.dataset.resolve())]
    selected = ["--seed", str(args.seed)] if args.seed is not None else []
    commands = [
        [sys.executable, str(ROOT / "scripts/generate_physdelta.py"), *common,
         "--track", "real/parameter_control", "--base-model", str(args.base_model.resolve()),
         "--checkpoint", str(args.checkpoint.resolve()), "--output-dir", str(out / "generation"), *selected],
        [sys.executable, str(ROOT / "scripts/track_physdelta_real.py"), *common,
         "--videos", str(out / "generation/videos"), "--sam2-checkpoint", str(args.sam2_checkpoint.resolve()),
         "--output-dir", str(out / "tracking"), *selected],
        [sys.executable, str(ROOT / "scripts/evaluate_physdelta_real_pc.py"), *common,
         "--tracking-dir", str(out / "tracking"), "--output-dir", str(out / "score"), *selected],
    ]
    if args.dry_run:
        subprocess.run(commands[0] + ["--dry-run"], check=True)
        print(json.dumps(dict(status="INPUTS_VALIDATED", commands=commands,
                              gpu_execution=False), indent=2))
        return
    required = [args.sam2_checkpoint, args.checkpoint / "complete.json",
                args.checkpoint / "trainable_model.safetensors", args.base_model / "models_t5_umt5-xxl-enc-bf16.pth",
                args.base_model / "Wan2.2_VAE.pth"]
    required += [args.base_model / ("diffusion_pytorch_model-%05d-of-00003.safetensors" % i) for i in [1,2,3]]
    for asset in required:
        if not asset.is_file():
            raise FileNotFoundError(asset)
    if not (args.base_model / "google/umt5-xxl").is_dir():
        raise FileNotFoundError(args.base_model / "google/umt5-xxl")
    if out.exists():
        raise FileExistsError("Use a new output directory, or resume individual stage commands")
    out.mkdir(parents=True)
    (out / "workflow.json").write_text(json.dumps(dict(commands=commands), indent=2) + "\n")
    for stage, command in zip(["generation", "tracking", "scoring"], commands):
        print(json.dumps(dict(stage=stage, status="STARTED")), flush=True)
        subprocess.run(command, check=True)
        print(json.dumps(dict(stage=stage, status="COMPLETE")), flush=True)
    print((out / "score/summary.json").read_text())


if __name__ == "__main__":
    main()
