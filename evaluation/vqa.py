"""Deterministic FAST-VQA and FasterVQA video scoring."""
import hashlib
import random
import sys
from pathlib import Path
import numpy as np
import torch

SCORERS = {"FAST-VQA", "FasterVQA"}

FAST_MEAN_STDS = {
    "FasterVQA": (0.14759505, 0.03613452),
    "FAST-VQA": (-0.110198185, 0.04178565),
}

FAST_CONFIGS = {
    "FasterVQA": "options/fast/f3dvqa-b.yml",
    "FAST-VQA": "options/fast/fast-b.yml",
}

def task_seed(task_id: str) -> int:
    return int.from_bytes(hashlib.sha256(task_id.encode()).digest()[:4], "big")

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class FastScorer:
    def __init__(self, scorer: str, repo: Path) -> None:
        import yaml

        torch.cuda.init()
        self.scorer = scorer
        self.repo = repo.resolve()
        sys.path.insert(0, str(self.repo))
        from fastvqa.models import DiViDeAddEvaluator

        with (self.repo / FAST_CONFIGS[scorer]).open() as handle:
            self.options = yaml.safe_load(handle)
        checkpoint = self.repo / self.options["test_load_path"]
        self.model = DiViDeAddEvaluator(**self.options["model"]["args"]).cuda().eval()
        state = torch.load(checkpoint, map_location="cuda")
        self.model.load_state_dict(state["state_dict"])

    def __call__(self, video_path: str, seed: int) -> tuple[float, float]:
        import decord
        from fastvqa.datasets import FragmentSampleFrames, SampleFrames, get_spatial_fragments

        set_seed(seed)
        reader = decord.VideoReader(video_path, ctx=decord.cpu(0), num_threads=1)
        data_options = self.options["data"]["val-kv1k"]["args"]
        samples = {}
        for sample_type, sample_args in data_options["sample_types"].items():
            if data_options.get("t_frag", 1) > 1:
                sampler = FragmentSampleFrames(
                    fsize_t=sample_args["clip_len"] // sample_args.get("t_frag", 1),
                    fragments_t=sample_args.get("t_frag", 1),
                    num_clips=sample_args.get("num_clips", 1),
                )
            else:
                sampler = SampleFrames(
                    clip_len=sample_args["clip_len"],
                    num_clips=sample_args["num_clips"],
                )
            frames = sampler(len(reader))
            frame_dict = {index: reader[index] for index in np.unique(frames)}
            video = torch.stack([frame_dict[index] for index in frames], 0).permute(3, 0, 1, 2)
            sampled = get_spatial_fragments(video, **sample_args)
            mean = torch.tensor([123.675, 116.28, 103.53])
            std = torch.tensor([58.395, 57.12, 57.375])
            sampled = ((sampled.permute(1, 2, 3, 0) - mean) / std).permute(3, 0, 1, 2)
            num_clips = sample_args.get("num_clips", 1)
            sampled = sampled.reshape(sampled.shape[0], num_clips, -1, *sampled.shape[2:]).transpose(0, 1)
            samples[sample_type] = sampled.cuda()
        with torch.inference_mode():
            raw_score = float(self.model(samples).mean().item())
        mean, std = FAST_MEAN_STDS[self.scorer]
        normalized = float(1.0 / (1.0 + np.exp(-((raw_score - mean) / std))))
        return normalized, raw_score
