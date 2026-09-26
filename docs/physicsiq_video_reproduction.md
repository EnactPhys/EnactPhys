# Physics-IQ video reproduction

The fixed Solid Mechanics manifest contains 114 views and three seeds (342 videos): 80 views use the EnactPhys adapter and 34 use the base model for each seed. The scripts retain the per-task prompts, conditions, route and seed. No parameter search is performed.

## Inputs

```bash
hf download EnactPhys/PhysDelta physicsiq_inputs.tar.gz --repo-type dataset --local-dir data
mkdir -p data/PhysDelta
tar -xzf data/physicsiq_inputs.tar.gz -C data/PhysDelta
hf download EnactPhys/EnactPhys --include 'enactphys/*' --local-dir weights
python -m pip install -r requirements-physicsiq-video.txt
```

Obtain the Wan2.2-TI2V-5B base model as described in the main README. Obtain the official reference videos and masks from [Physics-IQ Verified](https://huggingface.co/datasets/Anates-Labs-Research/Physics-IQ-Verified), following its access and download instructions. The arguments below point to its `split-videos/testing/24FPS` and `video-masks/real/24FPS` directories.

## Generate and evaluate

```bash
python scripts/reproduce_physicsiq.py \
  --assets-root data/PhysDelta \
  --base-model weights/Wan2.2-TI2V-5B --checkpoint weights/enactphys \
  --real-folder data/physics-iq-verified/split-videos/testing/24FPS \
  --real-masks data/physics-iq-verified/video-masks/real/24FPS \
  --output-dir outputs/physicsiq --dry-run
```

The dry run checks input images, conditions, model files and reference-video/mask paths. Remove `--dry-run` to run generation followed by video scoring. Add `--seed 43278311` for one of the three fixed seeds. The output directory must be new.

## Evaluate existing generated videos

```bash
python scripts/evaluate_physicsiq.py \
  --videos outputs/physicsiq \
  --real-folder data/physics-iq-verified/split-videos/testing/24FPS \
  --real-masks data/physics-iq-verified/video-masks/real/24FPS \
  --output-dir outputs/physicsiq_scores
```

This accepts flat `<task_id>.mp4` files or the two `adapter/videos` and `base/videos` directories created by the workflow. It requires all 114 views for each selected seed and fails on missing references or duplicated videos.

Each generated 49-frame clip is extended to 120 frames at 24 FPS by repeating its final frame. Foreground masks and pixel metrics use the included Physics-IQ implementation. Each seed is aggregated with `IQTable.final_score_view`, multiplied by 100; the table value is the arithmetic mean across seeds. Per-view measurements and the final `RESULT.json` are retained. The script computes scores from supplied videos; it does not substitute the recorded table values.
