# MORPHEUS video scoring

`evaluate_morpheus.py` uses the bundled MORPHEUS scorer and SAM2.1 tracking code. Their original MIT and Apache licenses are retained under `runtime/morpheus/`. Install the main PyTorch runtime and `requirements-morpheus.txt`, and obtain the official SAM2.1 Hiera Large checkpoint. Depth tracking also loads `nielsr/depth-anything-large` from Hugging Face; download it into the Hugging Face cache before offline evaluation.

Provide a CSV with `model,task_id,case,seed,prompt_type,video`. Videos are 49 frames, 768 × 448, at 24 FPS. Relative video paths resolve beneath `--video-root`. Seeds are 937, 5318 and 1888. The released `results/morpheus/per_video.csv` provides the fixed model/task identities; when scoring regenerated videos, point its video column at those new files.

```bash
python scripts/evaluate_morpheus.py --manifest inputs/morpheus.csv \
  --video-root data --sam2-checkpoint weights/sam2.1_hiera_large.pt \
  --output-dir outputs/morpheus_pixels --dry-run
```

Remove `--dry-run` to perform preprocessing, SAM2 tracking and dynamical/physical scoring on the evaluation machine. The output directory must be new. Each video's tracking evidence and `combined_scores.json` are retained; `raw_measurements.json` collects the measurements.

The frame transformation, label coordinates and collision-mass estimator match the packaged evaluation code. The PINN seed is 3407. For falling-marker, falling-apple, falling-ball and falling-tape, the minimum crop index is 15; the underlying stopping formulas and thresholds are unchanged. Other cases retain the scorer's default minimum of 20.

Raw measurements precede the [screening and aggregation procedure](morpheus_screening.md). New videos need their own screening decisions; recorded decisions must not be reassigned to different videos. `scripts/finalize_morpheus_review.py` applies supplied decisions to their matching raw measurements.
