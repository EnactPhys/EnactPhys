# Video evaluation

These commands evaluate supplied videos. Run pixel processing on the evaluation
machine with its required dependencies. Use the benchmark's fixed sample manifest;
do not substitute the website's selected examples. The entries below are ported
from the recorded evaluation implementation. End-to-end execution of these
portable entries has not yet been validated.

## Video quality

Prepare the FAST-VQA repository, its dependencies and the checkpoint referenced
by `options/fast/f3dvqa-b.yml`. The checkpoint path in that configuration resolves
relative to the repository. Supply a CSV with `model,task_id,video` columns.
Video paths may be absolute or relative to the manifest directory.

```bash
python scripts/evaluate_vqa.py --manifest inputs/vqa.csv \
  --fast-repo dependencies/FAST-VQA --scorer FasterVQA \
  --out-dir outputs/vqa --dry-run
```

Remove `--dry-run` to execute on a CUDA GPU. The task-ID-derived sampling seed,
fragment sampling, normalization and score transformation match the recorded
implementation. The output retains raw scores and normalized scores per video;
the reported percentage is 100 times their mean. An unsuccessful sample produces
an error record and makes the run partial, with a nonzero exit status.

## Physical plausibility

Supply a JSONL manifest with `id`, `model`, `task_id`, `video_path` and
`trajectory_path`. IDs must be unique safe filenames. Paths may be relative to
the manifest directory. Trajectories use the existing SAM2 record schema:
`video_contract.width/height`, and `records` containing `frame_index` and an
`objects` map with `total_area` and component `bbox_xywh` measurements.

The evaluator constructs the fixed 16-frame grid and uses the exact
[PP prompt](pp_prompt.txt). Configure an OpenAI-compatible chat-completions
endpoint through `EVAL_API_BASE_URL` (including `/v1`) and `EVAL_API_KEY` in the
environment. Do not place credentials in manifests or tracked files.

```bash
python scripts/evaluate_pp.py --manifest inputs/pp.jsonl \
  --model qwen3-vl-235b-a22b-instruct \
  --out-dir outputs/pp --dry-run
```

Remove `--dry-run` to build grids and call the configured evaluator. This requires
OpenCV and Pillow, in addition to the supplied trajectory measurements. Successful
responses, including `UNCERTAIN`, are not queried again within the run. Failed
calls have at most three attempts, with response records retained.

To use existing response JSONs without frame extraction or API calls:

```bash
python scripts/evaluate_pp.py --manifest inputs/pp.jsonl \
  --responses-dir measurements/pp/responses --out-dir outputs/pp_from_responses
```

Each response file is named `<id>.json` and contains `ok` plus the original
chat-completions `response`. The final rule is FAIL when the VLM reports a
supported visual defect or penetration, or the trajectory check finds interior
track loss. The nine supported defect categories are enumerated in
`evaluation/pp_rules.py`. A VLM disappearance label alone is insufficient: that
case uses the trajectory rule. A valid `UNCERTAIN` response with no confirmed
failure signal passes under the fixed rule. Invalid or missing evaluator
responses remain errors; they are not automatically assigned PASS or FAIL.

An error-free run reports the mean final PASS indicator. Partial runs report
their valid-sample mean and full-denominator bounds, and exit unsuccessfully.
Published recorded labels are available separately through the recorded-table
workflow. New calls can differ from those recorded judgments.

## Other metrics

PC and OC table aggregation, Physics-IQ recorded metrics and MORPHEUS recorded
scores have the entry points in [the reproduction guide](reproduction.md).
The [MORPHEUS screening workflow](morpheus_screening.md) includes the fixed
reviewer instructions and a decision-to-score join command.

Portable pixel-to-trajectory entry points for all PC/OC variants and complete
pixel-level Physics-IQ/MORPHEUS execution remain to be integrated. These are not
provided by the recorded-table commands. Base models, evaluation models,
checkpoints and benchmark inputs must be obtained separately before GPU runs.

## PhysDelta-Real parameter control

After generating videos with `scripts/generate_physdelta.py`, extract their trajectories with the published initial tracking masks. The bundled SAM2 implementation accepts in-memory image sequences. Obtain the SAM2 Hiera Large checkpoint from its original distributor and install `requirements-tracking.txt` in the runtime environment.

```bash
python scripts/track_physdelta_real.py --dataset data/PhysDelta \
  --videos outputs/physdelta_real/videos --seed 3407 \
  --sam2-checkpoint weights/sam2_hiera_large.pt --output-dir outputs/real_tracking
python scripts/evaluate_physdelta_real_pc.py --dataset data/PhysDelta \
  --tracking-dir outputs/real_tracking --seed 3407 --output-dir outputs/real_pc
```

Omit `--seed` to process all three seeds. Tracking may be split using `--shard-index` and `--shard-count`; use the same output directory and distinct shard indices. The runner accepts 49-frame, 768 × 448 videos. Other model outputs require their documented preprocessing before this entry point.

Tracking preserves the observed initial mask for frame-zero measurement and propagates SAM2 masks for subsequent frames. The scorer uses the published 708 matched task pairs, fixed scene-specific displacement responses and strict signed change greater than 0.5 pixels. Missing readouts count as failures in the fixed denominator. It writes every pair and a per-seed summary. These entries preserve the recovered readout and tracking implementation; fresh GPU validation of this packaged workflow is pending.
