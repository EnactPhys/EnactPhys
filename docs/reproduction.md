# Reproduction workflows

## Main-table version

The included main-table records correspond to the approved 2026-09-26 table:
nine models, eight PhysDelta Sim/Real metrics, Physics-IQ Verified on Solid
Mechanics, and two MORPHEUS metrics. The table command checks all 99 values.
EnactPhys Physics-IQ aggregates to 48.15780133372367, displayed as 48.16.

| Operation | Entry point | Inputs | Current coverage |
| --- | --- | --- | --- |
| Aggregate recorded table measurements | `scripts/reproduce.py --task main-table-recorded` | Included measurements, evaluator means and judgments | Current main table, 99 values |
| Aggregate architecture ablations | `scripts/reproduce.py --task ablation-recorded` | Included PC/OC components and comparisons | Seven variants, 28 values |
| Aggregate Physics-IQ metrics | `scripts/reproduce.py --task physicsiq-recorded` | Included per-video metrics | Solid Mechanics, 342 EnactPhys records and eight baseline summaries |
| Aggregate MORPHEUS measurements | `scripts/reproduce.py --task morpheus-recorded` | Included per-video scores and screening decisions | Nine models, 432 entries |
| Aggregate object-state measurements | `scripts/reproduce.py --task mechanisms-recorded` | Included measurements | Archived state analyses |
| Generate Physics-IQ videos | `scripts/generate.py` | Base model, adapter where enabled, input images and conditions | Fixed adapter/base task manifests; GPU execution of this package pending validation |
| Score VQA and PP from videos | `scripts/evaluate_vqa.py`, `scripts/evaluate_pp.py` | Videos, evaluator models, PP trajectories and API access | Entry points prepared; GPU/API validation pending |
| Real parameter-control tracking and score | `scripts/track_physdelta_real.py`, `scripts/evaluate_physdelta_real_pc.py` | Generated videos, published masks and task pairs, SAM2 checkpoint | Packaged; fresh GPU validation pending |
| Physics-IQ generation and video scoring | `scripts/reproduce_physicsiq.py` | Fixed inputs, checkpoint, base model and official reference videos/masks | 114 views per seed; see [video workflow](physicsiq_video_reproduction.md) |
| MORPHEUS video scoring | `scripts/evaluate_morpheus.py` | Videos, SAM2.1 checkpoint, bundled labels and scoring model | [Tracking and scoring workflow](morpheus_video_scoring.md), followed by separate screening |
| Other pixel metrics | Benchmark-specific evaluator | Videos, references and evaluator dependencies | Portable Sim PC/OC integration remains separate |

## Recorded measurements

Run from the repository root. Each command requires a new output directory.

```bash
python scripts/reproduce.py --list
python scripts/reproduce.py --task main-table-recorded --out-dir outputs/recorded_table
python scripts/reproduce.py --task ablation-recorded --out-dir outputs/recorded_ablation
python scripts/reproduce.py --task physicsiq-recorded --out-dir outputs/recorded_physicsiq
python scripts/reproduce.py --task morpheus-recorded --out-dir outputs/recorded_morpheus
python scripts/reproduce.py --task mechanisms-recorded --out-dir outputs/recorded_mechanisms
```

Add `--dry-run` to inspect the command. These workflows aggregate existing
measurements. They do not generate videos or extract new measurements from pixels.
These aggregation commands use the Python standard library. To recompute the
eight baselines from the supplied metric CSVs with the vendored scorer:

```bash
python -m pip install -r physicsiq/requirements.txt
python physicsiq/reproduce.py --from-metrics --out-dir outputs/physicsiq_metrics
```

EnactPhys uses its 342 measured per-video Verified scores in both modes. The
baseline metric-CSV check is additional to per-seed aggregation.

## Generation

The generation entry point selects tasks from the included manifests. It reads
the recorded seed, prompt, condition, number of sampling steps, CFG scale,
resolution and frame rate for each task.

```bash
python scripts/generate.py --route adapter \
  --assets-root data --base-model weights/Wan2.2-TI2V-5B \
  --checkpoint weights/enactphys --output-dir outputs/adapter_example \
  --benchmark-id 0001 --seed 43278311 --dry-run
```

After the required assets are available, remove `--dry-run` to execute generation
on a compatible GPU. Select `--route base` for tasks with the adapter disabled;
this route does not require `--checkpoint`. Omit the view and seed filters to run
all tasks assigned to that route. The assignment and its evaluation scope are in
[the protocol](protocol.md#physics-iq).

## Evaluating a model

A complete generation-and-evaluation configuration specifies:

- Code revision, checkpoint and base-model revision.
- Benchmark version, sample IDs, split, input frames, masks and physical conditions.
- Per-sample seeds, prompts, sampling settings and model switches.
- Frame selection, resizing, tracking, evaluator versions and metric aggregation.
- Recorded judgments or external evaluator responses for metrics that use them.

Each model uses its own configuration and model dependencies. Third-party API
models also require access to the specified service and model version. Optional
scored-video archives allow measurement checks without repeating generation.

## Output records

Keep each run's task manifest with its generated videos, per-sample measurements
and aggregate output. Use matching sample IDs throughout the pipeline. Published
qualitative examples use separate selections from the quantitative evaluation set.

## Anonymous MORPHEUS screening

The [screening instructions and workflow](morpheus_screening.md) specify reviewer
inputs, decisions and score aggregation. `scripts/finalize_morpheus_review.py`
joins supplied decisions to raw scores without dropping samples.

Pixel-scoring commands for VQA and PP are described in the
[video evaluation guide](video_evaluation.md).

## PhysDelta generation

The [dataset release](https://huggingface.co/datasets/EnactPhys/PhysDelta) includes the 3,594 control and Real-quality tasks, 18 additional Sim-quality task inputs, and six separately listed friction diagnostics. Extract both input archives under one dataset root. `scripts/generate_physdelta.py` validates task IDs and all selected image, condition and tracking-mask paths before generation.

Supported tracks are `sim/parameter_control`, `sim/invariance`, `sim/object_control`, `real/parameter_control`, `real/object_control`, and `real/quality`, `sim/plausibility`, `sim/video_quality`, and `real/friction_diagnostic`. The last track is not included in the formal object-control result.

The EnactPhys entry point uses the released step-6000 checkpoint, MLP mass encoder, injection blocks 10–17, CFG 1.2, 30 sampling steps, sigma shift 5.0, and 49 frames at 768 × 448. Output frame rate follows each manifest. Measurement readers use their specified time basis, which need not equal the display frame rate.

```bash
python scripts/generate_physdelta.py --dataset data/PhysDelta \
  --track sim/parameter_control --seed 42 \
  --base-model weights/Wan2.2-TI2V-5B --checkpoint weights/enactphys \
  --output-dir outputs/physdelta_sim --dry-run
```

Remove `--dry-run` to execute. This is the EnactPhys configuration; baseline and ablation methods require their own implementation and settings. The published-input check does not certify a fresh GPU run or a complete regenerated table.

The quality selections reuse numeric-control inputs. `real/quality` contains 75 additional seed-42 PC tasks and 25 existing OC tasks; `sim/plausibility` and `sim/video_quality` each select 215 tasks. The union of the two Sim quality panels contains 18 additional quality-only task IDs; their inputs are included explicitly.

Detailed commands: [Physics-IQ](physicsiq_video_reproduction.md) and [MORPHEUS](morpheus_video_scoring.md).
