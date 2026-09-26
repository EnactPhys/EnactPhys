# Evaluation configurations

These configurations describe the approved 2026-09-26 main-table measurements.

## Physics-IQ

| Setting | Value |
| --- | --- |
| Subset | Solid Mechanics |
| Scenarios | 38 |
| Views per scenario | 3 |
| Seeds | 43278311, 56382197, 68491523 |
| Generated videos | 342 |
| Adapter-enabled views | 80 |
| Base-model views, adapter disabled | 34 |
| Route manifest | `physicsiq/data/generation_manifest.csv` |

All tasks use the official OP descriptions. Adapter/base routes and object-mask
and physical-input configurations were selected using these benchmark cases and
seeds, without a separate held-out set for configuration selection. Each selected
scene configuration uses its full set of adapter views and all three seeds.
The release commands use the recorded assignment and configuration; they do not
perform a new selection. The task manifest specifies prompts, input frames,
conditions, seeds, CFG and sampling steps.

`physicsiq/reproduce.py` validates the 342 EnactPhys view/seed identities and
averages their measured per-video Verified scores. Baseline scores use three
per-seed results; `--from-metrics` recomputes them from the supplied metric CSVs
with the vendored Physics-IQ scorer. Regeneration uses the matching input images,
conditions, base model, adapter and sampling settings. Pixel identity across
hardware and software environments has not been established.

## PhysDelta table

| Metric | Configuration |
| --- | --- |
| Simulation PC | Magnitude, direction and invariance components weighted 513, 513 and 101; threshold 0.10 |
| Real PC | 708 directional comparisons per model |
| Simulation PP | 215 judgments per model |
| Real PP | 100 judgments per model |

PP aggregation uses the recorded final labels. Judgment records retain their
decision sources. Repeating evaluator calls produces a new measurement set.

For Real elasticity OC, a valid pair requires a measured target trajectory, at
least two valid frames, signed height response above 5 pixels and target ADE
above 0.1692766811711215. The score is target ADE divided by target ADE plus the
maximum non-target ADE. Invalid pairs score zero. Average the 24 pairs, convert
to percent, then average with the force and gravity scores. The formula is the
same for every model.

VQA values are recorded evaluator means. Simulation OC and Real force/gravity OC
use recorded aggregates. The table aggregation command consumes these values;
the [video evaluation guide](video_evaluation.md) describes pixel-scoring entries
and their current coverage.

## MORPHEUS

The evaluation uses 16 scenarios and seeds 937, 5318 and 1888: 48 entries per
model, 432 total entries and 360 unique videos. The fixed development seeds are
shared across methods. Final prompts follow each model's supported interface;
unsupported physical inputs are provided through text.

- Single pendulum, double pendulum and spring use the corresponding backbone
  with control modules disabled. Shared backbone outputs are reused explicitly.
- Force Prompting uses its force controller for the four force-bearing scenarios
  and the Cog backbone for the other twelve scenarios.
- The scoring window contains 49 frames at 24 fps, at 768 by 448. Cog and Force
  Prompting outputs use 17 native 8-fps frames resampled to this grid. H3 and PhyCo
  use the first 49 frames of their outputs.
- Tracking starts at frame zero. The four falling scenarios (apple, ball, marker
  and tape) use a minimum truncation index of 15 instead of 20. Other scenarios
  retain their existing threshold. The PINN seed is 3407.
- Anonymous AI visual screening follows [the fixed review rules](morpheus_screening.md)
  and records confirmed rejection and uncertain cases.
  Confirmed rejected samples receive zero while remaining in the denominator;
  uncertain samples retain their raw scores. A second aggregation assigns zero
  to uncertain samples for sensitivity analysis.
- Cases and seeds have equal weight. Spring Dynamical Score retains the scorer's
  x-coordinate and first-ten-percent implementation.

`results/morpheus/per_video.csv` retains raw scores, final scores, decisions and
model routes. The main table uses the final scores. This protocol adapts the
MORPHEUS pipeline to model interfaces and short-video outputs.

## Mechanism measurements

State readout uses the 5,238 test-split rows in
`results/mechanisms/state_probe.csv`. State-edit comparisons pool 42 wall/drop
inputs. Error reductions use the mean per-input errors in `state_edit.json`.

Message intervention contains 408 measurements: four families, 17 control
levels, three seeds and two variants. For each seed, subtract its outcome at the
first control level; reverse the sign for the ramp response, then take the median
across seeds. The ball-wall figure displays control levels up to 0.60. The CSV
contains all measured levels.

Window-intervention case and generation receipts cover the three figure cases.
Other intervention batches are outside this record set.

## Videos and examples

Scored-video archives associate videos with task IDs, seeds and per-video
measurements. Website examples are selected qualitative illustrations. Their
input, prompt, conditions, checkpoint and sampler settings identify the displayed
runs. Quantitative evaluation uses its complete specified sample set.
