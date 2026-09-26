# MORPHEUS screening

The same evaluability rules apply to all methods. Model names and physical scores
are hidden during visual review. The recorded review was conducted by AI agents
using GPT-6 Astra, including a second reviewer. It was not implemented as a batch
of independent per-video API calls.

## Reviewer instructions

You review video evaluability. Inputs contain only an anonymous ID, a scenario,
sampled frames or all 49 frames, and anonymous trajectory-span diagnostics.
Do not infer which method should perform better or select examples by score.

1. Check target count and identity relative to the initial frame. If at least two
   sampled frames suggest persistent extra copies, inspect all 49 frames. Reject
   only confirmed persistent duplication. Shadows, reflections, motion blur,
   growth in apparent size and occlusion do not establish duplication. Mark the
   case uncertain when identity or count cannot be determined.
2. Flag a low-motion candidate only if **every** target's full-trajectory bounding
   box has a diagonal no larger than 1% of the tracking-frame diagonal. Confirm
   near-stationarity by inspecting all frames before rejection. Missing tracks
   require further review. A stationary recipient in a collision is insufficient
   to reject the video. Background motion does not count as target motion.
3. Ordinary occlusion, leaving the frame, shape changes, camera movement, poor
   physical behavior and an unfinished event do not automatically cause
   rejection. The fixed scorer assesses physical behavior. Mark ambiguous
   identity or count as uncertain and record the evidence.
4. Return `pass`, `reject` or `uncertain`, with the anonymous ID, reason, frames
   actually reviewed and evidence reference. For confirmed stationarity, also
   set `low_motion_confirmed: true`. Do not pass a case before review, modify
   scores, change its seed or remove the original video.

The 1% threshold is a fixed screening criterion. It is separate from the PP
evaluator's agreement calibration.

## Workflow

1. Freeze sample identities and maintain a separate model-to-anonymous-ID map.
   The recorded additional review used shuffle seed 6669. Its initial sampled
   frame indices were 0, 8, 16, 24, 32, 40 and 48.
2. Compute each target's x/y trajectory span over finite coordinates. Divide the
   bounding-box diagonal by the tracking-frame diagonal (1280 by 1024 for the
   recorded tracker outputs). Retain track completeness and the upstream
   trajectory-filter decision separately. Review missing or incomplete tracks.
3. Give reviewers only anonymous images, scenario and motion diagnostics.
   Inspect the full 49 frames for low-motion candidates and suspected persistent
   duplication. Apply the instructions above and freeze the decisions before
   joining them to model names or scores.
4. Join those decisions with the raw scores using:

```bash
python scripts/finalize_morpheus_review.py \
  --input outputs/review_input.csv \
  --decisions outputs/decisions.json \
  --out-dir outputs/reviewed_scores
```

The input CSV requires `blind_id`, `model`, `task_id`, `D_raw`, `I_raw`,
`trajectory_reject` and `low_motion_candidate`. Boolean fields use `true` or
`false`. Additional columns are preserved. The JSON maps each anonymous ID to:

```json
{
  "verdict": "pass",
  "reason": "Reviewer's evidence-based reason",
  "frames_reviewed": "0-48",
  "evidence": "review/anonymous-id.png",
  "low_motion_confirmed": false
}
```

This schema example is not a recorded judgment. Every input must have exactly
one decision. The script validates coverage and required review fields; it does
not itself perform visual review or verify the contents of the evidence image.

An upstream trajectory rejection or a confirmed visual rejection sets both
scores to zero and retains the sample in the denominator. Uncertain cases retain
their raw scores in the primary result; a second result sets uncertain cases to
zero. Raw scores remain in the output. Use the same case and seed weights in
both aggregations. Shared backbone outputs remain explicitly identified when
assembling the complete benchmark table.
