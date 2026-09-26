<div align="center">

# EnactPhys
### Executing Evolving Physical Processes in Video Diffusion

**Anonymous Authors**

[Project page](https://enactphys.github.io/) · [Setup](#setup) · [Inference](#inference) · [Evaluation](#evaluation) · [Reproduction guide](docs/reproduction.md)

</div>

EnactPhys controls generated object motion through evolving object states and parameter-conditioned interactions. Given an initial frame, object masks and physical parameters, it updates object states and writes their information into video diffusion. PhysDelta evaluates the response of target and non-target objects to physical interventions.

> Initial code release. Recorded-measurement aggregation is available for the main and ablation tables. Eight checkpoints and PhysDelta control and quality-evaluation inputs are available; full video-to-table execution of this release package has not yet been validated.

## Method

- **Read:** extract object-specific information from video features.
- **Evolve:** update object states over time with force, gravity and mass conditioning.
- **Interact:** exchange messages between objects and support surfaces, conditioned on friction and restitution.
- **Write:** inject updated object-state information into video features during denoising.

The [project page](https://enactphys.github.io/) presents selected examples of generalization to human motion, single-parameter control, joint control and composed events.

## Resources

| Resource | Contents | Status |
| --- | --- | --- |
| [Project page](https://enactphys.github.io/) | Selected videos and state visualizations | Available |
| [GitHub code](https://github.com/EnactPhys/EnactPhys) | Implementation, configurations and reproduction commands | Available |
| [Hugging Face model](https://huggingface.co/EnactPhys/EnactPhys) | Eight checkpoints and loading configuration | Available |
| [Hugging Face dataset](https://huggingface.co/datasets/EnactPhys/PhysDelta) | 3,612 evaluation task inputs, plus six separate diagnostics | Available |

See [resource organization](docs/resources.md) for training data and optional evaluation-output archives. Third-party base models are obtained from their original distributors.

## Setup

Recorded table, Physics-IQ and mechanism aggregation use the Python standard library. Recomputing baseline Physics-IQ scores from metric CSVs additionally requires:

```bash
python -m pip install -r physicsiq/requirements.txt
```

The GPU runtime specifies Python 3.10, PyTorch 2.7.0, torchvision 0.22.0 and CUDA 12.8 PyTorch builds. Dependencies are listed in [requirements-runtime.txt](requirements-runtime.txt). Clean-environment installation and end-to-end GPU execution of this package are pending validation.

## Inference

Download the EnactPhys adapter and PhysDelta benchmark inputs:

```bash
hf download EnactPhys/EnactPhys --include 'enactphys/*' --local-dir weights
hf download EnactPhys/PhysDelta --repo-type dataset --exclude 'training/*' 'training_raw/*' 'training_sources/*' --local-dir data/PhysDelta
tar -xzf data/PhysDelta/sim_inputs.tar.gz -C data/PhysDelta
tar -xzf data/PhysDelta/real_inputs.tar.gz -C data/PhysDelta
python scripts/generate_physdelta.py --dataset data/PhysDelta \
  --track real/parameter_control --seed 3407 \
  --base-model weights/Wan2.2-TI2V-5B --checkpoint weights/enactphys \
  --output-dir outputs/physdelta_real --dry-run
```

Remove `--dry-run` to run inference. Use `--task-id` for one task or `--shard-index` and `--shard-count` to divide tasks across GPU processes. `--dry-run` checks every selected input path without running the model. See [PhysDelta generation](docs/reproduction.md#physdelta-generation) for the track list and sampling settings.

### Physics-IQ

The included generation entry point reads fixed Physics-IQ task manifests. Prepare the Wan2.2-TI2V-5B base model, input images and conditions, and the step-6000 adapter for adapter-enabled tasks. The base-model directory contains the DiT shards, T5 encoder, VAE and tokenizer.

Inspect one task:

```bash
python scripts/generate.py --route adapter --assets-root data --base-model weights/Wan2.2-TI2V-5B --checkpoint weights/enactphys --output-dir outputs/example --benchmark-id 0001 --seed 43278311 --dry-run
```

Remove `--dry-run` to execute on a compatible GPU once the assets are available. `--route base` selects tasks with the adapter disabled. Omit the view and seed filters to run all tasks in the selected route. See [generation instructions](docs/reproduction.md#generation) and [evaluation configurations](docs/protocol.md#physics-iq).

## Evaluation

List the available workflows:

```bash
python scripts/reproduce.py --list
```

Aggregate a selected set of recorded measurements:

```bash
python scripts/reproduce.py --task main-table-recorded --out-dir outputs/recorded_table
python scripts/reproduce.py --task ablation-recorded --out-dir outputs/recorded_ablation
python scripts/reproduce.py --task physicsiq-recorded --out-dir outputs/recorded_physicsiq
python scripts/reproduce.py --task morpheus-recorded --out-dir outputs/recorded_morpheus
python scripts/reproduce.py --task mechanisms-recorded --out-dir outputs/recorded_mechanisms
```

These commands compute aggregates from included measurements, evaluator means and judgments. The current Physics-IQ records aggregate to 48.16. New-video generation and pixel-level scoring are separate operations. The main-table command includes both MORPHEUS columns and checks all 99 displayed values against the approved table. See [workflow coverage](docs/reproduction.md) and [metric definitions](docs/protocol.md).

## Training

The [raw training-data release](https://huggingface.co/datasets/EnactPhys/PhysDelta/tree/main/training_raw) contains original located videos, physical conditions and fixed train/validation manifests. The dataset's `training_raw/index.json` records shard availability and original-video coverage. Previously uploaded encoding caches are optional and incomplete.

Download, extract and check all required inputs with:

```bash
python scripts/prepare_training_data.py --download --output data/training_raw
```

The command stops with a coverage report if any required input is unavailable. To encode a resolved raw-data manifest on a GPU, use the released Wan preprocessing path (49 frames, 768 × 448, padding, seed 42):

```bash
python scripts/preprocess_training.py --manifest data/training_raw/train.csv \
  --data-root data/training_raw --model-root weights/Wan2.2-TI2V-5B \
  --output data/encoded/train --prompt-column neutral_prompt
```

`--prompt-column` is explicit: select the prompt field required by the training route. `text_prompt` contains numeric text controls; `neutral_prompt` contains scene descriptions. The encoder supports `--shard-index` and `--shard-count`; each shard must use a separate output directory. It writes a clip-to-cache index. Physical conditions remain in the raw dataset. Reencoding and historical prompt-context equivalence have not yet been validated across the complete corpus; these preparation tools do not establish an exact retraining reproduction.

[configs/train_enactphys.json](configs/train_enactphys.json) specifies 16 GPUs across two nodes, batch size 2 per rank, 6,000 steps, seed 42 and validation every 1,000 steps. Set model, data, output and rendezvous paths after preparing the training caches and conditions.

```bash
# Run on each node with its corresponding rank.
python scripts/train.py --config configs/train_enactphys.json --node-rank 0
python scripts/train.py --config configs/train_enactphys.json --node-rank 1
```

## Repository structure

```text
configs/                 Training and runtime configuration
docs/                    Protocol, resources and reproduction guide
licenses/                Third-party license notices
physicsiq/               Task manifests and metric aggregation
results/                 Recorded benchmark and mechanism measurements
runtime/inference/       Inference implementation
runtime/training/        Training implementation
scripts/                 Generation, training and aggregation entry points
requirements-runtime.txt Runtime dependencies
```

## Licenses

Included DiffSynth source retains its Apache-2.0 license under [licenses](licenses/). Physics-IQ scoring source retains its license under [physicsiq/vendor](physicsiq/vendor/). Third-party models and datasets retain their original licenses and attribution. A project-specific license for EnactPhys code, checkpoints and datasets has not yet been assigned.

Pixel-scoring commands for VQA and PP are described in the
[video evaluation guide](docs/video_evaluation.md).
