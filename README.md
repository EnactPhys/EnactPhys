<div align="center">

# EnactPhys
### Executing Evolving Physical Processes in Video Diffusion

**Anonymous Authors**

[Website](https://enactphys.github.io/) · [Checkpoints](https://huggingface.co/EnactPhys/EnactPhys) · [Dataset](https://huggingface.co/datasets/EnactPhys/PhysDelta) · [Reproduction](docs/reproduction.md)

</div>

EnactPhys controls video generation through evolving object states and physical interactions. Given an initial frame, object masks and physical parameters, it reads object information from video features, evolves and exchanges object states, and writes them back into the video diffusion model. PhysDelta measures how target and non-target objects respond to physical interventions.

The [project website](https://enactphys.github.io/) presents the architecture, single-parameter and joint control, generalization examples and composed events.

## Resources

| Repository | Contents |
| --- | --- |
| **This repository** | Model implementation, training and inference configurations, evaluators and table aggregation |
| [**Model**](https://huggingface.co/EnactPhys/EnactPhys) | Eight step-6000 checkpoints and loading metadata |
| [**Dataset**](https://huggingface.co/datasets/EnactPhys/PhysDelta) | PhysDelta inputs, fixed Physics-IQ inputs, raw training videos, conditions and split manifests |

Benchmark inputs and training data have separate downloads. Training-file availability is recorded in the dataset's [shard index](https://huggingface.co/datasets/EnactPhys/PhysDelta/blob/main/training_raw/index.json).

## Setup

Use Python 3.10 for the model runtime. Install the matching PyTorch CUDA build and runtime dependencies:

```bash
python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-runtime.txt
python -m pip install huggingface_hub
hf download Wan-AI/Wan2.2-TI2V-5B --local-dir weights/Wan2.2-TI2V-5B
hf download EnactPhys/EnactPhys --include 'enactphys/*' --local-dir weights
```

The released files contain trainable parameters. Keep `trainable_model.safetensors` and `complete.json` together in `weights/enactphys/`; base-model weights are downloaded separately. EnactPhys uses the MLP mass encoder and injection blocks 10–17.

## Generate PhysDelta videos

```bash
hf download EnactPhys/PhysDelta --repo-type dataset \
  --exclude 'training/*' 'training_raw/*' 'training_sources/*' --local-dir data/PhysDelta
tar -xzf data/PhysDelta/sim_inputs.tar.gz -C data/PhysDelta
tar -xzf data/PhysDelta/real_inputs.tar.gz -C data/PhysDelta
python scripts/generate_physdelta.py --dataset data/PhysDelta \
  --track real/parameter_control --seed 3407 \
  --base-model weights/Wan2.2-TI2V-5B --checkpoint weights/enactphys \
  --output-dir outputs/physdelta_real --dry-run
```

Remove `--dry-run` to generate videos. Use `--task-id` for one task or `--shard-index` and `--shard-count` for parallel processes. See the [track list and sampling settings](docs/reproduction.md#physdelta-generation).

## Evaluation

| Workflow | Guide |
| --- | --- |
| PhysDelta-Real parameter control: generation → tracking → score | [Single-command workflow](docs/video_evaluation.md#single-command-real-pc-workflow) |
| Physics-IQ Solid Mechanics: generation → video metrics → score | [Single-command workflow](docs/physicsiq_video_reproduction.md) |
| Video Quality and Physical Plausibility | [Video evaluators](docs/video_evaluation.md) |
| MORPHEUS tracking, video scoring and screening | [Video scoring](docs/morpheus_video_scoring.md) · [Screening protocol](docs/morpheus_screening.md) |
| Main table, ablations and object-state measurements | [Recorded measurement aggregation](docs/reproduction.md#recorded-measurements) |

To aggregate the released measurements:

```bash
python scripts/reproduce.py --list
python scripts/reproduce.py --task main-table-recorded --out-dir outputs/main_table
python scripts/reproduce.py --task ablation-recorded --out-dir outputs/ablation
```

These commands check the 99 main-table values and 28 ablation values against the released records. They aggregate measurements and judgments; generating new videos and extracting new scores use the separate video workflows above. Physics-IQ uses 114 views and three seeds, with the fixed adapter/base routing in its manifests. [Protocol details](docs/protocol.md) specify the metric definitions and evaluation settings.

The [reproduction guide](docs/reproduction.md) states the scope of each executable workflow. A full fresh GPU rerun of the release package is distinct from these recorded-data checks.

## Raw training data and preprocessing

The data split contains **109,800 training rows** and **4,196 validation rows**. Download raw videos, physical-condition tensors and manifests with:

```bash
python scripts/prepare_training_data.py --download \
  --download-dir data/PhysDelta --output data/training_raw
```

This command extracts the available shards and checks every video and condition reference. It stops if coverage is incomplete. Additional source archives and their mappings are described in the [dataset card](https://huggingface.co/datasets/EnactPhys/PhysDelta). Previously uploaded optional encoding caches are separate and are not part of this download.

[The preprocessing guide](docs/training_data.md) explains video encoding, prompt selection and binding generated caches to the training configuration. VAE and text features can be computed with the provided scripts.

[configs/train_enactphys.json](configs/train_enactphys.json) specifies 16 GPUs across two nodes, batch size 2 per rank, 6,000 steps, seed 42 and validation every 1,000 steps. Set data, model, output and rendezvous paths before launching:

```bash
# Run on each node with its corresponding rank.
python scripts/train.py --config configs/train_enactphys.json --node-rank 0
python scripts/train.py --config configs/train_enactphys.json --node-rank 1
```

## Structure

```text
configs/                 Training configurations
runtime/                 Inference and training implementations
evaluation/              Video measurements and scoring
physicsiq/               Fixed tasks and Physics-IQ scoring implementation
results/                 Recorded benchmark and mechanism measurements
scripts/                 Command-line entry points
docs/                    Protocols and reproduction instructions
licenses/                Third-party notices
```

## Licenses

Third-party code, pretrained models and datasets retain their original licenses and attribution. Included DiffSynth and Physics-IQ license notices are retained in the repository. A project-specific license for EnactPhys code, checkpoints and data has not yet been assigned.
