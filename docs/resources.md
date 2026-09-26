# Project resources

| Resource | Location | Contents |
| --- | --- | --- |
| Code | GitHub: EnactPhys/EnactPhys | Implementation, configurations, inference, training and evaluation entry points |
| Project page | https://enactphys.github.io/ | Selected videos and method description |
| Model | Hugging Face Model: EnactPhys/EnactPhys (planned) | Checkpoints, architecture variants and loading metadata |
| Dataset | Hugging Face Dataset: EnactPhys/PhysDelta (planned) | Benchmark inputs, physical conditions, masks, references and splits |

The model and dataset repositories share the EnactPhys account. A Hugging Face
collection can group them on one page. Reproduction instructions and executable
commands live in the GitHub repository.

## Checkpoints

Store architecture variants in named subdirectories of the model repository,
with the matching model configuration. Separate repositories are optional when
variants have different dependencies or distribution terms. Third-party base
models are downloaded from their original publishers.

## Dataset and optional output archives

Keep benchmark inputs and generated evaluation outputs in separate directories
and identify them explicitly in the dataset card. Generated videos may be an
optional download under `evaluation_outputs/`; a separate results repository is
also possible if the archive becomes large. Include sample IDs, configurations
and per-sample measurements with any released scored-video set.

Training data uses separate splits or a dedicated dataset repository, depending
on its size and license. Training samples, benchmark inputs and model-generated
outputs have distinct manifests.

## Reproduction

[The reproduction guide](reproduction.md) lists the available workflows and their
current coverage. Generate and score each model with its matching checkpoint,
inputs and configuration. Existing scored-video archives can support direct
re-evaluation without repeating generation.

Code, model, data and evaluator revisions identify an experimental release.
Updates to the website layout do not change those revisions.
