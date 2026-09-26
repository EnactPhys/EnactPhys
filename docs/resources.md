# Project resources

| Resource | Location | Contents |
| --- | --- | --- |
| Code | [GitHub](https://github.com/EnactPhys/EnactPhys) | Implementations, configurations, inference, training and evaluation |
| Website | [Project page](https://enactphys.github.io/) | Selected videos, architecture and method description |
| Models | [EnactPhys](https://huggingface.co/EnactPhys/EnactPhys) | Eight checkpoint directories, metadata and file manifest |
| Data | [PhysDelta](https://huggingface.co/datasets/EnactPhys/PhysDelta) | Evaluation inputs, raw training videos, physical conditions and split manifests |

## Data directories

- `sim_inputs.tar.gz`, `real_inputs.tar.gz`: PhysDelta evaluation inputs.
- `physicsiq_inputs.tar.gz`: 114 input images and 80 condition files for the fixed Physics-IQ tasks. Official reference videos and masks are downloaded from the benchmark distributor.
- `training_raw/`: raw-video and physical-condition shards, training/validation manifests and coverage index.
- `training_sources/`: additional original source videos and their sample mappings; consult its README for the relation to historical training copies.
- `training/`: retained optional, partial preencoded caches. These are excluded from the default benchmark and raw-training downloads.

Generated website demonstrations are selected qualitative examples. They are not a substitute for the fixed quantitative evaluation task manifests. Generating the evaluation videos uses the published prompts, inputs, seeds, checkpoint and sampling configuration.

Reproduction commands live in the [workflow guide](reproduction.md), with dedicated [Physics-IQ](physicsiq_video_reproduction.md), [video evaluation](video_evaluation.md), and [training-data](training_data.md) instructions. Model variants use their matching architecture and checkpoint metadata. Third-party base models and evaluation models are obtained from their distributors.
