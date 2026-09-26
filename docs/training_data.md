# Raw training data

The fixed split contains 109,800 training rows and 4,196 validation rows. Manifests retain sample order, prompts, physical parameters and duplication relationships. Videos and physical-condition tensors are separate from the VAE/text features computed during preprocessing.

## Download and check

```bash
python scripts/prepare_training_data.py --download \
  --download-dir data/PhysDelta --output data/training_raw
```

The command uses `training_raw/index.json`, extracts every available shard, and checks all video and condition references. It exits with an error when shards or originals are missing, so incomplete data cannot silently become a full-data training run. The dataset's `training_sources/` directory has additional source archives and sample mappings; these are kept distinct from the historical processed-video copies.

## Encode videos and prompts

Install the model runtime and download Wan2.2-TI2V-5B as shown in the main README. Choose the manifest prompt field for the intended training configuration: `prompt`, `neutral_prompt`, or `text_prompt`. The script requires an explicit field and rejects empty prompts. `text_prompt` includes numerical conditions for the text-conditioned configuration; it must not be substituted for an object-conditioned run's text input.

```bash
python scripts/preprocess_training.py \
  --manifest data/training_raw/train.csv --data-root data/training_raw \
  --model-root weights/Wan2.2-TI2V-5B \
  --prompt-column neutral_prompt --output outputs/encoded_train --gpu 0 --dry-run
```

Remove `--dry-run` to encode. This example selects the `neutral_prompt` field; it does not assert equivalence to every historical cache's text context. Select the matching prompt field when reproducing a specific configuration. Repeat with `validation.csv` and a separate output directory for validation.

Encoding uses the released `sft:data_process` implementation, 49 frames, 768 × 448 padded geometry, seed 42 and compact cache output. Each output has a sample-ID record. For parallel processing, pass distinct `--shard-index` values with a shared `--shard-count`, and use a different output directory for each process. Original video and condition files are retained.

## Bind encoded outputs to training

```bash
python scripts/bind_training_caches.py \
  --manifest data/training_raw/train.csv --data-root data/training_raw \
  --encoded outputs/encoded_train --split train --output data/bound_train
python scripts/bind_training_caches.py \
  --manifest data/training_raw/validation.csv --data-root data/training_raw \
  --encoded outputs/encoded_validation --split validation --output data/bound_validation
```

For sharded encoding, supply every shard directory after `--encoded`. The binder requires every manifest sample, preserves the frozen row order, and attaches the original physical-condition tensors to the fresh caches. It uses the text context encoded in the preceding step, so no precomputed prompt-context download is required.

Set `train_cache`, `train_override`, `validation_cache` and `validation_override` in `configs/train_enactphys.json` to the paths printed by the binder. Keep the encoded outputs in place: the cache views use symlinks. Configure the base model, output directory and two-node rendezvous, then launch `scripts/train.py` on both nodes. The training configuration includes validation every 1,000 steps.
