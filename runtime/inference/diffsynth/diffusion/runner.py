import os, json, re, torch, importlib, random, hashlib, math, csv
import torch.distributed as dist
import numpy as np
from dataclasses import dataclass
from pathlib import Path


PHY_PARAM_CONDITION_CONTRACT = "phyparam-canonical-fmueg-v1"
PHY_PARAM_MAX_OBJECTS = 8
_PHY_PARAM_OBJECT_KEYS = {
    "first_frame_masks", "object_valid_mask", "force", "force_present",
    "gravity", "gravity_present", "mu", "mu_present", "restitution",
    "restitution_present", "mass", "mass_present",
}
_PHY_PARAM_REQUIRED_KEYS = _PHY_PARAM_OBJECT_KEYS - {"mass", "mass_present"}
_PHY_PARAM_METADATA_KEYS = {"phyparam_condition_contract", "phyparam_identity"}
_SPARSE_OBJECT_DIM0_KEYS = {
    "first_frame_masks", "object_valid_mask", "force", "force_present",
    "gravity", "gravity_present", "mu", "mu_present", "restitution",
    "restitution_present", "mass", "mass_present"
}
_SPARSE_OBJECT_DIM1_KEYS = {"force_schedule", "object_attention_labels"}
_SPARSE_EDGE_KEYS = {
    "edge_mu",
    "edge_mu_present",
    "edge_restitution",
    "edge_restitution_present",
}
_SPARSE_PAIR_EVENT_KEYS = {
    "contact_label",
    "contact_valid",
    "e_contact_label",
    "e_contact_valid",
    "mu_contact_label",
    "mu_contact_valid",
    "e_pair_event_labels",
    "e_pair_event_valid",
    "mu_pair_event_labels",
    "mu_pair_event_valid",
}
_SPARSE_RUNTIME_PROVENANCE_KEYS = frozenset({
    "event_label_contract",
    "contact_provenance",
    "contact_evidence",
    "entity_first_frame_masks",
    "entity_gravity",
    "entity_gravity_present",
    "entity_kind",
    "entity_roles",
    "entity_valid_mask",
    "object_count",
    "support_count",
})
_SPARSE_DISABLED_CONTROL_KEYS = frozenset({"mass", "mass_present"})


class _EpochWeightedSampler(torch.utils.data.Sampler[int]):
    """Deterministic epoch-indexed replacement sampler, sharded by Accelerate."""

    def __init__(self, weights: torch.Tensor, num_samples: int, seed: int):
        self.weights = weights.detach().to(dtype=torch.float64, device="cpu")
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self.epoch = 0
        if self.weights.ndim != 1 or self.weights.numel() != self.num_samples:
            raise ValueError("weighted sampler rows must equal dataset rows")
        if self.num_samples <= 0 or not torch.isfinite(self.weights).all():
            raise ValueError("weighted sampler requires finite nonempty weights")
        if torch.any(self.weights <= 0):
            raise ValueError("weighted sampler probabilities must be positive")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + self.epoch)
        return iter(
            torch.multinomial(
                self.weights,
                self.num_samples,
                replacement=True,
                generator=generator,
            ).tolist()
        )

    def __len__(self) -> int:
        return self.num_samples


class _EpochTruncatedRandomSampler(torch.utils.data.Sampler[int]):
    """Shuffle without replacement and omit only an incomplete global batch."""

    def __init__(self, num_samples: int, global_batch: int, seed: int):
        self.num_samples = int(num_samples)
        self.global_batch = int(global_batch)
        self.seed = int(seed)
        self.epoch = 0
        if self.num_samples <= 0 or self.global_batch <= 0:
            raise ValueError("exact sampler requires positive rows and global batch")
        self.usable_samples = (
            self.num_samples // self.global_batch
        ) * self.global_batch
        if self.usable_samples <= 0:
            raise ValueError("dataset is smaller than one global batch")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(
            self.num_samples, generator=generator
        )[:self.usable_samples]
        return iter(indices.tolist())

    def __len__(self) -> int:
        return self.usable_samples


class _FixedTruncatedRandomSampler(torch.utils.data.Sampler[int]):
    """One fixed no-replacement subset for repeatable distributed validation."""

    def __init__(self, num_samples: int, world_size: int, seed: int):
        num_samples = int(num_samples)
        world_size = int(world_size)
        if num_samples <= 0 or world_size <= 0:
            raise ValueError("fixed sampler requires positive rows and world size")
        usable_samples = (num_samples // world_size) * world_size
        if usable_samples <= 0:
            raise ValueError("validation dataset is smaller than the world size")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        self.indices = torch.randperm(
            num_samples, generator=generator
        )[:usable_samples].tolist()

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_hierarchical_train_sampler(args, dataset):
    weights_raw = getattr(args, "train_sampler_weights", None)
    receipt_raw = getattr(args, "train_sampler_receipt", None)
    if (weights_raw is None) != (receipt_raw is None):
        raise ValueError("train sampler weights and receipt must be supplied together")
    if weights_raw is None:
        return None, None
    override_raw = getattr(args, "cache_override_manifest", None)
    if override_raw is None:
        raise ValueError("hierarchical sampler requires an authoritative override manifest")
    weights_path = Path(weights_raw).expanduser().resolve()
    receipt_path = Path(receipt_raw).expanduser().resolve()
    override_path = Path(override_raw).expanduser().resolve()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("status") != "complete"
        or receipt.get("contract") != "source-scene-group-hierarchical-sampler-v1"
        or receipt.get("requires_weighted_sampler") is not True
        or receipt.get("rows") != len(dataset)
        or receipt.get("weights_sha256") != _sha256_file(weights_path)
    ):
        raise ValueError("hierarchical sampler receipt differs from training dataset")
    with override_path.open(newline="", encoding="utf-8-sig") as handle:
        override_rows = list(csv.DictReader(handle))
    with weights_path.open(newline="", encoding="utf-8-sig") as handle:
        weight_rows = list(csv.DictReader(handle))
    required = {
        "clip_id", "source", "physical_scene_id", "group_id", "sampling_probability"
    }
    if len(override_rows) != len(dataset) or len(weight_rows) != len(dataset):
        raise ValueError("sampler/override row count differs from dataset")
    if weight_rows and not required <= set(weight_rows[0]):
        raise ValueError("sampler weights columns differ")
    override_ids = [str(row.get("clip_id", "")) for row in override_rows]
    weight_ids = [str(row.get("clip_id", "")) for row in weight_rows]
    if not all(override_ids) or override_ids != weight_ids or len(set(weight_ids)) != len(weight_ids):
        raise ValueError("sampler weights are not one-to-one and ordered with the dataset")
    probabilities = torch.tensor(
        [float(row["sampling_probability"]) for row in weight_rows], dtype=torch.float64
    )
    probability_sum = float(probabilities.sum().item())
    if not math.isclose(probability_sum, 1.0, rel_tol=0.0, abs_tol=2e-12):
        raise ValueError(f"sampler probability sum differs: {probability_sum}")
    source_mass = {}
    for probability, row in zip(probabilities.tolist(), weight_rows):
        source = str(row["source"])
        source_mass[source] = source_mass.get(source, 0.0) + probability
    expected_source = receipt.get("source_probability")
    if set(source_mass) != set(expected_source or {}):
        raise ValueError("sampler source set differs")
    for source, expected in expected_source.items():
        if not math.isclose(source_mass[source], float(expected), rel_tol=0.0, abs_tol=2e-12):
            raise ValueError(f"sampler source mass differs: {source}")
    sampler = _EpochWeightedSampler(probabilities, len(dataset), int(args.seed))
    epoch0 = list(iter(sampler))
    audit = {
        "status": "PASS",
        "contract": receipt["contract"],
        "rows": len(dataset),
        "replacement": True,
        "seed": int(args.seed),
        "override_sha256": _sha256_file(override_path),
        "weights_sha256": _sha256_file(weights_path),
        "receipt_sha256": _sha256_file(receipt_path),
        "probability_sum": probability_sum,
        "source_probability": source_mass,
        "epoch0_global_index_sha256": hashlib.sha256(
            np.asarray(epoch0, dtype=np.int64).tobytes()
        ).hexdigest(),
        "resume_contract": "seed_plus_epoch_then_skip_exact_batch_cursor",
    }
    return sampler, audit


def _validate_phyparam_condition(condition):
    if not isinstance(condition, dict):
        raise TypeError("PhyParam condition must be a mapping")
    keys = set(condition)
    missing = (_PHY_PARAM_REQUIRED_KEYS | _PHY_PARAM_METADATA_KEYS) - keys
    extra = keys - (_PHY_PARAM_OBJECT_KEYS | _PHY_PARAM_METADATA_KEYS)
    if missing or extra:
        raise ValueError(
            f"PhyParam condition schema mismatch; missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )
    if condition["phyparam_condition_contract"] != PHY_PARAM_CONDITION_CONTRACT:
        raise ValueError("PhyParam condition contract differs from the frozen run contract")
    identity = condition["phyparam_identity"]
    required_identity = {
        "source", "raw_clip_id", "clip_id", "video", "video_sha256",
        "source_row_sha256",
    }
    if not isinstance(identity, dict) or set(identity) != required_identity:
        raise ValueError("PhyParam condition identity schema mismatch")
    object_count = None
    for key in sorted(keys & _PHY_PARAM_OBJECT_KEYS):
        value = condition[key]
        if not torch.is_tensor(value) or value.ndim < 1:
            raise TypeError(f"PhyParam condition {key} must be an object-axis tensor")
        if object_count is None:
            object_count = int(value.shape[0])
        elif int(value.shape[0]) != object_count:
            raise ValueError(f"PhyParam condition object axis differs at {key}")
        if not torch.isfinite(value).all():
            raise ValueError(f"PhyParam condition {key} contains non-finite values")
    if object_count is None or not 1 <= object_count <= PHY_PARAM_MAX_OBJECTS:
        raise ValueError(f"PhyParam object count must be in [1,{PHY_PARAM_MAX_OBJECTS}]")
    return object_count


def _collate_phyparam_conditions(conditions):
    padded = []
    identities = []
    for condition in conditions:
        object_count = _validate_phyparam_condition(condition)
        item = {}
        for key in sorted(set(condition) & _PHY_PARAM_OBJECT_KEYS):
            value = condition[key]
            shape = (PHY_PARAM_MAX_OBJECTS, *value.shape[1:])
            output = value.new_zeros(shape)
            output[:object_count].copy_(value)
            item[key] = output
        padded.append(item)
        identities.append(dict(condition["phyparam_identity"]))
    keys = set(padded[0])
    if any(set(item) != keys for item in padded[1:]):
        raise ValueError("optional PhyParam condition fields differ within microbatch")
    output = {key: torch.stack([item[key] for item in padded]) for key in sorted(keys)}
    output["phyparam_condition_contract"] = PHY_PARAM_CONDITION_CONTRACT
    output["phyparam_identity"] = identities
    return output


def _pad_tensor_dimension(tensor, dimension, size):
    if tensor.shape[dimension] == size:
        return tensor
    shape = list(tensor.shape)
    shape[dimension] = size - tensor.shape[dimension]
    return torch.cat((tensor, tensor.new_zeros(shape)), dim=dimension)


def _collate_sparse_object_conditions(conditions):
    """Pad only the object axis for the existing Continuous Writer schema."""
    def validate_shared_contact_condition(condition):
        contact_keys = {"contact_contract", "contact_label", "contact_valid"}
        present = contact_keys & set(condition)
        contact_required = (
            globals().get("os") is not None
            and any(
                os.environ.get(name, "0").strip().lower()
                in {"1", "true", "yes", "on"}
                for name in (
                    "PHYSICAL_WM_SHARED_CONTACT_GATE",
                    "PHYSICAL_WM_THREE_CONTACT_GATES",
                )
            )
        )
        if not present and not contact_required:
            return
        if present != contact_keys:
            raise ValueError(
                f"shared contact condition is incomplete: {sorted(present)}"
            )
        three_contact_required = (
            globals().get("os") is not None
            and os.environ.get("PHYSICAL_WM_THREE_CONTACT_GATES", "0")
            .strip().lower() in {"1", "true", "yes", "on"}
        )
        shared_contact_required = (
            globals().get("os") is not None
            and os.environ.get("PHYSICAL_WM_SHARED_CONTACT_GATE", "0")
            .strip().lower() in {"1", "true", "yes", "on"}
        )
        allowed_contact_contracts = (
            {
                "physical-or-derived-loss-only-contact-occupancy-v2",
                "r4-current-a-three-head-contact-supervision-v1",
                "physical-per-head-loss-only-contact-occupancy-v3",
            }
            if three_contact_required
            else {"simulator-loss-only-shared-contact-occupancy-v1"}
            if shared_contact_required
            else {
                "simulator-loss-only-shared-contact-occupancy-v1",
                "physical-or-derived-loss-only-contact-occupancy-v2",
            }
        )
        if condition["contact_contract"] not in allowed_contact_contracts:
            raise ValueError("contact label contract differs")
        per_head = condition["contact_contract"] == (
            "physical-per-head-loss-only-contact-occupancy-v3"
        )
        per_head_keys = {
            "e_contact_label",
            "e_contact_valid",
            "mu_contact_label",
            "mu_contact_valid",
        }
        present_per_head = per_head_keys & set(condition)
        if per_head and present_per_head != per_head_keys:
            raise ValueError(
                f"per-head contact condition is incomplete: {sorted(present_per_head)}"
            )
        label = condition["contact_label"]
        valid = condition["contact_valid"]
        object_valid = condition.get("object_valid_mask")
        if not all(torch.is_tensor(value) for value in (label, valid, object_valid)):
            raise TypeError("shared contact labels and object_valid_mask must be tensors")
        objects = int(object_valid.shape[0])
        expected = (13, objects, objects + 1)
        if tuple(label.shape) != expected or tuple(valid.shape) != expected:
            raise ValueError(
                f"contact_label/contact_valid must both be {expected}"
            )
        if not torch.isfinite(label).all() or not torch.isfinite(valid).all():
            raise ValueError("shared contact labels must be finite")
        if not torch.all((label == 0) | (label == 1)) or not torch.all(
            (valid == 0) | (valid == 1)
        ):
            raise ValueError("shared contact label and valid must be binary")
        object_label = label[:, :, :objects]
        object_validity = valid[:, :, :objects]
        if not torch.equal(object_label, object_label.transpose(1, 2)):
            raise ValueError("object-object contact_label must be symmetric")
        if not torch.equal(object_validity, object_validity.transpose(1, 2)):
            raise ValueError("object-object contact_valid must be symmetric")
        if torch.any(torch.diagonal(object_validity, dim1=1, dim2=2)):
            raise ValueError("self contact must be invalid")
        if torch.any(label > valid):
            raise ValueError("positive contact must be marked valid")
        for name in (sorted(per_head_keys) if per_head else ()):
            value = condition[name]
            if not torch.is_tensor(value) or tuple(value.shape) != expected:
                raise ValueError(f"{name} must be {expected}")
            if not torch.isfinite(value).all() or not torch.all(
                (value == 0) | (value == 1)
            ):
                raise ValueError(f"{name} must be finite and binary")
        if per_head:
            for prefix in ("e", "mu"):
                head_label = condition[f"{prefix}_contact_label"]
                head_valid = condition[f"{prefix}_contact_valid"]
                if not torch.equal(
                    head_label[:, :, :objects],
                    head_label[:, :, :objects].transpose(1, 2),
                ):
                    raise ValueError(f"object-object {prefix}_contact_label must be symmetric")
                if not torch.equal(
                    head_valid[:, :, :objects],
                    head_valid[:, :, :objects].transpose(1, 2),
                ):
                    raise ValueError(f"object-object {prefix}_contact_valid must be symmetric")
                if torch.any(head_label > head_valid):
                    raise ValueError(f"positive {prefix} contact must be marked valid")

    if not conditions or not all(isinstance(item, dict) for item in conditions):
        raise TypeError("sparse_object_condition batch must contain mappings")
    mass_enabled = os.environ.get("PHYSICAL_WM_OBJECT_MASS", "0") == "1"
    for condition in conditions:
        mass_present = condition.get("mass_present")
        if mass_present is not None:
            if not torch.is_tensor(mass_present):
                raise TypeError("mass_present audit field must be a tensor")
            if not mass_enabled and torch.any(mass_present != 0):
                raise ValueError("mass branch is disabled but mass_present is nonzero")
        if mass_enabled:
            n = int(condition["object_valid_mask"].shape[0])
            for name in ("mass", "mass_present"):
                if name not in condition or not torch.is_tensor(condition[name]) or tuple(condition[name].shape) != (n,):
                    raise ValueError(f"{name} must be a per-object tensor of shape {(n,)}")
    # Pair49 retains this legacy source-contract marker while the newer mixed
    # sources do not.  The contact compiler also retains entity-level audit
    # tensors and zero-only mass placeholders.  None are model inputs or loss
    # targets.  Normalize them away before enforcing the strict runtime schema
    # so variable-cardinality cross-source microbatches remain batchable and
    # the disabled mass branch cannot enter forward.
    conditions = [
        {
            key: value
            for key, value in item.items()
            if key not in _SPARSE_RUNTIME_PROVENANCE_KEYS
            and (mass_enabled or key not in _SPARSE_DISABLED_CONTROL_KEYS)
        }
        for item in conditions
    ]
    runtime_os = globals().get("os")
    three_contact_required = runtime_os is not None and runtime_os.environ.get(
        "PHYSICAL_WM_THREE_CONTACT_GATES", "0"
    ).strip().lower() in {"1", "true", "yes", "on"}
    if three_contact_required:
        upgraded = []
        legacy_contracts = {
            "physical-or-derived-loss-only-contact-occupancy-v2",
            "r4-current-a-three-head-contact-supervision-v1",
        }
        for item in conditions:
            if item.get("contact_contract") in legacy_contracts:
                item = dict(item)
                item["e_contact_label"] = item["contact_label"]
                item["e_contact_valid"] = item["contact_valid"]
                item["mu_contact_label"] = item["contact_label"]
                item["mu_contact_valid"] = item["contact_valid"]
                item["contact_contract"] = (
                    "physical-per-head-loss-only-contact-occupancy-v3"
                )
            upgraded.append(item)
        conditions = upgraded
    for condition in conditions:
        validate_shared_contact_condition(condition)
    keys = set(conditions[0])
    if any(set(item) != keys for item in conditions[1:]):
        raise ValueError("sparse object condition keys differ within a microbatch")
    max_objects = max(int(item["object_valid_mask"].shape[0]) for item in conditions)
    output = {}
    for key in sorted(keys):
        values = [item[key] for item in conditions]
        if key in _SPARSE_PAIR_EVENT_KEYS:
            padded_pairs = []
            for value in values:
                if value.ndim != 3 or value.shape[0] != 13:
                    raise ValueError(f"{key} must be [13,N,N+1]")
                objects = int(value.shape[1])
                if tuple(value.shape) != (13, objects, objects + 1):
                    raise ValueError(f"{key} must be [13,N,N+1]")
                padded = value.new_zeros(13, max_objects, max_objects + 1)
                padded[:, :objects, :objects] = value[:, :, :objects]
                padded[:, :objects, max_objects] = value[:, :, objects]
                padded_pairs.append(padded)
            output[key] = torch.stack(padded_pairs)
        elif key in _SPARSE_EDGE_KEYS:
            padded_edges = []
            for value in values:
                objects = int(value.shape[0])
                if tuple(value.shape) != (objects, objects + 1):
                    raise ValueError(f"{key} must be [N,N+1]")
                padded = value.new_zeros(max_objects, max_objects + 1)
                padded[:objects, :objects] = value[:, :objects]
                # The support token is always the final partner column. Keep
                # it final after variable-cardinality object padding.
                padded[:objects, max_objects] = value[:, objects]
                padded_edges.append(padded)
            output[key] = torch.stack(padded_edges)
        elif key in _SPARSE_OBJECT_DIM0_KEYS:
            output[key] = torch.stack(
                [_pad_tensor_dimension(value, 0, max_objects) for value in values]
            )
        elif key in _SPARSE_OBJECT_DIM1_KEYS:
            output[key] = torch.stack(
                [_pad_tensor_dimension(value, 1, max_objects) for value in values]
            )
        elif torch.is_tensor(values[0]):
            if not all(torch.is_tensor(value) and value.shape == values[0].shape for value in values):
                raise ValueError(f"cannot batch sparse condition tensor {key}")
            output[key] = torch.stack(values)
        else:
            if any(value != values[0] for value in values[1:]):
                raise ValueError(f"sparse condition scalar {key} differs within batch")
            output[key] = values[0]
    return output


_PREENCODED_FORWARD_OVERRIDE_KEYS = frozenset({
    "impact_loss_weight",
    "object_attention_loss_weight",
    "reader_supervision_mode",
    "writer_loss_weight",
    "writer_centroid_loss_weight",
    "e_pair_event_loss_weight",
    "mu_pair_event_loss_weight",
    "contact_loss_weight",
    "base_contact_loss_weight",
    "e_contact_loss_weight",
    "mu_contact_loss_weight",
    "dynamic_loss_weight",
    "temporal_loss_weight",
    "phyparam_feature_loss_weight",
    "phyparam_temporal_loss_weight",
})


def _collate_preencoded_mapping(mappings):
    """Batch cached Wan inputs without trying to tensorize inert PIL payloads."""
    # These controls are replaced from the frozen run arguments immediately in
    # WanTrainingModule.forward.  Older R4 and newer Direction-CF caches do not
    # carry the same subset, so remove stale per-cache copies before enforcing
    # the otherwise strict microbatch schema.
    mappings = [
        {key: value for key, value in item.items() if key not in _PREENCODED_FORWARD_OVERRIDE_KEYS}
        for item in mappings
    ]
    keys = set(mappings[0])
    if any(set(item) != keys for item in mappings[1:]):
        raise ValueError("preencoded cache keys differ within a microbatch")
    output = {}
    for key in sorted(keys):
        values = [item[key] for item in mappings]
        first = values[0]
        if torch.is_tensor(first):
            if not all(torch.is_tensor(value) and value.shape == first.shape for value in values):
                raise ValueError(f"preencoded tensor shape differs at {key}")
            # Cached Wan tensors already carry a singleton batch dimension.
            output[key] = (
                torch.cat(values, dim=0)
                if first.ndim > 0 and first.shape[0] == 1
                else torch.stack(values, dim=0)
            )
            continue
        try:
            identical = all(bool(value == first) for value in values[1:])
        except (TypeError, ValueError):
            identical = False
        # Encoded tensors make prompt/source PIL payloads unreachable in
        # sft:train. Preserve every row rather than selecting row zero.
        output[key] = first if identical else values
    return output


def _collate_training_samples(batch):
    """Collate true microbatches with fixed object padding and DINO provenance."""
    if not batch:
        raise ValueError("cannot collate an empty training batch")
    stripped = []
    dino_targets = []
    conditions = []
    for sample in batch:
        sample_copy = list(sample)
        shared = dict(sample_copy[0])
        dino_targets.append(shared.pop("phyparam_dino_features", None))
        conditions.append(shared.pop("sparse_object_condition", None))
        if os.environ.get("PHYSICAL_WM_TEXT_ONLY", "0") == "1":
            conditions[-1] = None
        sample_copy[0] = shared
        stripped.append(tuple(sample_copy))
    if not all(len(sample) == 3 and all(isinstance(item, dict) for item in sample) for sample in stripped):
        raise TypeError("training microbatch requires preencoded 3-item Wan tuples")
    collated = (
        _collate_preencoded_mapping([sample[0] for sample in stripped]),
        _collate_preencoded_mapping([sample[1] for sample in stripped]),
        _collate_preencoded_mapping([sample[2] for sample in stripped]),
    )
    if any(condition is not None for condition in conditions):
        if not all(isinstance(condition, dict) for condition in conditions):
            raise TypeError("mixed or non-mapping sparse conditions in microbatch")
        phyparam_flags = ["phyparam_condition_contract" in condition for condition in conditions]
        if any(phyparam_flags) and not all(phyparam_flags):
            raise ValueError("PhyParam and Continuous Writer conditions cannot share a microbatch")
        collated[0]["sparse_object_condition"] = (
            _collate_phyparam_conditions(conditions)
            if all(phyparam_flags)
            else _collate_sparse_object_conditions(conditions)
        )
    if any(target is not None for target in dino_targets):
        if not all(isinstance(target, dict) for target in dino_targets):
            raise TypeError("mixed or non-mapping PhyParam DINO targets in microbatch")
        first = dino_targets[0]
        provenance_keys = (
            "teacher_model_sha256", "feature_contract", "decoded_frames",
            "feature_frame_indices", "training_spatial_preprocess", "spatial_shape",
        )
        for target in dino_targets[1:]:
            drift = [key for key in provenance_keys if target.get(key) != first.get(key)]
            if drift:
                raise ValueError(f"PhyParam DINO provenance differs within microbatch: {drift}")
        payload = dict(first)
        payload["features"] = torch.stack([target["features"] for target in dino_targets])
        collated[0]["phyparam_dino_features"] = payload
    return collated


def _prepare_cached_validation_sample(data, *, contact_required):
    """Batch and upgrade one cached sparse condition before validation forward."""
    if not isinstance(data, (tuple, list)) or len(data) != 3:
        raise TypeError("cached validation sample must be a three-item Wan tuple")
    sample = list(data)
    shared = dict(sample[0])
    if os.environ.get("PHYSICAL_WM_TEXT_ONLY", "0") == "1":
        shared.pop("sparse_object_condition", None)
    condition = shared.get("sparse_object_condition")
    if condition is None:
        if contact_required:
            raise ValueError("contact validation sample has no sparse condition")
    elif not isinstance(condition, dict):
        raise TypeError("validation sparse_object_condition must be a mapping")
    else:
        # Compatibility bridge from legacy shared-contact validation conditions
        # to the active base/e/mu three-head schema.
        shared["sparse_object_condition"] = _collate_sparse_object_conditions(
            [condition]
        )
    sample[0] = shared
    return tuple(sample)


def _allreduce_offload_gradients(accelerator, model, bucket_bytes=64 << 20):
    """Average trainable gradients before model/optimizer CPU offload.

    In the full-depth path the model itself is intentionally not wrapped by
    DDP, because individual layers are streamed from CPU.  This function is
    therefore the exact distributed synchronization point.
    """
    if not accelerator.sync_gradients or int(accelerator.num_processes) == 1:
        return
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("distributed gradient synchronization is not initialized")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("no trainable parameters to synchronize")
    device = next((parameter.grad.device for parameter in parameters if parameter.grad is not None), None)
    if device is None:
        raise RuntimeError("all trainable gradients are absent")
    presence = torch.tensor(
        [int(parameter.grad is not None) for parameter in parameters],
        dtype=torch.int32,
        device=device,
    )
    dist.all_reduce(presence, op=dist.ReduceOp.SUM)
    world = int(accelerator.num_processes)
    inconsistent = ((presence != 0) & (presence != world)).nonzero().flatten().tolist()
    if inconsistent:
        raise RuntimeError(
            f"trainable gradient presence differs across ranks at parameter indices {inconsistent[:8]}"
        )
    gradients = [
        parameter.grad for parameter, count in zip(parameters, presence.tolist()) if count == world
    ]
    grouped = {}
    for gradient in gradients:
        if gradient.device != device:
            raise RuntimeError("trainable gradients span devices before offload synchronization")
        grouped.setdefault((gradient.device, gradient.dtype), []).append(gradient)
    for group in grouped.values():
        bucket = []
        size = 0
        for gradient in group:
            item_bytes = gradient.numel() * gradient.element_size()
            if bucket and size + item_bytes > bucket_bytes:
                flat = torch._utils._flatten_dense_tensors(bucket)
                dist.all_reduce(flat, op=dist.ReduceOp.SUM)
                flat.div_(world)
                for target, averaged in zip(bucket, torch._utils._unflatten_dense_tensors(flat, bucket)):
                    target.copy_(averaged)
                bucket, size = [], 0
            bucket.append(gradient)
            size += item_bytes
        if bucket:
            flat = torch._utils._flatten_dense_tensors(bucket)
            dist.all_reduce(flat, op=dist.ReduceOp.SUM)
            flat.div_(world)
            for target, averaged in zip(bucket, torch._utils._unflatten_dense_tensors(flat, bucket)):
                target.copy_(averaged)


def _actual_microbatch_size(data):
    shared = data[0] if isinstance(data, (tuple, list)) else data
    latents = shared.get("input_latents") if isinstance(shared, dict) else None
    if torch.is_tensor(latents) and latents.ndim == 5:
        return int(latents.shape[0])
    dino = shared.get("phyparam_dino_features") if isinstance(shared, dict) else None
    features = dino.get("features") if isinstance(dino, dict) else None
    if torch.is_tensor(features) and features.ndim == 5:
        return int(features.shape[0])
    condition = shared.get("sparse_object_condition") if isinstance(shared, dict) else None
    masks = condition.get("first_frame_masks") if isinstance(condition, dict) else None
    if torch.is_tensor(masks) and masks.ndim == 4:
        return int(masks.shape[0])
    return 1


def _write_batch_contract_rank(accelerator, output_path, args, actual, step):
    root = Path(output_path) / "batch_contract"
    root.mkdir(parents=True, exist_ok=True)
    peak = (
        int(torch.cuda.max_memory_allocated(accelerator.device))
        if torch.cuda.is_available() else 0
    )
    payload = {
        "rank": int(accelerator.process_index),
        "world_size": int(accelerator.num_processes),
        "configured_microbatch_per_rank": int(args.train_batch_size),
        "actual_forward_microbatch_per_rank": int(actual),
        "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
        "global_effective_batch": int(actual) * int(args.gradient_accumulation_steps) * int(accelerator.num_processes),
        "observed_optimizer_step": int(step),
        "peak_memory_allocated_bytes": peak,
    }
    temporary = root / f".rank-{accelerator.process_index:02d}.json.part"
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, root / f"rank-{accelerator.process_index:02d}.json")
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list
from .training_module import DiffusionTrainingModule
from .logger import ModelLogger
from diffsynth.core import OffloadTrainingManager


def _trainable_state_checksum(state: dict) -> dict:
    """Deterministic numeric fingerprint of a trainable-parameter state dict.

    Used to numerically prove a resumed run's in-memory trainable weights are
    identical to what was written to a specific checkpoint on disk, instead of
    trusting log text alone (smoke/check_resume_verify_log.py compares the
    fingerprint written here at save time against the one written right after
    `accelerator.load_state` on resume).

    - `sha256`: exact byte-level digest over every tensor (float64-canonicalized,
      sorted by parameter name for order-independence) -- an exact-match check.
    - `l2_norm` / `num_tensors`: human-readable magnitude for the log/report,
      not the primary equality check (float reduction order can jitter it at
      the ULP level across devices; the sha256 digest is the authority).
    """
    hasher = hashlib.sha256()
    sq_sum = 0.0
    for name in sorted(state):
        arr = state[name].detach().to(dtype=torch.float64, device="cpu").contiguous().numpy()
        hasher.update(name.encode("utf-8"))
        hasher.update(arr.tobytes())
        sq_sum += float(np.square(arr).sum())
    return {
        "sha256": hasher.hexdigest(),
        "l2_norm": math.sqrt(sq_sum),
        "num_tensors": len(state),
    }


@dataclass
class TrainingCursor:
    """Serializable position in the deterministic shuffled data stream."""

    global_step: int = 0
    epoch: int = 0
    batch_in_epoch: int = 0

    def state_dict(self):
        return {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "batch_in_epoch": self.batch_in_epoch,
        }

    def load_state_dict(self, state):
        self.global_step = int(state["global_step"])
        self.epoch = int(state["epoch"])
        self.batch_in_epoch = int(state["batch_in_epoch"])


def _resolve_complete_checkpoint(output_path: str, requested: str | None) -> Path | None:
    if requested is None or os.path.isfile(requested):
        return None
    output = Path(output_path)
    if requested == "latest":
        candidates = []
        for path in output.glob("checkpoint-step-*"):
            match = re.fullmatch(r"checkpoint-step-(\d+)", path.name)
            if match and (path / "complete.json").is_file():
                candidates.append((int(match.group(1)), path))
        if not candidates:
            raise FileNotFoundError(f"no complete checkpoint found under {output}")
        return max(candidates)[1]
    path = Path(requested).expanduser().resolve()
    if not path.is_dir() or not (path / "complete.json").is_file():
        raise ValueError(f"resume checkpoint is absent or incomplete: {path}")
    return path


def _register_trainable_only_state_hooks(
    accelerator: Accelerator,
    checkpoint_model: torch.nn.Module | None = None,
) -> None:
    """Keep frozen Wan weights out of full-state checkpoints."""

    def save_hook(models, weights, output_dir):
        if models or weights:
            if len(models) != 1 or len(weights) != 1:
                raise RuntimeError("trainable-only checkpointing expects exactly one prepared model")
            model = accelerator.unwrap_model(models[0])
            full_state = weights[0]
        elif checkpoint_model is not None:
            model = checkpoint_model
            full_state = dict(model.named_parameters())
        else:
            raise RuntimeError("checkpoint hook has neither a prepared nor bound model")
        trainable_names = model.trainable_param_names()
        trainable_state = {
            name: value.detach()
            for name, value in full_state.items()
            if name in trainable_names
        }
        if set(trainable_state) != trainable_names:
            missing = sorted(trainable_names - set(trainable_state))
            raise RuntimeError(f"could not export all trainable parameters: {missing[:8]}")
        # Save a parameter checksum alongside the weights for resume verification.
        # The global main process writes the shared file atomically.
        if accelerator.is_main_process:
            checksum = _trainable_state_checksum(trainable_state)
            checksum_path = Path(output_dir) / "trainable_checksum.json"
            checksum_temporary = Path(output_dir) / ".trainable_checksum.json.new"
            checksum_temporary.write_text(
                json.dumps({"source": "save_hook", **checksum}, indent=2),
                encoding="utf-8",
            )
            os.replace(checksum_temporary, checksum_path)
        accelerator.save(
            trainable_state,
            Path(output_dir) / "trainable_model.safetensors",
            safe_serialization=True,
        )
        if weights:
            weights.clear()

    def load_hook(models, input_dir):
        if models:
            if len(models) != 1:
                raise RuntimeError("trainable-only checkpointing expects exactly one prepared model")
            model = accelerator.unwrap_model(models.pop())
        elif checkpoint_model is not None:
            model = checkpoint_model
        else:
            raise RuntimeError("checkpoint hook has neither a prepared nor bound model")
        from safetensors.torch import load_file
        path = Path(input_dir) / "trainable_model.safetensors"
        if not path.is_file():
            raise FileNotFoundError(path)
        state = load_file(str(path), device="cpu")
        expected = model.trainable_param_names()
        if set(state) != expected:
            missing = sorted(expected - set(state))
            extra = sorted(set(state) - expected)
            raise RuntimeError(
                f"trainable checkpoint key mismatch; missing={missing[:8]}, extra={extra[:8]}"
            )
        incompatible = model.load_state_dict(state, strict=False)
        if incompatible.unexpected_keys:
            raise RuntimeError(f"unexpected trainable keys: {incompatible.unexpected_keys}")

    accelerator.register_save_state_pre_hook(save_hook)
    accelerator.register_load_state_pre_hook(load_hook)


def _save_complete_checkpoint(
    accelerator: Accelerator,
    output_path: str,
    cursor: TrainingCursor,
) -> Path:
    output = Path(output_path)
    final_path = output / f"checkpoint-step-{cursor.global_step:08d}"
    temporary_path = output / f"checkpoint-step-{cursor.global_step:08d}.incomplete"
    accelerator.wait_for_everyone()
    collision = [None]
    if accelerator.is_main_process:
        collision[0] = final_path.exists() or temporary_path.exists()
    broadcast_object_list(collision)
    if collision[0]:
        raise FileExistsError(
            f"refusing to overwrite checkpoint target: {final_path} or {temporary_path}"
        )
    accelerator.save_state(str(temporary_path), safe_serialization=True)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        cursor_payload = cursor.state_dict()
        (temporary_path / "cursor.json").write_text(
            json.dumps(cursor_payload, indent=2),
            encoding="utf-8",
        )
        checkpoint_files = []
        for path in sorted(temporary_path.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(temporary_path).as_posix()
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                    digest.update(chunk)
            checkpoint_files.append(
                {
                    "path": relative,
                    "size": path.stat().st_size,
                    "sha256": digest.hexdigest(),
                }
            )
        checkpoint_manifest = {
            "contract": "diffsynth-trainable-complete-checkpoint-v2",
            "world_size": accelerator.num_processes,
            **cursor_payload,
            "files": checkpoint_files,
        }
        checkpoint_manifest_path = temporary_path / "checkpoint_manifest.json"
        checkpoint_manifest_path.write_text(
            json.dumps(checkpoint_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest_digest = hashlib.sha256(checkpoint_manifest_path.read_bytes()).hexdigest()
        (temporary_path / "complete.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "contract": "diffsynth-trainable-complete-checkpoint-v2",
                    "world_size": accelerator.num_processes,
                    "checkpoint_manifest_sha256": manifest_digest,
                    **cursor_payload,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.rename(temporary_path, final_path)
    accelerator.wait_for_everyone()
    return final_path


def get_optimizer_class(customized_optimizer=None):
    if customized_optimizer is None:
        return torch.optim.AdamW
    else:
        module_name, class_name = customized_optimizer.rsplit(".", 1)
        module = importlib.import_module(module_name)
        print(f"Customized opimizer `{customized_optimizer}` imported.")
        return getattr(module, class_name)


def _optimizer_param_groups(
    model: DiffusionTrainingModule,
    learning_rate: float,
    lora_lr_scale: float,
    erase_gate_lr_multiplier: float = 1.0,
):
    if not 0.0 < lora_lr_scale <= 1.0:
        raise ValueError(f"lora_lr_scale must be in (0, 1], got {lora_lr_scale}")
    new_parameters = []
    lora_parameters = []
    erase_gate_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if ".erase_gate." in name:
            erase_gate_parameters.append(parameter)
        elif ".lora_A." in name or ".lora_B." in name:
            lora_parameters.append(parameter)
        else:
            new_parameters.append(parameter)
    if os.environ.get("PHYSICAL_WM_TEXT_ONLY", "0") == "1":
        if new_parameters or erase_gate_parameters or not lora_parameters:
            raise RuntimeError("text-only optimizer requires exclusively LoRA parameters")
        return [{"params": lora_parameters, "lr": learning_rate, "group_name": "wan_lora"}]
    if not new_parameters:
        raise RuntimeError("no non-LoRA trainable condition parameters found")
    groups = [
        {
            "params": new_parameters,
            "lr": learning_rate,
            "group_name": "new_condition_modules",
        }
    ]
    if lora_parameters:
        groups.append({
            "params": lora_parameters,
            "lr": learning_rate * lora_lr_scale,
            "group_name": "wan_lora",
        })
    if erase_gate_parameters:
        if not 1.0 <= erase_gate_lr_multiplier <= 10.0:
            raise ValueError(
                "erase gate LR multiplier must be in [1, 10], got "
                f"{erase_gate_lr_multiplier}"
            )
        groups.append({
            "params": erase_gate_parameters,
            "lr": learning_rate * erase_gate_lr_multiplier,
            "group_name": "erase_then_write_gate",
        })
    return groups


def _write_first_gradient_audit(
    accelerator: Accelerator,
    model: DiffusionTrainingModule,
    output_path: str,
) -> None:
    """Prove that the first synchronized backward stays inside LoRA and the force interface."""
    unwrapped = accelerator.unwrap_model(model)
    trainable = []
    gradients = []
    nonzero_gradients = []
    frozen_with_gradients = []
    for name, parameter in unwrapped.named_parameters():
        if parameter.requires_grad:
            trainable.append(name)
        if parameter.grad is not None:
            gradients.append(name)
            if torch.count_nonzero(parameter.grad.detach()).item() > 0:
                nonzero_gradients.append(name)
            if not parameter.requires_grad:
                frozen_with_gradients.append(name)
    # Tuple, not a single string: str.startswith() accepts a tuple of
    # prefixes natively. Both the F cross-attention path and the independent
    # GlobalPhysicsCrossAttn path are legitimate trainable-condition prefixes;
    # without global_physics_ here, this audit would spuriously raise
    # RuntimeError the moment physics trainable params exist (they are
    # trainable + get gradients, but would fail `allowed(name)`).
    condition_prefix = (
        "pipe.dit.force_condition_",
        "pipe.dit.global_physics_",
        "pipe.dit.oracle_slot_adapter.",
        "pipe.dit.object_time_graph_adapter.",
        "pipe.dit.sparse_object_adapter.",
        "pipe.dit.phyparam_control_dit.",
    )
    oracle_route = any(
        name.startswith("pipe.dit.oracle_slot_adapter.") for name in trainable
    )
    object_time_graph_route = any(
        name.startswith("pipe.dit.object_time_graph_adapter.")
        for name in trainable
    )
    sparse_object_route = any(
        name.startswith("pipe.dit.sparse_object_adapter.") for name in trainable
    )
    phyparam_route = any(
        name.startswith("pipe.dit.phyparam_control_dit.") for name in trainable
    )

    def allowed(name: str) -> bool:
        return (
            name.startswith(condition_prefix)
            or ".lora_A." in name
            or ".lora_B." in name
        )

    condition_nonzero = [
        name for name in nonzero_gradients if name.startswith(condition_prefix)
    ]
    lora_nonzero = [
        name for name in nonzero_gradients
        if ".lora_A." in name or ".lora_B." in name
    ]
    passed = bool(
        trainable
        and gradients
        and nonzero_gradients
        and (condition_nonzero or (os.environ.get("PHYSICAL_WM_TEXT_ONLY", "0") == "1" and lora_nonzero))
        and (
            oracle_route or object_time_graph_route or sparse_object_route
            or phyparam_route or lora_nonzero
        )
        and all(allowed(name) for name in trainable)
        and all(allowed(name) for name in gradients)
        and not frozen_with_gradients
    )
    if not passed:
        raise RuntimeError(
            "gradient boundary failed: "
            f"trainable={trainable[:5]} gradients={gradients[:5]} "
            f"nonzero={nonzero_gradients[:5]} frozen_grad={frozen_with_gradients[:5]}"
        )
    if accelerator.is_main_process:
        payload = {
            "passed": True,
            "trainable_parameter_count": len(trainable),
            "parameters_with_gradient_count": len(gradients),
            "parameters_with_nonzero_gradient_count": len(nonzero_gradients),
            "frozen_parameters_with_gradient_count": 0,
            "allowed_condition_prefix": condition_prefix,
            "allowed_lora_parameters": bool(lora_nonzero),
            "condition_parameters_with_nonzero_gradient": condition_nonzero,
            "lora_parameters_with_nonzero_gradient": lora_nonzero,
            "trainable_names": trainable,
            "gradient_names": gradients,
            "nonzero_gradient_names": nonzero_gradients,
        }
        Path(output_path, "gradient_audit.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )


def _update_three_step_gradient_family_audit(
    accelerator: Accelerator,
    model: DiffusionTrainingModule,
    output_path: str,
    step_number: int,
    cumulative_hits: set[str],
    active_condition_families: set[str] | None = None,
) -> set[str]:
    """By step 3, prove gradients propagated past every zero output layer."""
    unwrapped = accelerator.unwrap_model(model)
    names = [name for name, _ in unwrapped.named_parameters()]
    if any("phyparam_control_dit." in name for name in names):
        predicates = {
            "condition_encoder": lambda name: "phyparam_control_dit.condition_encoder." in name,
            "copied_self_attention": lambda name: (
                "phyparam_control_dit.blocks." in name and ".self_attn." in name
            ),
            "restricted_physical_cross_attention": lambda name: (
                "phyparam_control_dit.blocks." in name and ".cross_attn." in name
            ),
            "control_ffn": lambda name: (
                "phyparam_control_dit.blocks." in name and ".ffn." in name
            ),
            "dense_residual_projection": lambda name: (
                "phyparam_control_dit.output_projections." in name
            ),
        }
        branch = unwrapped.pipe.dit.phyparam_control_dit
        if getattr(branch, "feature_supervision_enabled", False):
            predicates["dino_projection_heads"] = lambda name: (
                "phyparam_control_dit.feature_supervisor.heads." in name
            )
        route = "phyparam_paper_reimplementation"
    elif any("sparse_object_adapter." in name for name in names):
        temporal_route = getattr(
            getattr(unwrapped, "pipe", None).dit
            if getattr(unwrapped, "pipe", None) is not None
            else None,
            "sparse_object_architecture",
            None,
        ) in {
            "temporal", "temporal_schedule",
            "temporal_no_interaction_schedule",
            "temporal_no_interaction_schedule_mugate",
            "temporal_no_interaction_schedule_mugate_decoupled_writer",
            "temporal_no_interaction_schedule_mugate_decoupled_writer_continuous",
            "temporal_independent_edge_schedule_decoupled_writer_continuous",
            "temporal_no_interaction_schedule_trackprev_mugate",
            "temporal_no_interaction_schedule_ictr",
            "temporal_pair_schedule",
        }
        if temporal_route:
            architecture = getattr(
                getattr(unwrapped, "pipe", None).dit
                if getattr(unwrapped, "pipe", None) is not None
                else None,
                "sparse_object_architecture",
                None,
            )
            pair_route = architecture == "temporal_pair_schedule"
            independent_edge_route = architecture == (
                "temporal_independent_edge_schedule_decoupled_writer_continuous"
            )
            sparse_adapter = getattr(
                getattr(
                    getattr(unwrapped, "pipe", None), "dit", None
                ),
                "sparse_object_adapter",
                None,
            )
            predicates = {
                "parallel_locator": lambda name: (
                    "sparse_object_adapter.groups." in name and ".router." in name
                ),
                "position_encoder": lambda name: ".router.position_encoder." in name,
                **(
                    {
                        "independent_writer": lambda name: (
                            "sparse_object_adapter.groups." in name
                            and ".writer." in name
                        )
                    }
                    if getattr(
                        getattr(unwrapped, "pipe", None).dit
                        if getattr(unwrapped, "pipe", None) is not None
                        else None,
                        "sparse_object_architecture",
                        None,
                    )
                    in {
                        "temporal_no_interaction_schedule_mugate_decoupled_writer",
                        "temporal_no_interaction_schedule_mugate_decoupled_writer_continuous",
                        "temporal_independent_edge_schedule_decoupled_writer_continuous",
                    }
                    else {}
                ),
                **(
                    {
                        "continuous_state_fusion": lambda name: (
                            "sparse_object_adapter.groups." in name
                            and ".state_fusion_gate." in name
                        )
                    }
                    if getattr(
                        getattr(unwrapped, "pipe", None).dit
                        if getattr(unwrapped, "pipe", None) is not None
                        else None,
                        "sparse_object_architecture",
                        None,
                    )
                    in {
                        "temporal_no_interaction_schedule_mugate_decoupled_writer_continuous",
                        "temporal_independent_edge_schedule_decoupled_writer_continuous",
                    }
                    else {}
                ),
                **(
                    {
                        "bounded_previous_frame_mix": lambda name: name.endswith(
                            ".router.tracking_mix_logit"
                        )
                    }
                    if getattr(
                        getattr(unwrapped, "pipe", None).dit
                        if getattr(unwrapped, "pipe", None) is not None
                        else None,
                        "sparse_object_architecture",
                        None,
                    )
                    == "temporal_no_interaction_schedule_ictr"
                    else {}
                ),
                "force_residual": lambda name: "condition_encoder.force." in name,
                "gravity_residual": lambda name: "condition_encoder.gravity." in name,
                "same_object_temporal_attention": lambda name: ".temporal.attention." in name,
                "background_pool": lambda name: any(
                    token in name for token in (
                        ".background_queries", ".background_value."
                    )
                ),
                "object_interaction": lambda name: ".interaction." in name,
                **(
                    {
                        "pair_condition_encoder": lambda name: (
                            ".interaction.condition_encoder." in name
                        ),
                        "pair_response": lambda name: (
                            ".interaction.pair_response." in name
                        ),
                    }
                    if pair_route
                    else (
                        {
                            **(
                                {
                                    "base_contact_head": lambda name: (
                                        ".interaction.base_contact_head." in name
                                    ),
                                    "e_contact_head": lambda name: (
                                        ".interaction.e_contact_head." in name
                                    ),
                                    "mu_contact_head": lambda name: (
                                        ".interaction.mu_contact_head." in name
                                    ),
                                }
                                if getattr(sparse_adapter, "three_contact_gates", False)
                                else {
                                    "independent_edge_gate": lambda name: any(
                                        token in name
                                        for token in (
                                            ".interaction.query_projection.",
                                            ".interaction.key_projection.",
                                        )
                                    ),
                                }
                            ),
                            "independent_edge_base_message": lambda name: any(
                                token in name
                                for token in (
                                    ".interaction.sender_projection.",
                                    ".interaction.receiver_projection.",
                                    ".interaction.base_output.",
                                )
                            ),
                            "independent_edge_e_adaln": lambda name: (
                                ".interaction.e_adaln." in name
                            ),
                            "independent_edge_e_output": lambda name: (
                                ".interaction.e_output." in name
                            ),
                            "independent_edge_mu_adaln": lambda name: (
                                ".interaction.mu_adaln." in name
                            ),
                            "independent_edge_mu_output": lambda name: (
                                ".interaction.mu_output." in name
                            ),
                            **(
                                {
                                    "shared_contact_head": lambda name: (
                                        ".interaction.e_pair_event_head." in name
                                    ),
                                }
                                if getattr(sparse_adapter, "shared_contact_gate", False)
                                else {
                                    "e_pair_event_head": lambda name: (
                                        ".interaction.e_pair_event_head." in name
                                    ),
                                    "mu_pair_event_head": lambda name: (
                                        ".interaction.mu_pair_event_head." in name
                                    ),
                                }
                                if getattr(sparse_adapter, "pair_event_supervision", False)
                                else {}
                            ),
                        }
                        if independent_edge_route
                        else {
                        "e_modulation": lambda name: (
                            ".interaction.e_modulation." in name
                        ),
                        "mu_modulation": lambda name: (
                            ".interaction.mu_modulation." in name
                        ),
                        }
                    )
                ),
                "sparse_writeback": lambda name: ".output_projection." in name,
            }
            route = (
                "sparse_object_interaction_independent_undirected_edge_v1"
                if independent_edge_route
                else (
                    "sparse_object_interaction_temporal_pair_schedule_v1"
                    if pair_route
                    else "sparse_object_interaction_temporal_v1"
                )
            )
            skipped_inactive_families = []
            if independent_edge_route and active_condition_families is not None:
                condition_families = {
                    "force_residual",
                    "gravity_residual",
                    "independent_edge_e_adaln",
                    "independent_edge_e_output",
                    "independent_edge_mu_adaln",
                    "independent_edge_mu_output",
                }
                skipped_inactive_families = sorted(
                    condition_families - active_condition_families
                )
                predicates = {
                    family: predicate
                    for family, predicate in predicates.items()
                    if family not in skipped_inactive_families
                }
        else:
            predicates = {
                "parallel_locator": lambda name: (
                    "sparse_object_adapter.groups." in name and ".router." in name
                ),
                "position_encoder": lambda name: ".router.position_encoder." in name,
                "force_residual": lambda name: "condition_encoder.force." in name,
                "mu_residual": lambda name: "condition_encoder.mu." in name,
                "e_residual": lambda name: "condition_encoder.restitution." in name,
                "temporal_pool": lambda name: ".temporal_score." in name,
                "background_pool": lambda name: any(
                    token in name for token in (
                        ".background_queries", ".background_key.", ".background_value."
                    )
                ),
                "object_interaction": lambda name: ".interaction." in name,
                "time_decoder": lambda name: ".time_decoder." in name,
                "sparse_writeback": lambda name: ".output_projection." in name,
            }
            route = "sparse_object_interaction_v2"
    elif any("object_time_graph_adapter." in name for name in names):
        predicates = {
            "learned_routers": lambda name: (
                "object_time_graph_adapter.groups." in name
                and ".router." in name
            ),
            "force_condition_encoder": lambda name: (
                "object_time_graph_adapter.condition_encoder.force_" in name
            ),
            "mu_condition_encoder": lambda name: (
                "object_time_graph_adapter.condition_encoder.mu." in name
            ),
            "e_condition_encoder": lambda name: (
                "object_time_graph_adapter.condition_encoder.restitution." in name
            ),
            "temporal_modulation": lambda name: any(
                token in name
                for token in (
                    ".temporal_attention.",
                    ".mu_modulation.",
                    ".force_modulation.",
                    ".force_kick.",
                )
            ),
            "object_ffn": lambda name: (
                "object_time_graph_adapter.groups." in name
                and ".ffn." in name
            ),
            "contact_heads": lambda name: ".contact_head." in name,
            "e_relations": lambda name: any(
                token in name
                for token in (
                    ".relation_encoder.",
                    ".e_modulation.",
                    ".relation_ffn.",
                    ".relation_to_a.",
                    ".relation_to_b.",
                )
            ),
            "learned_writeback": lambda name: ".output_projection." in name,
        }
        route = "object_time_graph"
    elif any("oracle_slot_adapter." in name for name in names):
        predicates = {
            "slot_input_projections": lambda name: "oracle_slot_adapter.input_projections." in name,
            "typed_local_branches": lambda name: any(
                token in name
                for token in (
                    "force_branch", "gravity_branch", "friction_branch",
                    "force_encoder", "gravity_encoder", "friction_encoder",
                )
            ),
            "carry": lambda name: "oracle_slot_adapter.shared_mixer.carry." in name,
            "impact_head": lambda name: "oracle_slot_adapter.shared_mixer.impact_head." in name,
            "impact_message": lambda name: "oracle_slot_adapter.shared_mixer.impact_value." in name,
            "slot_output_projections": lambda name: "oracle_slot_adapter.output_projections." in name,
        }
        route = "oracle_slot"
    elif any("unified_side_dit." in name for name in names):
        predicates = {
            "side_input_projection": lambda name: "unified_side_dit.condition_projection" in name,
            "side_copied_blocks": lambda name: "unified_side_dit.blocks." in name,
            "side_output_projections": lambda name: "unified_side_dit.output_projections." in name,
            "main_wan_lora": lambda name: ".lora_A." in name or ".lora_B." in name,
        }
        route = "side"
    elif os.environ.get("PHYSICAL_WM_TEXT_ONLY", "0") == "1":
        predicates = {
            "lora_A": lambda name: ".lora_A." in name,
            "lora_B": lambda name: ".lora_B." in name,
        }
        route = "textonly_lora"
    else:
        predicates = {
            "force_input_projection": lambda name: "force_condition_projection" in name,
            "force_attention": lambda name: "force_condition_attention." in name,
            "physics_encoder": lambda name: "global_physics_token_encoder" in name,
            "physics_attention": lambda name: "global_physics_attention." in name,
            "main_wan_lora": lambda name: ".lora_A." in name or ".lora_B." in name,
        }
        route = "cross"
    skipped_inactive_families = locals().get("skipped_inactive_families", [])
    hits = set(cumulative_hits)
    for name, parameter in unwrapped.named_parameters():
        if parameter.grad is None or torch.count_nonzero(parameter.grad.detach()).item() == 0:
            continue
        for family, predicate in predicates.items():
            if predicate(name):
                hits.add(family)
    missing = sorted(set(predicates) - hits)
    passed = step_number >= 3 and not missing
    if accelerator.is_main_process:
        Path(output_path, f"gradient_family_audit_step{step_number}.json").write_text(
            json.dumps(
                {
                    "route": route,
                    "step": step_number,
                    "passed": passed,
                    "cumulative_nonzero_gradient_families": sorted(hits),
                    "missing_families": missing,
                    "skipped_inactive_condition_families": skipped_inactive_families,
                    "note": "zero-output initialization makes upstream gradients expectedly zero at step 1",
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    if step_number == 3 and missing and accelerator.is_main_process:
        print(
            "[gradient-family-audit] non-blocking diagnostic: "
            f"missing={missing}",
            flush=True,
        )
    return hits


def _update_active_gradient_condition_families(
    data,
    cumulative_active: set[str],
) -> set[str]:
    """Track condition families actually present in the first audited batches."""
    presence_to_families = {
        "force_present": ("force_residual",),
        "gravity_present": ("gravity_residual",),
        "edge_restitution_present": (
            "independent_edge_e_adaln",
            "independent_edge_e_output",
        ),
        "edge_mu_present": (
            "independent_edge_mu_adaln",
            "independent_edge_mu_output",
        ),
    }
    shared = data[0] if isinstance(data, (tuple, list)) else data
    if not isinstance(shared, dict):
        raise TypeError("gradient condition audit requires a shared input mapping")
    condition = shared.get("sparse_object_condition")
    if not isinstance(condition, dict):
        raise TypeError("gradient condition audit requires sparse_object_condition")
    if "e_pair_event_valid" in condition:
        presence_to_families["e_pair_event_valid"] = ("e_pair_event_head",)
    if "mu_pair_event_valid" in condition:
        presence_to_families["mu_pair_event_valid"] = ("mu_pair_event_head",)
    missing = sorted(set(presence_to_families) - set(condition))
    if missing:
        raise ValueError(
            f"gradient condition audit is missing presence tensors: {missing}"
        )
    tensors = [condition[key] for key in presence_to_families]
    invalid = [
        key
        for key, value in zip(presence_to_families, tensors)
        if not torch.is_tensor(value)
    ]
    if invalid:
        raise TypeError(
            f"gradient condition audit presence values must be tensors: {invalid}"
        )
    device = next(
        (value.device for value in tensors if torch.is_tensor(value)),
        torch.device("cpu"),
    )
    flags = torch.tensor(
        [
            int(torch.is_tensor(value) and torch.count_nonzero(value).item() > 0)
            for value in tensors
        ],
        dtype=torch.int32,
        device=device,
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(flags, op=dist.ReduceOp.MAX)
    active = set(cumulative_active)
    for is_active, families in zip(flags.cpu().tolist(), presence_to_families.values()):
        if is_active:
            active.update(families)
    return active


def _write_model_contract_audit(
    accelerator: Accelerator,
    model: DiffusionTrainingModule,
    output_path: str,
) -> None:
    """Audit the fully instantiated model, not a hand-computed estimate."""
    named = list(model.named_parameters())
    total = sum(parameter.numel() for _, parameter in named)
    trainable = [(name, parameter) for name, parameter in named if parameter.requires_grad]
    has_oracle_slot = any("oracle_slot_adapter." in name for name, _ in named)
    has_object_time_graph = any(
        "object_time_graph_adapter." in name for name, _ in named
    )
    has_sparse_object = any(
        "sparse_object_adapter." in name for name, _ in named
    )
    has_side = any("unified_side_dit." in name for name, _ in named)
    has_phyparam = any("phyparam_control_dit." in name for name, _ in named)
    if has_phyparam:
        dit = model.pipe.dit
        branch = dit.phyparam_control_dit
        allowed = lambda name: "phyparam_control_dit." in name
        feature_enabled = bool(branch.feature_supervision_enabled)
        route_details = {
            "route": "phyparam_paper_reimplementation",
            "control_blocks": int(branch.num_control_blocks),
            "harmonic_bands": int(branch.condition_encoder.harmonic_bands),
            "max_objects": int(branch.condition_encoder.max_objects),
            "feature_supervision_enabled": feature_enabled,
            "feature_tap_blocks": (
                list(branch.feature_supervisor.tap_blocks) if feature_enabled else []
            ),
            "loss_contract": (
                "flow_matching_plus_dino_feature_cosine_plus_dino_temporal_cosine"
                if feature_enabled else "flow_matching_only_ablation"
            ),
            "future_gt_in_forward": False,
            "backbone_frozen": True,
        }
        route_ok = bool(
            getattr(dit, "phyparam_control_enabled", False)
            and 0 < branch.num_control_blocks <= len(dit.blocks)
            and all(
                not parameter.requires_grad
                for name, parameter in named
                if "phyparam_control_dit." not in name
            )
        )
    elif has_sparse_object:
        dit = model.pipe.dit
        blocks = list(dit.sparse_object_adapter.injection_blocks)
        groups = dit.sparse_object_adapter.groups
        condition_encoder_ids = [id(group.condition_encoder) for group in groups.values()]
        allowed = lambda name: "sparse_object_adapter." in name
        architecture = getattr(dit, "sparse_object_architecture", None)
        no_interaction_route = architecture in {
            "temporal_no_interaction_schedule",
            "temporal_no_interaction_schedule_mugate",
            "temporal_no_interaction_schedule_mugate_decoupled_writer",
            "temporal_no_interaction_schedule_mugate_decoupled_writer_continuous",
            "temporal_no_interaction_schedule_trackprev_mugate",
            "temporal_no_interaction_schedule_ictr",
        }
        independent_edge_route = architecture == (
            "temporal_independent_edge_schedule_decoupled_writer_continuous"
        )
        support_token_route = no_interaction_route or independent_edge_route
        temporal_route = architecture in {
            "temporal",
            "temporal_schedule",
            "temporal_no_interaction_schedule",
            "temporal_no_interaction_schedule_mugate",
            "temporal_no_interaction_schedule_mugate_decoupled_writer",
            "temporal_no_interaction_schedule_mugate_decoupled_writer_continuous",
            "temporal_independent_edge_schedule_decoupled_writer_continuous",
            "temporal_no_interaction_schedule_trackprev_mugate",
            "temporal_no_interaction_schedule_ictr",
            "temporal_pair_schedule",
        }
        route = (
            "sparse_object_interaction_independent_undirected_edge_v1"
            if independent_edge_route
            else (
                "sparse_object_interaction_temporal_pair_schedule_v1"
                if architecture == "temporal_pair_schedule"
                else (
                    "sparse_object_interaction_temporal_no_interaction_schedule_v1"
                    if no_interaction_route
                    else (
                        "sparse_object_interaction_temporal_schedule_v1"
                        if architecture == "temporal_schedule"
                        else (
                            "sparse_object_interaction_temporal_v1"
                            if temporal_route
                            else "sparse_object_interaction_v2"
                        )
                    )
                )
            )
        )
        first_group = next(iter(groups.values()))
        pair_event_supervision = bool(
            getattr(dit.sparse_object_adapter, "pair_event_supervision", False)
        )
        shared_contact_gate = bool(
            getattr(dit.sparse_object_adapter, "shared_contact_gate", False)
        )
        three_contact_gates = bool(
            getattr(dit.sparse_object_adapter, "three_contact_gates", False)
        )
        contact_supervision = bool(
            (
                shared_contact_gate
                and float(getattr(model, "contact_loss_weight", 0.0)) > 0.0
            )
            or (
                three_contact_gates
                and any(
                    float(getattr(model, name, 0.0)) > 0.0
                    for name in (
                        "base_contact_loss_weight",
                        "e_contact_loss_weight",
                        "mu_contact_loss_weight",
                    )
                )
            )
        )
        e_pair_event_loss_mode = getattr(
            dit.sparse_object_adapter, "e_pair_event_loss_mode", "hard_slot"
        )
        object_dim = (
            first_group.interaction.object_dim
            if temporal_route
            else first_group.interaction.attention.embed_dim
        )
        route_details = {
            "route": route,
            "injection_blocks_zero_based": blocks,
            "hard_support_enabled": bool(
                getattr(dit.sparse_object_adapter, "hard_support_enabled", False)
            ),
            "hard_support_loss_weight": float(
                getattr(dit.sparse_object_adapter, "hard_support_loss_weight", 0.0)
            ),
            "correctable_route_carry": bool(
                getattr(
                    dit.sparse_object_adapter,
                    "correctable_route_carry",
                    False,
                )
            ),
            "route_prior_alpha": float(
                getattr(first_group.router, "route_prior_alpha", 0.0)
            ),
            "position_dependent_writer_value": bool(
                getattr(first_group, "position_dependent_writer_value", False)
            ),
            "relative_position_writer_value": bool(
                getattr(first_group, "relative_position_writer_value", False)
            ),
            "relative_position_writer_hidden_dim": (
                int(first_group.relative_position_writer_mlp[0].out_features)
                if getattr(first_group, "relative_position_writer_value", False)
                else None
            ),
            "interaction_blocks_per_group": {key: 1 for key in groups},
            "object_dim": object_dim,
            "background_tokens": (
                1
                if architecture in {
                    "temporal_no_interaction_schedule",
                    "temporal_no_interaction_schedule_mugate",
                    "temporal_no_interaction_schedule_mugate_decoupled_writer",
                    "temporal_no_interaction_schedule_mugate_decoupled_writer_continuous",
                    "temporal_independent_edge_schedule_decoupled_writer_continuous",
                    "temporal_no_interaction_schedule_trackprev_mugate",
                    "temporal_no_interaction_schedule_ictr",
                    "temporal_pair_schedule",
                }
                else 2
            ),
            "null_tokens": (
                1 if no_interaction_route else 0
            ),
            "support_tokens": (
                1 if support_token_route else 0
            ),
            "support_source": (
                "audited_real_support_system_mask_no_background_fallback"
                if support_token_route else None
            ),
            "empty_platform_detection_uses_frame0_mask": (
                support_token_route
            ),
            "empty_platform_fallback_value_source": (
                "implementation_present_but_loader_preflight_unreachable"
                if support_token_route else None
            ),
            "condition_encoder_count": len(condition_encoder_ids),
            "condition_encoders_are_unshared": (
                len(set(condition_encoder_ids)) == len(condition_encoder_ids)
            ),
            "loss_contract": (
                "flow_matching_plus_0.3_object_localization_plus_"
                + (
                    "0.3_writer_routing_plus_"
                    if architecture in {
                        "temporal_no_interaction_schedule_mugate_decoupled_writer",
                        "temporal_no_interaction_schedule_mugate_decoupled_writer_continuous",
                        "temporal_independent_edge_schedule_decoupled_writer_continuous",
                    }
                    else ""
                )
                + "0.2_dynamic_weighted_fm_plus_0.2_temporal_residual"
                + (
                    (
                        "_plus_three_contact_bce_0.1_times_local_active_head_mean_weights_"
                        + str(
                            (
                                float(getattr(model, "base_contact_loss_weight", 0.0)),
                                float(getattr(model, "e_contact_loss_weight", 0.0)),
                                float(getattr(model, "mu_contact_loss_weight", 0.0)),
                            )
                        )
                    )
                    if three_contact_gates
                    else "_plus_shared_contact_bce_weight_"
                    + str(float(getattr(model, "contact_loss_weight", 0.0)))
                    if shared_contact_gate and contact_supervision
                    else (
                        "_plus_0.1_e_pair_event_"
                        + str(e_pair_event_loss_mode)
                        + "_plus_0.1_mu_pair_event_bce"
                        if pair_event_supervision
                        else ""
                    )
                )
            ),
            "future_gt_in_forward": False,
            "pair_event_supervision": pair_event_supervision,
            "shared_contact_gate": shared_contact_gate,
            "three_contact_gates": three_contact_gates,
            "contact_supervision": contact_supervision,
            "e_pair_event_loss_mode": (
                e_pair_event_loss_mode
                if pair_event_supervision else None
            ),
            "pair_event_labels_in_forward": False,
            "pair_event_aggregation": (
                "C_base*M_base+C_e*Phi_e+C_mu*Phi_mu"
                if three_contact_gates
                else "C_ij(t)*(M_base+Phi_e+Phi_mu)"
                if shared_contact_gate
                else (
                    "g_rel*M_base+g_e*Phi_e+g_mu*Phi_mu"
                    if pair_event_supervision
                    else None
                )
            ),
            "e_pair_type": (
                "object-object_and_object-support"
                if pair_event_supervision or shared_contact_gate or three_contact_gates
                else None
            ),
            "mu_pair_type": (
                "endpoint-present-object-object-and-object-support"
                if shared_contact_gate or three_contact_gates
                else ("object-table" if pair_event_supervision else None)
            ),
            "e_pair_event_loss_weight": (
                float(getattr(model, "e_pair_event_loss_weight", 0.0))
                if pair_event_supervision else 0.0
            ),
            "mu_pair_event_loss_weight": (
                float(getattr(model, "mu_pair_event_loss_weight", 0.0))
                if pair_event_supervision else 0.0
            ),
            "contact_loss_weight": (
                float(getattr(model, "contact_loss_weight", 0.0))
                if shared_contact_gate else 0.0
            ),
            "base_contact_loss_weight": (
                float(getattr(model, "base_contact_loss_weight", 0.0))
                if three_contact_gates else 0.0
            ),
            "e_contact_loss_weight": (
                float(getattr(model, "e_contact_loss_weight", 0.0))
                if three_contact_gates else 0.0
            ),
            "mu_contact_loss_weight": (
                float(getattr(model, "mu_contact_loss_weight", 0.0))
                if three_contact_gates else 0.0
            ),
            "contact_head": (
                "three_independent_pair_mlp" if three_contact_gates
                else "shared_contact" if shared_contact_gate
                else None
            ),
            "three_contact_loss_aggregation": (
                "0.1_times_local_active_head_mean"
                if three_contact_gates else None
            ),
            "relation_qk_gate": False if three_contact_gates else None,
            "manual_d_a": False if three_contact_gates else None,
            "pairwise_condition_tokens": (
                architecture == "temporal_pair_schedule"
            ),
            "pairwise_condition_source": (
                "edge_scalars"
                if independent_edge_route
                else (
                    "pair_tokens"
                    if architecture == "temporal_pair_schedule"
                    else "none"
                )
            ),
            "independent_sigmoid_edges": independent_edge_route,
            "undirected_edge_gate": independent_edge_route,
            "null_competition": bool(no_interaction_route),
            "edge_mu_e_separate_parameters": independent_edge_route,
                "previous_frame_tracking": bool(
                    getattr(first_group.router, "previous_frame_tracking", False)
                ),
                "joint_slot_background_competition": bool(
                    getattr(first_group.router, "joint_slot_competition", False)
                ),
                "single_center_template_writeback": bool(
                    getattr(
                        first_group.router,
                        "single_center_template_writeback",
                        False,
                    )
                ),
                "locator_grid": (
                    [28, 48]
                    if architecture == "temporal_no_interaction_schedule_ictr"
                    else [14, 24]
                ),
                "wan_write_grid": [14, 24],
                "read_write_route_shared": not bool(
                    getattr(first_group, "decoupled_writer", False)
                ),
                "independent_writer": bool(
                    getattr(first_group, "decoupled_writer", False)
                ),
                "writer_radius_tokens": (
                    getattr(first_group.writer, "radius", None)
                    if getattr(first_group, "writer", None) is not None
                    else None
                ),
                "continuous_object_state": bool(
                    getattr(dit.sparse_object_adapter, "continuous_object_state", False)
                ),
                "high_resolution_locator_to_wan_projection": bool(
                    getattr(
                        first_group.router,
                        "wan_write_grid_projection",
                        False,
                    )
                ),
            "mu_message_gate_alpha": getattr(
                first_group.interaction, "mu_message_gate_alpha", None
            ),
            "gravity_injected": bool(
                temporal_route and not getattr(dit.sparse_object_adapter, "disable_gravity", False)
            ),
            "gravity_object_bound": bool(temporal_route),
            "gravity_runtime_off": bool(
                getattr(dit.sparse_object_adapter, "disable_gravity", False)
            ),
            "backbone_frozen": True,
            "expected_trainable_parameter_count": (
                84_502_296
                if architecture == "temporal_pair_schedule"
                else None
            ),
        }
        if temporal_route:
            route_details.update(
                {
                    "temporal_attention_axis": "same_object_across_time",
                    "same_object_temporal_causal_mask": bool(
                        getattr(first_group, "causal_object_temporal", False)
                    ),
                    "event_after_effect_persistence": bool(
                        getattr(first_group, "event_after_effect", False)
                    ),
                    "object_attention_axis": "same_time_across_objects",
                    "e_mu_qk_modulation": False,
                    "e_mu_value_and_receiver_residual_modulation": (
                        architecture != "temporal_pair_schedule"
                        and not independent_edge_route
                    ),
                    "e_mu_post_selection_pair_response": (
                        architecture == "temporal_pair_schedule"
                    ),
                    "e_mu_edge_local_residuals": independent_edge_route,
                    "pair_response_excludes_self": (
                        architecture == "temporal_pair_schedule"
                    ),
                    "nonself_attention_renormalized": False,
                    "pooled_object_token": False,
                }
            )
        # The formal launcher freezes the requested block list. Audit the
        # instantiated topology against that list instead of silently forcing
        # the historical eight-group layout: single-block ablations must still
        # prove that no hidden group was constructed.
        route_ok = (
            bool(blocks)
            and len(set(blocks)) == len(blocks)
            and all(0 <= index < 30 for index in blocks)
            and list(groups.keys()) == [str(index) for index in blocks]
            and len(condition_encoder_ids) == len(blocks)
            and len(set(condition_encoder_ids)) == len(blocks)
            and (
                architecture != "temporal_no_interaction_schedule_ictr"
                or (
                    route_details["previous_frame_tracking"]
                    and route_details["joint_slot_background_competition"]
                    and route_details["single_center_template_writeback"]
                    and route_details["locator_grid"] == [28, 48]
                    and route_details[
                        "high_resolution_locator_to_wan_projection"
                    ]
                )
            )
            and (
                architecture not in {
                    "temporal_no_interaction_schedule_mugate_decoupled_writer",
                    "temporal_no_interaction_schedule_mugate_decoupled_writer_continuous",
                    "temporal_independent_edge_schedule_decoupled_writer_continuous",
                }
                or (
                    route_details["independent_writer"]
                    and not route_details["read_write_route_shared"]
                    and route_details["writer_radius_tokens"] == 1
                )
            )
            and (
                architecture
                not in {
                    "temporal_no_interaction_schedule_mugate_decoupled_writer_continuous",
                    "temporal_independent_edge_schedule_decoupled_writer_continuous",
                }
                or route_details["continuous_object_state"]
            )
            and (
                architecture != "temporal_pair_schedule"
                or sum(parameter.numel() for _, parameter in trainable)
                == 84_502_296
            )
        )
    elif has_object_time_graph:
        dit = model.pipe.dit
        blocks = list(dit.object_time_graph_adapter.injection_blocks)
        groups = dit.object_time_graph_adapter.groups
        allowed = lambda name: "object_time_graph_adapter." in name
        group_depths = {
            key: len(group.blocks) for key, group in groups.items()
        }
        route_details = {
            "route": "object_time_graph",
            "injection_blocks_zero_based": blocks,
            "graph_blocks_per_group": group_depths,
            "loss_contract": (
                "flow_matching_plus_object_localization_bce_plus_contact_bce"
            ),
            "future_gt_in_forward": False,
            "backbone_frozen": True,
        }
        route_ok = (
            blocks == [2, 6, 10, 14, 18, 22, 25, 27]
            and len(group_depths) == 8
            and set(group_depths.values()) == {2}
        )
    elif has_oracle_slot:
        dit = model.pipe.dit
        blocks = list(dit.oracle_slot_adapter.injection_blocks)
        allowed = lambda name: "oracle_slot_adapter." in name
        route_details = {
            "route": "oracle_slot",
            "injection_blocks": blocks,
            "loss_contract": "flow_matching_plus_balanced_impact_bce",
            "backbone_frozen": True,
        }
        route_ok = blocks == [7, 15, 23]
    elif has_side:
        side_count = sum(
            parameter.numel()
            for name, parameter in named
            if "unified_side_dit." in name
        )
        allowed = lambda name: (
            "unified_side_dit." in name
            or ".lora_A." in name
            or ".lora_B." in name
        )
        route_details = {
            "route": "side",
            "side_parameter_count": side_count,
            "expected_side_parameter_count": 1_385_073_664,
        }
        route_ok = side_count == 1_385_073_664
    elif os.environ.get("PHYSICAL_WM_TEXT_ONLY", "0") == "1":
        dit = model.pipe.dit
        allowed = lambda name: ".lora_A." in name or ".lora_B." in name
        route_details = {"route": "textonly_lora", "backbone_frozen": True,
                         "loss_contract": "flow_matching_only"}
        route_ok = (not dit.force_condition_attention and not dit.global_physics_attention
                    and all(not p.requires_grad for n, p in named if not allowed(n)))
    else:
        dit = model.pipe.dit
        force_blocks = sorted(int(key) for key in dit.force_condition_attention.keys())
        physics_blocks = sorted(int(key) for key in dit.global_physics_attention.keys())
        allowed = lambda name: (
            "force_condition_" in name
            or "global_physics_" in name
            or ".lora_A." in name
            or ".lora_B." in name
        )
        route_details = {
            "route": "cross",
            "force_attention_blocks": force_blocks,
            "physics_attention_blocks": physics_blocks,
        }
        route_ok = force_blocks == [0, 4, 8, 12, 16, 20, 24, 28] and physics_blocks == force_blocks
    unexpected = [name for name, _ in trainable if not allowed(name)]
    passed = bool(trainable and route_ok and not unexpected)
    payload = {
        "passed": passed,
        "distributed_world_size": int(accelerator.num_processes),
        "model_parameter_count": total,
        "trainable_parameter_count": sum(parameter.numel() for _, parameter in trainable),
        "trainable_tensor_count": len(trainable),
        "unexpected_trainable_names": unexpected,
        **route_details,
    }
    if accelerator.is_main_process:
        Path(output_path, "model_contract_audit.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    if not passed:
        raise RuntimeError(f"full-model contract audit failed: {payload}")


def save_training_args(args):
    output_path = getattr(args, "output_path", None) if args is not None else None
    if output_path is None:
        return
    try:
        os.makedirs(args.output_path, exist_ok=True)
        save_path = os.path.join(args.output_path, "training_args.json")
        payload = dict(vars(args))
        payload["e_pair_event_loss_mode"] = os.environ.get(
            "PHYSICAL_WM_E_PAIR_EVENT_LOSS_MODE", "hard_slot"
        ).strip().lower()
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4, ensure_ascii=False, default=str)
        print(f"Training arguments saved to `{save_path}`.")
    except Exception as e:
        print(f"Warning: failed to save training arguments: {e}")


def _phyparam_resume_contract(args, model: DiffusionTrainingModule) -> dict | None:
    dit = getattr(getattr(model, "pipe", None), "dit", None)
    branch = getattr(dit, "phyparam_control_dit", None)
    if branch is None:
        return None
    feature_enabled = bool(branch.feature_supervision_enabled)
    metadata = Path(str(args.dataset_metadata_path)).expanduser().resolve()
    metadata_sha256 = None
    if metadata.is_file():
        digest = hashlib.sha256()
        with metadata.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        metadata_sha256 = digest.hexdigest()
    return {
        "contract": "phyparam-nonparameter-resume-v1",
        "num_control_blocks": int(branch.num_control_blocks),
        "harmonic_bands": int(branch.condition_encoder.harmonic_bands),
        "default_mass_normalized": float(branch.condition_encoder.default_mass_normalized),
        "max_objects": int(branch.condition_encoder.max_objects),
        "feature_supervision_enabled": feature_enabled,
        "mode": "paper_full_system" if feature_enabled else "flow_only_ablation",
        "dino_feature_dim": int(branch.feature_supervisor.feature_dim) if feature_enabled else None,
        "dino_target_grid": list(branch.feature_supervisor.target_grid) if feature_enabled else None,
        "feature_tap_blocks": list(branch.feature_supervisor.tap_blocks) if feature_enabled else [],
        "temporal_min_norm": float(branch.feature_supervisor.temporal_min_norm) if feature_enabled else None,
        "phyparam_feature_loss_weight": float(args.phyparam_feature_loss_weight),
        "phyparam_temporal_loss_weight": float(args.phyparam_temporal_loss_weight),
        "dino_teacher_path": getattr(args, "phyparam_dino_teacher_path", None),
        "dino_teacher_sha256": getattr(args, "phyparam_dino_teacher_sha256", None),
        "dino_feature_contract": getattr(args, "phyparam_dino_feature_contract", None),
        "dino_decoded_frames": getattr(args, "phyparam_dino_decoded_frames", None),
        "dino_feature_frame_indices": getattr(
            args, "phyparam_dino_feature_frame_indices", None
        ),
        "training_spatial_preprocess": {
            "height": getattr(args, "height", None),
            "width": getattr(args, "width", None),
            "resize_mode": getattr(args, "video_resize_mode", None),
        },
        "dataset_metadata_path": str(metadata),
        "dataset_metadata_sha256": metadata_sha256,
        "dataset_base_path": str(Path(str(args.dataset_base_path)).expanduser().resolve()),
        "data_file_keys": sorted(set(args.data_file_keys.split(",")) - {""}),
        "extra_inputs": sorted(set((args.extra_inputs or "").split(",")) - {""}),
    }


def _verify_or_write_phyparam_resume_contract(args, model, output_path: str) -> None:
    current = _phyparam_resume_contract(args, model)
    if current is None:
        return
    path = Path(output_path, "phyparam_resume_contract.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    requested = getattr(args, "resume_from_checkpoint", None)
    if requested is not None:
        requested_path = None if requested == "latest" else Path(requested).expanduser().resolve()
        if requested_path is None or not requested_path.is_file():
            source_root = Path(output_path).resolve() if requested == "latest" else requested_path
            if source_root.is_dir() and source_root.name.startswith("checkpoint-step-"):
                source_root = source_root.parent
            source = source_root / "phyparam_resume_contract.json"
            if not source.is_file():
                raise FileNotFoundError(f"PhyParam resume contract is missing: {source}")
            previous = json.loads(source.read_text(encoding="utf-8"))
            if previous != current:
                raise RuntimeError("PhyParam non-parameter resume contract mismatch")
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous != current:
            raise RuntimeError("refusing to overwrite a different PhyParam resume contract")
    elif requested is None:
        part = path.with_suffix(path.suffix + f".{os.getpid()}.part")
        part.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(part, path)


def _validate_phyparam_memory_contract(args, model) -> None:
    dit = getattr(getattr(model, "pipe", None), "dit", None)
    branch = getattr(dit, "phyparam_control_dit", None)
    if branch is None or branch.num_control_blocks < len(dit.blocks):
        return
    if not getattr(args, "enable_model_cpu_offload", False):
        raise RuntimeError(
            "full-depth PhyParam 5B requires --enable_model_cpu_offload; "
            "the optimizer placement is selected by the frozen launcher and "
            "remote 5B memory smoke remains mandatory"
        )


def _select_training_offload_scope(model):
    """Limit full-depth PhyParam offload hooks to models used by cached SFT.

    ``sft:train`` consumes pre-encoded Wan inputs, so its forward calls only the
    pipeline's non-null ``in_iteration_models``.  Registering the offloader on
    the outer training module also pins the unused T5/VAE preprocessing models
    once per rank.  On an eight-rank node that can exhaust the host cgroup
    before the first optimizer step.  Keep the generic path unchanged and
    fail closed if a PhyParam trainable parameter or a retained pipeline unit
    lies outside the selected execution scope.
    """
    pipe = getattr(model, "pipe", None)
    dit = getattr(pipe, "dit", None)
    branch = getattr(dit, "phyparam_control_dit", None)
    if branch is None or branch.num_control_blocks < len(dit.blocks):
        return model, ("model",)

    iteration_names = tuple(getattr(pipe, "in_iteration_models", ()))
    active_names = []
    active_modules = []
    seen_module_ids = set()
    for name in iteration_names:
        module = getattr(pipe, name, None)
        if module is None:
            continue
        active_names.append(name)
        if id(module) not in seen_module_ids:
            active_modules.append(module)
            seen_module_ids.add(id(module))
    if "dit" not in active_names or not active_modules:
        raise RuntimeError(
            "full-depth PhyParam offload scope must contain pipe.dit"
        )

    retained_model_names = {
        name
        for unit in getattr(pipe, "units", ())
        for name in (getattr(unit, "onload_model_names", None) or ())
        if getattr(pipe, name, None) is not None
    }
    unexpected_units = sorted(retained_model_names - set(active_names))
    if unexpected_units:
        raise RuntimeError(
            "cached PhyParam training retained preprocessing model units outside "
            f"the offload scope: {unexpected_units}"
        )

    scope = torch.nn.ModuleList(active_modules)
    scope_parameter_ids = {id(parameter) for parameter in scope.parameters()}
    outside_trainable = sorted(
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and id(parameter) not in scope_parameter_ids
    )
    if outside_trainable:
        raise RuntimeError(
            "PhyParam trainable parameters lie outside the cached-training "
            f"offload scope: {outside_trainable[:8]}"
        )
    return scope, tuple(f"pipe.{name}" for name in active_names)


def _dump_object_attention_visualization(
    accelerator: Accelerator,
    dit,
    data: dict,
    step: int,
    validation_item_index: int,
) -> None:
    """Optionally preserve real per-sample routing maps during validation.

    Normal training is unchanged unless ``OTG_ATTENTION_DUMP_DIR`` is set.
    The dump contains the group-averaged logits used by the existing
    validation IoU, its GT labels, and the 13 RGB frames aligned to the Wan
    latent times. This makes checkpoint comparisons reproducible without
    treating aggregate IoU as a visualization.
    """

    root_raw = os.environ.get("OTG_ATTENTION_DUMP_DIR")
    if not root_raw:
        return
    limit = int(os.environ.get("OTG_ATTENTION_DUMP_LIMIT", "4"))
    if validation_item_index >= limit:
        return
    logits = getattr(dit, "object_time_graph_last_attention_logits", None)
    labels = getattr(dit, "object_time_graph_last_attention_labels", None)
    route = "object_time_graph_v05"
    if logits is None or labels is None:
        logits = getattr(dit, "sparse_object_last_attention_logits", None)
        labels = getattr(dit, "sparse_object_last_attention_labels", None)
        architecture = getattr(dit, "sparse_object_architecture", None)
        route = (
            "sparse_object_interaction_independent_undirected_edge_v1"
            if architecture
            == "temporal_independent_edge_schedule_decoupled_writer_continuous"
            else (
                "sparse_object_interaction_temporal_pair_schedule_v1"
                if architecture == "temporal_pair_schedule"
                else (
                    "sparse_object_interaction_temporal_no_interaction_schedule_v1"
                    if architecture in {
                        "temporal_no_interaction_schedule",
                        "temporal_no_interaction_schedule_mugate",
                        "temporal_no_interaction_schedule_mugate_decoupled_writer",
                        "temporal_no_interaction_schedule_mugate_decoupled_writer_continuous",
                        "temporal_no_interaction_schedule_trackprev_mugate",
                        "temporal_no_interaction_schedule_ictr",
                    }
                    else (
                        "sparse_object_interaction_temporal_schedule_v1"
                        if architecture == "temporal_schedule"
                        else (
                            "sparse_object_interaction_temporal_v1"
                            if architecture == "temporal"
                            else "sparse_object_interaction_v2"
                        )
                    )
                )
            )
        )
    if logits is None or labels is None:
        return

    clip_id = str(
        data.get(
            "clip_id",
            f"rank{accelerator.process_index:02d}_item{validation_item_index:04d}",
        )
    )
    safe_clip_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", clip_id)
    sample_dir = (
        Path(root_raw)
        / f"step-{int(step):08d}"
        / f"rank{accelerator.process_index:02d}_{safe_clip_id}"
    )
    sample_dir.mkdir(parents=True, exist_ok=True)
    metadata_keys = (
        "clip_id",
        "group_id",
        "expected_event",
        "first_contact_frame",
        "force_target_instance_id",
        "force_magnitude_n",
        "force_direction_x",
        "force_direction_y",
        "mu_A",
        "mu_B",
        "restitution_requested",
        "video",
    )
    metadata = {
        key: str(data[key])
        for key in metadata_keys
        if key in data and key != "video"
    }
    payload = {
        "step": int(step),
        "rank": int(accelerator.process_index),
        "validation_item_index": int(validation_item_index),
        "clip_id": clip_id,
        "route": route,
        "attention_logits": logits.detach().float().cpu(),
        "attention_labels": labels.detach().float().cpu(),
        "metadata": metadata,
    }
    tensor_path = sample_dir / "routing.pt"
    temporary_tensor_path = sample_dir / "routing.pt.incomplete"
    torch.save(payload, temporary_tensor_path)
    os.replace(temporary_tensor_path, tensor_path)

    video = data.get("video")
    if isinstance(video, (list, tuple)) and video:
        for latent_time in range(13):
            frame_index = min(latent_time * 4, len(video) - 1)
            frame = video[frame_index]
            if hasattr(frame, "save"):
                frame.save(sample_dir / f"frame_t{latent_time:02d}.png")
    manifest_path = sample_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "step": int(step),
                "rank": int(accelerator.process_index),
                "validation_item_index": int(validation_item_index),
                "clip_id": clip_id,
                "route": route,
                "metadata": metadata,
                "routing_tensor": str(tensor_path),
                "latent_frame_mapping": {
                    str(latent_time): min(latent_time * 4, 48)
                    for latent_time in range(13)
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _run_deterministic_validation(
    accelerator: Accelerator,
    model: DiffusionTrainingModule,
    dataloader: torch.utils.data.DataLoader,
    model_logger: ModelLogger,
    step: int,
    seed: int,
) -> float:
    """Evaluate noise-prediction loss over the complete frozen validation split.

    Each sample uses one model(data) forward pass under eval()/no_grad with a
    fixed per-rank seed. The forward computes the same task-specific MSE as
    training; validation does not decode videos or run multi-step sampling.
    """

    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(accelerator.device) if torch.cuda.is_available() else None
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    module_training_modes = [(module, module.training) for module in model.modules()]
    unwrapped = accelerator.unwrap_model(model)
    validation_checkpointing = getattr(unwrapped, "use_gradient_checkpointing", None)
    rank_seed = int(seed) + int(accelerator.process_index)
    torch.manual_seed(rank_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(rank_seed)
    random.seed(rank_seed)
    np.random.seed(rank_seed % (2**32))
    model.eval()
    if validation_checkpointing is not None:
        unwrapped.use_gradient_checkpointing = False
    local_sum = torch.zeros((), dtype=torch.float64, device=accelerator.device)
    local_count = torch.zeros((), dtype=torch.float64, device=accelerator.device)
    # tp, fp, tn, fn, hit_peak_exact, hit_count, peak_abs_error
    local_impact = torch.zeros(7, dtype=torch.float64, device=accelerator.device)
    # future-object binary-mask intersection and union
    local_object = torch.zeros(2, dtype=torch.float64, device=accelerator.device)
    classification_kind = "impact"
    try:
        # CPU-offload hooks move trainable Parameters during every forward.
        # inference_mode would permanently turn those moved tensors into
        # inference tensors, which cannot later participate in backward.
        with torch.no_grad():
            for validation_item_index, data in enumerate(dataloader):
                cached_validation = getattr(dataloader.dataset, "load_from_cache", False)
                if cached_validation:
                    contact_required = any(
                        os.environ.get(name, "0").strip().lower()
                        in {"1", "true", "yes", "on"}
                        for name in (
                            "PHYSICAL_WM_SHARED_CONTACT_GATE",
                            "PHYSICAL_WM_THREE_CONTACT_GATES",
                        )
                    )
                    data = _prepare_cached_validation_sample(
                        data, contact_required=contact_required
                    )
                    validation_inputs = data[0]
                else:
                    validation_inputs = data
                loss = model({}, inputs=data) if cached_validation else model(data)
                local_sum += loss.detach().to(dtype=torch.float64)
                local_count += 1
                dit = accelerator.unwrap_model(model).pipe.dit
                logits = getattr(dit, "oracle_slot_last_impact_logits", None)
                labels = getattr(dit, "oracle_slot_last_impact_labels", None)
                if logits is None or labels is None:
                    logits = getattr(
                        dit, "object_time_graph_last_contact_logits", None
                    )
                    labels = getattr(
                        dit, "object_time_graph_last_contact_labels", None
                    )
                    if logits is not None and labels is not None:
                        classification_kind = "contact"
                if logits is not None and labels is not None:
                    predictions = logits.sigmoid() >= 0.5
                    truth = labels >= 0.5
                    local_impact[0] += (predictions & truth).sum()
                    local_impact[1] += (predictions & ~truth).sum()
                    local_impact[2] += (~predictions & ~truth).sum()
                    local_impact[3] += (~predictions & truth).sum()
                    hit_samples = truth.flatten(1).any(dim=1)
                    if hit_samples.any():
                        flat_logits = logits.flatten(1)
                        flat_truth = truth.flatten(1)
                        predicted_peak = flat_logits.argmax(dim=1)
                        true_peak = flat_truth.float().argmax(dim=1)
                        errors = (
                            predicted_peak[hit_samples] - true_peak[hit_samples]
                        ).abs()
                        local_impact[4] += (errors == 0).sum()
                        local_impact[5] += hit_samples.sum()
                        local_impact[6] += errors.sum()
                object_logits = getattr(
                    dit, "object_time_graph_last_attention_logits", None
                )
                object_labels = getattr(
                    dit, "object_time_graph_last_attention_labels", None
                )
                if object_logits is None or object_labels is None:
                    object_logits = getattr(
                        dit, "sparse_object_last_attention_logits", None
                    )
                    object_labels = getattr(
                        dit, "sparse_object_last_attention_labels", None
                    )
                _dump_object_attention_visualization(
                    accelerator,
                    dit,
                    validation_inputs,
                    step,
                    validation_item_index,
                )
                if object_logits is not None and object_labels is not None:
                    predicted_objects = object_logits[:, 1:].sigmoid() >= 0.5
                    true_objects = object_labels[:, 1:] >= 0.5
                    sparse_condition = validation_inputs.get("sparse_object_condition")
                    if sparse_condition is not None:
                        object_valid = sparse_condition[
                            "object_valid_mask"
                        ].to(predicted_objects.device) > 0.5
                        if object_valid.ndim == 1:
                            object_valid = object_valid.unsqueeze(0)
                        valid_grid = object_valid[:, None, :, None, None]
                        predicted_objects = predicted_objects & valid_grid
                        true_objects = true_objects & valid_grid
                    local_object[0] += (
                        predicted_objects & true_objects
                    ).sum()
                    local_object[1] += (
                        predicted_objects | true_objects
                    ).sum()
        totals = torch.stack([local_sum, local_count])
        totals = accelerator.reduce(totals, reduction="sum")
        if totals[1].item() <= 0:
            raise RuntimeError("validation split produced zero samples")
        validation_loss = (totals[0] / totals[1]).item()
        if not np.isfinite(validation_loss):
            raise FloatingPointError(f"non-finite validation loss at step {step}: {validation_loss}")
        model_logger.log_metric(accelerator, "validation_loss", validation_loss, step)
        impact_totals = accelerator.reduce(local_impact, reduction="sum")
        object_totals = accelerator.reduce(local_object, reduction="sum")
        impact_payload = None
        classified = impact_totals[:4].sum().item()
        if classified > 0:
            tp, fp, tn, fn = [float(x) for x in impact_totals[:4].tolist()]
            accuracy = (tp + tn) / max(tp + fp + tn + fn, 1.0)
            precision = tp / max(tp + fp, 1.0)
            recall = tp / max(tp + fn, 1.0)
            hit_count = float(impact_totals[5].item())
            peak_accuracy = float(impact_totals[4].item()) / max(hit_count, 1.0)
            peak_mae = float(impact_totals[6].item()) / max(hit_count, 1.0)
            impact_payload = {
                "accuracy_at_0.5": accuracy,
                "precision_at_0.5": precision,
                "recall_at_0.5": recall,
                "hit_peak_window_accuracy": peak_accuracy,
                "hit_peak_window_mae": peak_mae,
                "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
            }
            for key, value in impact_payload.items():
                if key not in ("tp", "fp", "tn", "fn"):
                    model_logger.log_metric(
                        accelerator,
                        f"validation_{classification_kind}_{key}",
                        value,
                        step,
                    )
            if accelerator.is_main_process:
                Path(
                    model_logger.output_path,
                    f"{classification_kind}_validation_step{step:08d}.json",
                ).write_text(
                    json.dumps(impact_payload, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
        object_iou = None
        if object_totals[1].item() > 0:
            object_iou = float(
                (object_totals[0] / object_totals[1]).item()
            )
            model_logger.log_metric(
                accelerator,
                "validation_object_attention_iou_at_0.5",
                object_iou,
                step,
            )
            if accelerator.is_main_process:
                Path(
                    model_logger.output_path,
                    f"object_attention_validation_step{step:08d}.json",
                ).write_text(
                    json.dumps(
                        {
                            "future_attention_iou_at_0.5": object_iou,
                            "intersection": int(object_totals[0].item()),
                            "union": int(object_totals[1].item()),
                            "time_scope": "latent times 1..12",
                        },
                        indent=2,
                        sort_keys=True,
                    ),
                    encoding="utf-8",
                )
        if accelerator.is_main_process:
            validation_receipt = {
                "step": int(step),
                "loss": validation_loss,
                "samples": int(totals[1].item()),
                "seed": int(seed),
                "finite": bool(math.isfinite(validation_loss)),
            }
            Path(
                model_logger.output_path,
                f"validation_metrics_step{step:08d}.json",
            ).write_text(
                json.dumps(validation_receipt, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(
                f"[validation] step={step} loss={validation_loss:.10f} "
                f"samples={int(totals[1].item())} seed={seed}",
                flush=True,
            )
            if impact_payload is not None:
                print(
                    f"[{classification_kind}-validation] step={step} "
                    f"accuracy={impact_payload['accuracy_at_0.5']:.6f} "
                    f"precision={impact_payload['precision_at_0.5']:.6f} "
                    f"recall={impact_payload['recall_at_0.5']:.6f} "
                    f"peak_acc={impact_payload['hit_peak_window_accuracy']:.6f}",
                    flush=True,
                )
            if object_iou is not None:
                print(
                    f"[object-attention-validation] step={step} "
                    f"iou={object_iou:.6f}",
                    flush=True,
                )
        return validation_loss
    finally:
        for module, training in module_training_modes:
            module.training = training
        if validation_checkpointing is not None:
            unwrapped.use_gradient_checkpointing = validation_checkpointing
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng, accelerator.device)
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)


def launch_training_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = None,
    num_epochs: int = 1,
    max_train_steps: int = None,
    enable_model_cpu_offload: bool = False,
    enable_optimizer_cpu_offload: bool = False,
    cpu_offload_split_threshold: int = None,
    customized_optimizer: str = None,
    validation_dataset: torch.utils.data.Dataset = None,
    args = None,
    **kwargs,
):
    if args is not None:
        learning_rate = args.learning_rate
        lora_lr_scale = args.lora_lr_scale
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
        max_train_steps = args.max_train_steps
        enable_model_cpu_offload = args.enable_model_cpu_offload
        enable_optimizer_cpu_offload = args.enable_optimizer_cpu_offload
        cpu_offload_split_threshold = args.cpu_offload_split_threshold
        customized_optimizer = args.customized_optimizer
        validation_steps = args.validation_steps
        validation_seed = args.validation_seed
    else:
        validation_steps = None
        validation_seed = 424242
        lora_lr_scale = 1.0

    if accelerator.is_main_process:
        _verify_or_write_phyparam_resume_contract(args, model, model_logger.output_path)
        save_training_args(args)
    _validate_phyparam_memory_contract(args, model)
    _write_model_contract_audit(accelerator, model, model_logger.output_path)

    optimizer_class = get_optimizer_class(customized_optimizer)
    optimizer = optimizer_class(
        _optimizer_param_groups(
            model,
            learning_rate,
            lora_lr_scale,
            float(getattr(args, "sparse_object_erase_gate_lr_multiplier", 1.0)),
        ),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
    train_batch_size = int(getattr(args, "train_batch_size", 1))
    if train_batch_size <= 0:
        raise ValueError("train_batch_size must be positive")
    train_sampler, sampler_audit = _build_hierarchical_train_sampler(args, dataset)
    if bool(getattr(args, "drop_incomplete_global_batch", False)):
        if train_sampler is not None:
            raise ValueError(
                "exact no-padding sampler cannot be combined with weighted sampling"
            )
        global_batch = train_batch_size * int(accelerator.num_processes)
        train_sampler = _EpochTruncatedRandomSampler(
            len(dataset), global_batch, int(getattr(args, "seed", 0))
        )
        sampler_audit = {
            "contract": "epoch-shuffle-without-replacement-drop-incomplete-global-batch-v1",
            "dataset_rows": int(len(dataset)),
            "usable_rows_per_epoch": int(len(train_sampler)),
            "dropped_rows_per_epoch": int(len(dataset) - len(train_sampler)),
            "global_batch": int(global_batch),
            "seed": int(getattr(args, "seed", 0)),
            "replacement": False,
            "accelerate_padding_required": False,
        }
    if accelerator.is_main_process and sampler_audit is not None:
        sampler_audit_path = Path(model_logger.output_path) / "sampler_runtime_audit.json"
        sampler_audit_path.parent.mkdir(parents=True, exist_ok=True)
        sampler_audit_path.write_text(
            json.dumps(sampler_audit, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        batch_size=train_batch_size,
        collate_fn=_collate_training_samples, num_workers=num_workers,
    )
    validation_dataloader = None
    if validation_dataset is not None:
        if validation_steps is None or validation_steps <= 0:
            raise ValueError("validation_dataset requires a positive validation_steps interval")
        validation_sampler = None
        if len(validation_dataset) % accelerator.num_processes != 0:
            if not bool(getattr(args, "drop_incomplete_global_batch", False)):
                raise ValueError(
                    "validation size must be divisible by the distributed world size to avoid padded duplicates: "
                    f"rows={len(validation_dataset)} world={accelerator.num_processes}"
                )
            validation_sampler = _FixedTruncatedRandomSampler(
                len(validation_dataset),
                int(accelerator.num_processes),
                int(validation_seed),
            )
            if accelerator.is_main_process:
                Path(
                    model_logger.output_path,
                    "validation_sampler_runtime_audit.json",
                ).write_text(
                    json.dumps(
                        {
                            "contract": "fixed-shuffle-without-replacement-drop-incomplete-world-v1",
                            "dataset_rows": int(len(validation_dataset)),
                            "evaluated_rows": int(len(validation_sampler)),
                            "dropped_rows": int(
                                len(validation_dataset) - len(validation_sampler)
                            ),
                            "world_size": int(accelerator.num_processes),
                            "seed": int(validation_seed),
                            "replacement": False,
                            "accelerate_padding_required": False,
                        },
                        indent=2,
                        sort_keys=True,
                    ) + "\n",
                    encoding="utf-8",
                )
        validation_dataloader = torch.utils.data.DataLoader(
            validation_dataset,
            shuffle=False,
            sampler=validation_sampler,
            collate_fn=lambda x: x[0],
            num_workers=0,
        )

    if enable_model_cpu_offload:
        optimizer, dataloader, scheduler = accelerator.prepare(optimizer, dataloader, scheduler)
        model.pipe.device = accelerator.device
        offload_scope, offload_scope_names = _select_training_offload_scope(model)
        if accelerator.is_main_process:
            offload_audit_path = Path(model_logger.output_path) / "offload_scope_audit.json"
            offload_audit_path.parent.mkdir(parents=True, exist_ok=True)
            offload_audit_path.write_text(
                json.dumps(
                    {
                        "scope": list(offload_scope_names),
                        "scope_parameter_count": sum(
                            parameter.numel() for parameter in offload_scope.parameters()
                        ),
                        "scope_frozen_parameter_count": sum(
                            parameter.numel()
                            for parameter in offload_scope.parameters()
                            if not parameter.requires_grad
                        ),
                        "scope_trainable_parameter_count": sum(
                            parameter.numel()
                            for parameter in offload_scope.parameters()
                            if parameter.requires_grad
                        ),
                        "excluded_preprocessing_models": [
                            *getattr(model, "cached_phyparam_released_models", ()),
                            *[
                                name
                                for name in ("text_encoder", "image_encoder", "vae")
                                if getattr(model.pipe, name, None) is not None
                                and f"pipe.{name}" not in offload_scope_names
                            ],
                        ],
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        offload_manager = OffloadTrainingManager(
            offload_scope,
            accelerator.device,
            enable_optimizer_cpu_offload,
            cpu_offload_split_threshold,
        )
    else:
        model.to(device=accelerator.device)
        model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    if validation_dataloader is not None:
        validation_dataloader = accelerator.prepare(validation_dataloader)

    cursor = TrainingCursor()
    accelerator.register_for_checkpointing(cursor)
    _register_trainable_only_state_hooks(
        accelerator,
        checkpoint_model=model if enable_model_cpu_offload else None,
    )
    resume_checkpoint = _resolve_complete_checkpoint(
        model_logger.output_path,
        getattr(args, "resume_from_checkpoint", None),
    )
    if resume_checkpoint is not None:
        accelerator.load_state(str(resume_checkpoint))
        model_logger.num_steps = cursor.global_step
        if accelerator.is_main_process:
            print(
                f"Resumed complete state from `{resume_checkpoint}` at "
                f"step={cursor.global_step}, epoch={cursor.epoch}, batch={cursor.batch_in_epoch}."
            )
            # Numeric resume-verification fingerprint (FIX-3): fingerprint the
            # just-loaded in-memory trainable weights the same way save_hook
            # fingerprinted them at checkpoint time, and dump it so
            # smoke/check_resume_verify_log.py can assert exact sha256 equality
            # against `<resume_checkpoint>/trainable_checksum.json` -- proof
            # that resume actually loaded that checkpoint's weights, rather
            # than silently reinitializing or loading something stale/wrong.
            resume_unwrapped = accelerator.unwrap_model(model)
            resume_trainable_names = resume_unwrapped.trainable_param_names()
            resumed_state = {
                name: parameter
                for name, parameter in resume_unwrapped.named_parameters()
                if name in resume_trainable_names
            }
            resumed_checksum = _trainable_state_checksum(resumed_state)
            saved_checksum_path = resume_checkpoint / "trainable_checksum.json"
            if not saved_checksum_path.is_file():
                raise FileNotFoundError(
                    f"resume checkpoint has no trainable checksum: {saved_checksum_path}"
                )
            saved_checksum = json.loads(saved_checksum_path.read_text(encoding="utf-8"))
            for key in ("sha256", "num_tensors"):
                if resumed_checksum[key] != saved_checksum.get(key):
                    raise RuntimeError(
                        "post-resume trainable state differs from the selected checkpoint: "
                        f"{key} saved={saved_checksum.get(key)!r} "
                        f"loaded={resumed_checksum[key]!r}"
                    )
            audit_dir = Path(model_logger.output_path) / "resume_audit"
            audit_dir.mkdir(parents=True, exist_ok=True)
            audit_path = audit_dir / f"post_resume_step{cursor.global_step:08d}.json"
            audit_path.write_text(
                json.dumps(
                    {
                        "source": "post_resume",
                        "resumed_from": str(resume_checkpoint),
                        "resumed_at_step": cursor.global_step,
                        **resumed_checksum,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(
                f"[resume-audit] post-resume trainable-state sha256={resumed_checksum['sha256']} "
                f"l2_norm={resumed_checksum['l2_norm']:.10f} num_tensors={resumed_checksum['num_tensors']} "
                f"verified against {saved_checksum_path} and written to {audit_path}"
            )

    initialize_deepspeed_gradient_checkpointing(accelerator)
    gradient_family_hits: set[str] = set()
    active_gradient_condition_families: set[str] = set()
    independent_edge_gradient_audit = (
        getattr(accelerator.unwrap_model(model).pipe.dit, "sparse_object_architecture", None)
        == "temporal_independent_edge_schedule_decoupled_writer_continuous"
    )
    observed_microbatch = None
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(accelerator.device)
    last_validated_step = -1
    if validation_dataloader is not None:
        _run_deterministic_validation(
            accelerator,
            model,
            validation_dataloader,
            model_logger,
            cursor.global_step,
            validation_seed,
        )
        last_validated_step = cursor.global_step
    reached_max_steps = max_train_steps is not None and cursor.global_step >= max_train_steps
    last_saved_step = cursor.global_step if resume_checkpoint is not None else -1
    for epoch_id in range(cursor.epoch, num_epochs):
        if reached_max_steps:
            break
        if hasattr(dataloader, "set_epoch"):
            dataloader.set_epoch(epoch_id)
        batches_to_skip = cursor.batch_in_epoch if epoch_id == cursor.epoch else 0
        epoch_dataloader = (
            accelerator.skip_first_batches(dataloader, batches_to_skip)
            if batches_to_skip
            else dataloader
        )
        # skip_first_batches constructs a new DataLoaderShard whose iteration
        # defaults to zero, so restore the epoch on the wrapper as well.
        if hasattr(epoch_dataloader, "set_epoch"):
            epoch_dataloader.set_epoch(epoch_id)
        for data in tqdm(epoch_dataloader, disable=not accelerator.is_local_main_process):
            actual_microbatch = _actual_microbatch_size(data)
            if actual_microbatch != train_batch_size:
                raise RuntimeError(
                    f"actual microbatch {actual_microbatch} differs from configured {train_batch_size}"
                )
            observed_microbatch = actual_microbatch
            with accelerator.accumulate(model):
                # Runtime-only progress for weak pair-event regularizer warmup.
                # This scalar is not checkpoint state and is restored from the
                # authoritative cursor on every batch, including resume.
                accelerator.unwrap_model(model).pipe.dit.sparse_object_global_step = int(
                    cursor.global_step
                )
                if dataset.load_from_cache:
                    loss = model({}, inputs=data)
                else:
                    if (
                        accelerator.sync_gradients
                        and cursor.global_step == 0
                        and not getattr(args, "skip_step0_equivalence", False)
                    ):
                        # Run the clean pass first inside a forked RNG context;
                        # on exit, the actual conditioned forward below sees
                        # byte-identical random noise/timestep draws.
                        clean_data = dict(data)
                        force = data.get("force_condition_tensor")
                        oracle_condition = data.get("oracle_slot_condition")
                        object_time_graph_condition = data.get(
                            "object_time_graph_condition"
                        )
                        sparse_object_condition = data.get(
                            "sparse_object_condition"
                        )
                        if (
                            (force is None or not torch.is_tensor(force))
                            and oracle_condition is None
                            and object_time_graph_condition is None
                            and sparse_object_condition is None
                        ):
                            raise RuntimeError(
                                "step-0 equivalence audit requires a force, "
                                "oracle-slot, or object-time graph condition"
                            )
                        clean_data["__return_denoiser_prediction__"] = True
                        clean_data["__disable_condition_modules__"] = True
                        conditioned_data = dict(data)
                        conditioned_data["__return_denoiser_prediction__"] = True
                        devices = (
                            [torch.cuda.current_device()]
                            if torch.cuda.is_available()
                            else []
                        )
                        audit_model = accelerator.unwrap_model(model)
                        with torch.random.fork_rng(devices=devices, enabled=True):
                            with torch.no_grad():
                                clean_loss, clean_prediction = audit_model(
                                    clean_data
                                )
                        with torch.random.fork_rng(devices=devices, enabled=True):
                            with torch.no_grad():
                                loss, conditioned_prediction = audit_model(
                                    conditioned_data
                                )

                        local_difference = (
                            loss.detach().float() - clean_loss.detach().float()
                        ).abs().reshape(1)
                        differences = accelerator.gather(local_difference)
                        local_prediction_difference = (
                            conditioned_prediction.detach().float()
                            - clean_prediction.detach().float()
                        ).abs().max().reshape(1)
                        prediction_differences = accelerator.gather(
                            local_prediction_difference
                        )
                        local_bitwise = torch.tensor(
                            [int(torch.equal(
                                conditioned_prediction.detach(),
                                clean_prediction.detach(),
                            ))],
                            device=local_difference.device,
                            dtype=torch.int64,
                        )
                        bitwise_flags = accelerator.gather(local_bitwise)
                        raw_tokens = data.get("phys_tokens_json")
                        oracle_present = bool(
                            oracle_condition is not None
                            and torch.is_tensor(oracle_condition.get("masks"))
                            and torch.count_nonzero(
                                oracle_condition["masks"]
                            ).item() > 0
                        )
                        object_time_graph_present = bool(
                            object_time_graph_condition is not None
                            and torch.is_tensor(
                                object_time_graph_condition.get(
                                    "first_frame_masks"
                                )
                            )
                            and torch.count_nonzero(
                                object_time_graph_condition[
                                    "first_frame_masks"
                                ]
                            ).item() > 0
                        )
                        sparse_object_present = bool(
                            sparse_object_condition is not None
                            and torch.is_tensor(
                                sparse_object_condition.get("first_frame_masks")
                            )
                            and torch.count_nonzero(
                                sparse_object_condition["first_frame_masks"]
                            ).item() > 0
                        )
                        force_present = bool(
                            torch.is_tensor(force)
                            and torch.count_nonzero(force).item() > 0
                        )
                        condition_present = torch.tensor(
                            [
                                int(
                                    force_present
                                    or oracle_present
                                    or object_time_graph_present
                                    or sparse_object_present
                                    or raw_tokens not in (None, "", "[]", [], ())
                                )
                            ],
                            device=local_difference.device,
                            dtype=torch.int64,
                        )
                        condition_flags = accelerator.gather(condition_present)
                        maximum_difference = float(differences.max().item())
                        maximum_prediction_difference = float(
                            prediction_differences.max().item()
                        )
                        all_predictions_bitwise_equal = bool(
                            torch.all(bitwise_flags > 0).item()
                        )
                        all_ranks_conditioned = bool(
                            torch.all(condition_flags > 0).item()
                        )
                        passed = bool(
                            maximum_difference == 0.0
                            and maximum_prediction_difference == 0.0
                            and all_predictions_bitwise_equal
                            and all_ranks_conditioned
                        )
                        if accelerator.is_main_process:
                            Path(
                                model_logger.output_path,
                                "step0_equivalence_audit.json",
                            ).write_text(
                                json.dumps(
                                    {
                                        "passed": passed,
                                        "scope": "direct denoiser output tensor plus scalar flow-matching loss",
                                        "comparison": "conditioned candidate vs condition modules disabled",
                                        "same_sample_noise_and_timestep": True,
                                        "world_size": int(differences.numel()),
                                        "all_ranks_had_nonempty_F_or_physics_condition": all_ranks_conditioned,
                                        "max_abs_loss_difference": maximum_difference,
                                        "max_abs_denoiser_prediction_difference": maximum_prediction_difference,
                                        "all_ranks_denoiser_prediction_bitwise_equal": all_predictions_bitwise_equal,
                                        "per_rank_abs_loss_difference": [
                                            float(value)
                                            for value in differences.cpu().tolist()
                                        ],
                                        "rank0_conditioned_loss": float(
                                            loss.detach().float().item()
                                        ),
                                        "rank0_clean_loss": float(
                                            clean_loss.detach().float().item()
                                        ),
                                    },
                                    indent=2,
                                    sort_keys=True,
                                ),
                                encoding="utf-8",
                            )
                        if not passed:
                            raise RuntimeError(
                                "step-0 end-to-end equivalence failed: "
                                f"max_abs_loss_difference={maximum_difference}, "
                                f"max_abs_prediction_difference={maximum_prediction_difference}, "
                                f"bitwise={all_predictions_bitwise_equal}, "
                                f"all_ranks_conditioned={all_ranks_conditioned}"
                            )
                        # The equivalence probe intentionally returns before
                        # auxiliary losses. Run one ordinary conditioned
                        # forward for the actual optimizer step so step 1
                        # includes the full route-specific training objective.
                        loss = model(data)
                    else:
                        loss = model(data)
                accelerator.backward(loss)
                if enable_model_cpu_offload:
                    _allreduce_offload_gradients(accelerator, model)
                    offload_manager.after_backward()
                if accelerator.sync_gradients and cursor.global_step == 0:
                    _write_first_gradient_audit(
                        accelerator,
                        model,
                        model_logger.output_path,
                    )
                if accelerator.sync_gradients and cursor.global_step < 3:
                    if independent_edge_gradient_audit:
                        active_gradient_condition_families = (
                            _update_active_gradient_condition_families(
                                data,
                                active_gradient_condition_families,
                            )
                        )
                    gradient_family_hits = _update_three_step_gradient_family_audit(
                        accelerator,
                        model,
                        model_logger.output_path,
                        cursor.global_step + 1,
                        gradient_family_hits,
                        (
                            active_gradient_condition_families
                            if independent_edge_gradient_audit
                            else None
                        ),
                    )
                max_grad_norm = float(os.environ.get("PHYSICAL_WM_MAX_GRAD_NORM", "0"))
                if accelerator.sync_gradients and max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                cursor.batch_in_epoch += 1
                if accelerator.sync_gradients:
                    cursor.global_step += 1
                    if cursor.global_step == 1:
                        _write_batch_contract_rank(
                            accelerator, model_logger.output_path, args,
                            observed_microbatch, cursor.global_step,
                        )
                    train_loss = accelerator.reduce(loss.detach().float(), reduction="mean")
                    if not torch.isfinite(train_loss):
                        raise FloatingPointError(
                            f"non-finite train loss at step {cursor.global_step}: {train_loss.item()}"
                        )
                    model_logger.on_step_end(accelerator, model, None, loss=train_loss)
                    unwrapped_dit = accelerator.unwrap_model(model).pipe.dit
                    component_losses = getattr(
                        unwrapped_dit,
                        "sparse_object_last_losses",
                        None,
                    )
                    if not component_losses:
                        component_losses = getattr(
                            unwrapped_dit, "object_time_graph_last_losses", None
                        )
                    if not component_losses:
                        component_losses = getattr(
                            unwrapped_dit, "oracle_slot_last_losses", None
                        )
                    if not component_losses:
                        component_losses = getattr(
                            unwrapped_dit, "phyparam_last_losses", None
                        )
                    if component_losses:
                        persisted_components = {}
                        for name, component in component_losses.items():
                            reduced_component = accelerator.reduce(
                                component.detach().float(), reduction="mean"
                            )
                            persisted_components[name] = float(reduced_component.item())
                            model_logger.log_metric(
                                accelerator,
                                f"train_{name}",
                                reduced_component,
                                cursor.global_step,
                            )
                        if accelerator.is_main_process:
                            metrics_path = Path(
                                model_logger.output_path, "training_metrics.jsonl"
                            )
                            with metrics_path.open("a", encoding="utf-8") as handle:
                                handle.write(
                                    json.dumps(
                                        {
                                            "step": int(cursor.global_step),
                                            "train_loss": float(train_loss.item()),
                                            **persisted_components,
                                        },
                                        sort_keys=True,
                                    )
                                    + "\n"
                                )
                    if save_steps is not None and cursor.global_step % save_steps == 0:
                        _save_complete_checkpoint(
                            accelerator,
                            model_logger.output_path,
                            cursor,
                        )
                        last_saved_step = cursor.global_step
                    if (
                        validation_dataloader is not None
                        and cursor.global_step % validation_steps == 0
                    ):
                        _run_deterministic_validation(
                            accelerator,
                            model,
                            validation_dataloader,
                            model_logger,
                            cursor.global_step,
                            validation_seed,
                        )
                        last_validated_step = cursor.global_step
                    if max_train_steps is not None and cursor.global_step >= max_train_steps:
                        reached_max_steps = True
            if reached_max_steps:
                break
        if reached_max_steps:
            break
        cursor.epoch = epoch_id + 1
        cursor.batch_in_epoch = 0
        if save_steps is None and cursor.global_step != last_saved_step:
            _save_complete_checkpoint(accelerator, model_logger.output_path, cursor)
            last_saved_step = cursor.global_step

    if cursor.global_step > 0 and cursor.global_step != last_saved_step:
        _save_complete_checkpoint(accelerator, model_logger.output_path, cursor)
    if validation_dataloader is not None and cursor.global_step != last_validated_step:
        _run_deterministic_validation(
            accelerator,
            model,
            validation_dataloader,
            model_logger,
            cursor.global_step,
            validation_seed,
        )
    model_logger.on_training_end(accelerator, model, None)


def launch_data_process_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    num_workers: int = 8,
    args = None,
    **kwargs,
):
    if args is not None:
        num_workers = args.dataset_num_workers
        enable_model_cpu_offload = args.enable_model_cpu_offload
        enable_optimizer_cpu_offload = args.enable_optimizer_cpu_offload
        cpu_offload_split_threshold = args.cpu_offload_split_threshold
        
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers)
    if enable_model_cpu_offload:
        dataloader = accelerator.prepare(dataloader)
        offload_manager = OffloadTrainingManager(model, accelerator.device, enable_optimizer_cpu_offload, cpu_offload_split_threshold)
        model.pipe.device = accelerator.device
    else:
        model.to(device=accelerator.device)
        model, dataloader = accelerator.prepare(model, dataloader)
    
    for data_id, data in enumerate(tqdm(dataloader)):
        with accelerator.accumulate(model):
            with torch.no_grad():
                folder = os.path.join(model_logger.output_path, str(accelerator.process_index))
                os.makedirs(folder, exist_ok=True)
                save_path = os.path.join(model_logger.output_path, str(accelerator.process_index), f"{data_id}.pth")
                provenance_path = save_path + ".provenance.json"
                clip_id = data.get("clip_id")
                if clip_id is not None and (not isinstance(clip_id, str) or not clip_id):
                    raise ValueError(
                        "cache provenance clip_id must be a non-empty string"
                    )
                data = model(data)
                if args is not None and args.compact_data_process_cache:
                    if not (
                        isinstance(data, (tuple, list))
                        and len(data) == 3
                        and all(isinstance(part, dict) for part in data)
                    ):
                        raise TypeError("compact data-process cache requires the standard three-part payload")
                    shared = dict(data[0])
                    shared.pop("input_video", None)
                    shared.pop("input_image", None)
                    for required in ("latents", "input_latents", "first_frame_latents"):
                        if not torch.is_tensor(shared.get(required)):
                            raise ValueError(f"compact data-process cache lacks {required}")
                    data = (shared, dict(data[1]), dict(data[2]))
                torch.save(data, save_path)
                if clip_id is not None:
                    temporary_provenance_path = provenance_path + ".incomplete"
                    Path(temporary_provenance_path).write_text(
                        json.dumps({"clip_id": clip_id}, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    os.replace(temporary_provenance_path, provenance_path)
                if enable_model_cpu_offload:
                    offload_manager.after_backward()

def initialize_deepspeed_gradient_checkpointing(accelerator: Accelerator):
    if getattr(accelerator.state, "deepspeed_plugin", None) is not None:
        ds_config = accelerator.state.deepspeed_plugin.deepspeed_config
        if "activation_checkpointing" in ds_config:
            import deepspeed
            act_config = ds_config["activation_checkpointing"]
            deepspeed.checkpointing.configure(
                mpu_=None, 
                partition_activations=act_config.get("partition_activations", False),
                checkpoint_in_cpu=act_config.get("cpu_checkpointing", False),
                contiguous_checkpointing=act_config.get("contiguous_memory_optimization", False)
            )
        else:
            print("Do not find activation_checkpointing config in deepspeed config, skip initializing deepspeed gradient checkpointing.")
