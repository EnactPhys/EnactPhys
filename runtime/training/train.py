import torch, os, argparse, accelerate, warnings, json, math
from pathlib import Path
from diffsynth.core import UnifiedDataset
from diffsynth.core.data.operators import LoadVideo, LoadAudio, LoadTorchPickle, ImageCropAndResize, ToAbsolutePath
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.diffusion import *
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def validate_phyparam_data_file_keys(enable_phyparam, extra_inputs, data_file_keys):
    extra_input_keys = set((extra_inputs or "").split(",")) - {""}
    file_keys = set((data_file_keys or "").split(",")) - {""}
    required = extra_input_keys & {"sparse_object_condition", "phyparam_dino_features"}
    missing = sorted(required - file_keys)
    if enable_phyparam and missing:
        raise ValueError(
            "PhyParam extra_inputs must also be listed in --data_file_keys: "
            f"{missing}"
        )


def _select_cache_override_key(
    cache_override_manifest,
    *,
    enable_phyparam_control_dit,
    enable_prompt_context_override,
):
    if cache_override_manifest is None:
        return None
    if enable_phyparam_control_dit:
        return "phyparam_bundle"
    if enable_prompt_context_override:
        return "sparse_object_condition_with_prompt"
    return "sparse_object_condition"


def validate_phyparam_full_system_identity(
    enable_phyparam,
    flow_only_ablation,
    teacher_sha256,
    feature_contract,
    decoded_frames,
    feature_frame_indices,
):
    if enable_phyparam and not flow_only_ablation:
        if (
            not teacher_sha256
            or not feature_contract
            or decoded_frames is None
            or not feature_frame_indices
        ):
            raise ValueError(
                "PhyParam full-system mode requires DINO teacher SHA256, feature "
                "contract, decoded frames, and exact feature frame indices"
            )


def release_cached_phyparam_preprocessors(pipe, task, enable_phyparam):
    """Release models that cannot execute in the frozen cached-train graph."""
    if not enable_phyparam or not str(task).endswith(":train"):
        return ()
    preprocessing_models = ("text_encoder", "image_encoder", "vae")
    retained_model_names = {
        name
        for unit in getattr(pipe, "units", ())
        for name in (getattr(unit, "onload_model_names", None) or ())
        if getattr(pipe, name, None) is not None
    }
    unexpected = sorted(retained_model_names & set(preprocessing_models))
    if unexpected:
        raise RuntimeError(
            "cached PhyParam training still requires preprocessing models: "
            f"{unexpected}"
        )
    released = []
    for name in preprocessing_models:
        if getattr(pipe, name, None) is not None:
            setattr(pipe, name, None)
            released.append(name)
    return tuple(released)


def _validate_three_contact_loss_weights(base_weight, e_weight, mu_weight):
    weights = tuple(float(value) for value in (base_weight, e_weight, mu_weight))
    if any(value <= 0.0 for value in weights) or any(
        abs(value - weights[0]) > 1.0e-12 for value in weights[1:]
    ):
        raise ValueError(
            "three-contact active-head averaging requires three equal positive "
            "loss weights"
        )
    return weights


class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        tokenizer_path=None, audio_processor_path=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="", lora_rank=32, lora_checkpoint=None,
        preset_lora_path=None, preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        resume_from_checkpoint=None, remove_prefix_in_ckpt=None,
        initial_sparse_object_checkpoint_dir=None,
        initial_sparse_object_checkpoint_sha256=None,
        initial_sparse_object_checkpoint_step=None,
        device="cpu",
        task="sft",
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        enable_friction_controlnet=False,
        enable_force_controlnet=False,
        force_condition_mode=None,
        enable_global_physics_cross_attn=False,
        enable_oracle_slot_adapter=False,
        enable_object_time_graph_adapter=False,
        enable_sparse_object_interaction_adapter=False,
        enable_phyparam_control_dit=False,
        phyparam_num_control_blocks=None,
        phyparam_harmonic_bands=8,
        phyparam_default_mass_normalized=0.5,
        phyparam_max_objects=8,
        phyparam_dino_feature_dim=4096,
        phyparam_feature_tap_blocks="",
        phyparam_flow_only_ablation=False,
        phyparam_dino_target_grid="14,14",
        phyparam_temporal_min_norm=1e-6,
        phyparam_dino_teacher_path=None,
        phyparam_dino_teacher_sha256=None,
        phyparam_dino_feature_contract=None,
        phyparam_dino_decoded_frames=None,
        phyparam_dino_feature_frame_indices=None,
        phyparam_training_height=None,
        phyparam_training_width=None,
        phyparam_training_resize_mode=None,
        sparse_object_adapter_architecture="v2",
        sparse_object_injection_blocks="2,6,10,14,18,22,25,27",
        impact_loss_weight=0.1,
        object_attention_loss_weight=0.1,
        reader_supervision_mode="balanced_bce",
        writer_loss_weight=0.0,
        writer_centroid_loss_weight=0.0,
        e_pair_event_loss_weight=0.0,
        mu_pair_event_loss_weight=0.0,
        contact_loss_weight=0.1,
        base_contact_loss_weight=0.0,
        e_contact_loss_weight=0.0,
        mu_contact_loss_weight=0.0,
        dynamic_loss_weight=0.2,
        temporal_loss_weight=0.2,
        phyparam_feature_loss_weight=0.1,
        phyparam_temporal_loss_weight=0.1,
    ):
        super().__init__()
        # Warning
        if enable_sparse_object_interaction_adapter and enable_phyparam_control_dit:
            raise ValueError(
                "Mainline-2 sparse-object and PhyParam branches are mutually exclusive"
            )
        validate_phyparam_full_system_identity(
            enable_phyparam_control_dit,
            phyparam_flow_only_ablation,
            phyparam_dino_teacher_sha256,
            phyparam_dino_feature_contract,
            phyparam_dino_decoded_frames,
            phyparam_dino_feature_frame_indices,
        )
        phyparam_inputs = set((extra_inputs or "").split(",")) - {""}
        if enable_phyparam_control_dit and "sparse_object_condition" not in phyparam_inputs:
            raise ValueError("PhyParam requires sparse_object_condition in --extra_inputs")
        if (
            enable_phyparam_control_dit
            and not phyparam_flow_only_ablation
            and "phyparam_dino_features" not in phyparam_inputs
        ):
            raise ValueError("PhyParam full-system mode requires phyparam_dino_features in --extra_inputs")
        if not use_gradient_checkpointing:
            warnings.warn("Gradient checkpointing is detected as disabled. To prevent out-of-memory errors, the training framework will forcibly enable gradient checkpointing.")
            use_gradient_checkpointing = True

        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, fp8_models=fp8_models, offload_models=offload_models, device=device)
        tokenizer_config = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/") if tokenizer_path is None else ModelConfig(tokenizer_path)
        audio_processor_config = self.parse_path_or_model_id(audio_processor_path)
        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device,
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
            audio_processor_config=audio_processor_config,
            enable_friction_controlnet=enable_friction_controlnet,
            enable_force_controlnet=enable_force_controlnet,
            force_condition_mode=force_condition_mode,
            enable_global_physics_cross_attn=enable_global_physics_cross_attn,
            enable_oracle_slot_adapter=enable_oracle_slot_adapter,
            enable_object_time_graph_adapter=enable_object_time_graph_adapter,
            enable_sparse_object_interaction_adapter=enable_sparse_object_interaction_adapter,
            enable_phyparam_control_dit=enable_phyparam_control_dit,
            phyparam_num_control_blocks=phyparam_num_control_blocks,
            phyparam_harmonic_bands=phyparam_harmonic_bands,
            phyparam_default_mass_normalized=phyparam_default_mass_normalized,
            phyparam_max_objects=phyparam_max_objects,
            phyparam_dino_feature_dim=phyparam_dino_feature_dim,
            phyparam_feature_tap_blocks=(
                tuple(
                    int(value.strip())
                    for value in phyparam_feature_tap_blocks.split(",")
                    if value.strip()
                )
                or None
            ),
            phyparam_enable_feature_supervision=not phyparam_flow_only_ablation,
            phyparam_dino_target_grid=tuple(
                int(value.strip())
                for value in phyparam_dino_target_grid.split(",")
                if value.strip()
            ),
            phyparam_temporal_min_norm=phyparam_temporal_min_norm,
            sparse_object_adapter_architecture=sparse_object_adapter_architecture,
            sparse_object_injection_blocks=tuple(
                int(value.strip())
                for value in sparse_object_injection_blocks.split(",")
                if value.strip()
            ),
        )
        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)
        self.cached_phyparam_released_models = release_cached_phyparam_preprocessors(
            self.pipe,
            task,
            enable_phyparam_control_dit,
        )
        initial_sparse_values = (
            initial_sparse_object_checkpoint_dir,
            initial_sparse_object_checkpoint_sha256,
            initial_sparse_object_checkpoint_step,
        )
        if any(value is not None for value in initial_sparse_values):
            if not all(value is not None for value in initial_sparse_values):
                raise ValueError(
                    "the Sparse Object Interaction warm-start requires checkpoint "
                    "directory, SHA256, and global step together"
                )
            if resume_from_checkpoint is not None:
                raise ValueError(
                    "weight warm-start and optimizer-state resume are mutually exclusive"
                )
            adapter = getattr(self.pipe.dit, "sparse_object_adapter", None)
            if adapter is None:
                raise ValueError(
                    "Sparse Object Interaction warm-start requires the adapter to be enabled"
                )
            # Use the same strict checkpoint identity loader as evaluation.
            # ICTR changes only the router support and therefore admits exactly
            # one newly initialized bounded tracking-mix scalar per injection
            # group; every other tensor must match the parent checkpoint.
            from sparse_object_interaction_v2.evaluation.checkpoint_io import (
                load_sparse_object_adapter,
            )

            allowed_missing_keys = ()
            if (
                sparse_object_adapter_architecture
                == "temporal_no_interaction_schedule_ictr"
            ):
                allowed_missing_keys = tuple(
                    f"groups.{index}.router.tracking_mix_logit"
                    for index in adapter.injection_blocks
                )
            if os.environ.get("PHYSICAL_WM_WRITE_GATE_MODE", "legacy") == "erase_then_write_v1":
                allowed_missing_keys = tuple(
                    f"groups.{index}.erase_gate.{suffix}"
                    for index in adapter.injection_blocks
                    for suffix in ("0.weight", "0.bias", "2.weight", "2.bias")
                )
            if os.environ.get(
                "PHYSICAL_WM_PAIR_EVENT_SUPERVISION", "0"
            ).strip().lower() in {"1", "true", "yes", "on"}:
                allowed_missing_keys = tuple(allowed_missing_keys) + tuple(
                    f"groups.{index}.interaction.{axis}_pair_event_head.{suffix}"
                    for index in adapter.injection_blocks
                    for axis in ("e", "mu")
                    for suffix in ("weight", "bias")
                )
            self.initial_sparse_object_checkpoint_identity = load_sparse_object_adapter(
                adapter,
                Path(initial_sparse_object_checkpoint_dir),
                initial_sparse_object_checkpoint_sha256,
                int(initial_sparse_object_checkpoint_step),
                allowed_missing_keys=allowed_missing_keys,
            )
        # File checkpoints retain the upstream weight-only behaviour.  Complete
        # checkpoint directories are restored by the runner after optimizer and
        # scheduler creation.
        if resume_from_checkpoint is not None and os.path.isfile(resume_from_checkpoint):
            self.resume_from_checkpoint(resume_from_checkpoint, remove_prefix_in_ckpt)
        
        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint,
            preset_lora_path, preset_lora_model,
            task=task,
        )
        if getattr(self.pipe.dit, "friction_controlnet", None) is not None:
            self.pipe.dit.freeze_base_for_friction_controlnet()
        if getattr(self.pipe.dit, "force_controlnet", None) is not None:
            self.pipe.dit.freeze_base_for_force_controlnet()
        if getattr(self.pipe.dit, "force_condition_mode", None) is not None or getattr(self.pipe.dit, "global_physics_enabled", False):
            self.pipe.dit.freeze_base_for_force_and_physics_conditioning()
        if getattr(self.pipe.dit, "oracle_slot_enabled", False):
            self.pipe.dit.freeze_base_for_oracle_slot_adapter()
        if getattr(self.pipe.dit, "object_time_graph_enabled", False):
            self.pipe.dit.freeze_base_for_object_time_graph_adapter()
        if getattr(self.pipe.dit, "sparse_object_enabled", False):
            self.pipe.dit.freeze_base_for_sparse_object_adapter()
        if getattr(self.pipe.dit, "phyparam_control_enabled", False):
            self.pipe.dit.freeze_base_for_phyparam_control_dit()
        
        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.task = task
        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "direct_distill:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
        }
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        self.impact_loss_weight = float(impact_loss_weight)
        self.object_attention_loss_weight = float(object_attention_loss_weight)
        self.reader_supervision_mode = str(reader_supervision_mode)
        if self.reader_supervision_mode not in {"balanced_bce", "mask_route_read"}:
            raise ValueError(
                "reader_supervision_mode must be balanced_bce or mask_route_read"
            )
        self.writer_loss_weight = float(writer_loss_weight)
        self.writer_centroid_loss_weight = float(writer_centroid_loss_weight)
        self.e_pair_event_loss_weight = float(e_pair_event_loss_weight)
        self.mu_pair_event_loss_weight = float(mu_pair_event_loss_weight)
        self.contact_loss_weight = float(contact_loss_weight)
        self.base_contact_loss_weight = float(base_contact_loss_weight)
        self.e_contact_loss_weight = float(e_contact_loss_weight)
        self.mu_contact_loss_weight = float(mu_contact_loss_weight)
        pair_event_supervision = os.environ.get(
            "PHYSICAL_WM_PAIR_EVENT_SUPERVISION", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.pair_event_supervision = pair_event_supervision
        self.shared_contact_gate = os.environ.get(
            "PHYSICAL_WM_SHARED_CONTACT_GATE", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.three_contact_gates = os.environ.get(
            "PHYSICAL_WM_THREE_CONTACT_GATES", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        if self.three_contact_gates:
            if (
                self.shared_contact_gate
                or pair_event_supervision
                or self.contact_loss_weight != 0.0
                or self.e_pair_event_loss_weight != 0.0
                or self.mu_pair_event_loss_weight != 0.0
            ):
                raise ValueError(
                    "three-contact route forbids shared and legacy contact objectives"
                )
            _validate_three_contact_loss_weights(
                self.base_contact_loss_weight,
                self.e_contact_loss_weight,
                self.mu_contact_loss_weight,
            )
            if os.environ.get(
                "PHYSICAL_WM_DISABLE_GRAVITY", "0"
            ).strip().lower() in {"1", "true", "yes", "on"}:
                raise ValueError(
                    "three-contact training forbids global gravity disable"
                )
            self.contact_supervision = True
        elif self.shared_contact_gate:
            contact_supervision = os.environ.get(
                "PHYSICAL_WM_CONTACT_SUPERVISION", "0"
            ).strip().lower() in {"1", "true", "yes", "on"}
            self.contact_supervision = contact_supervision
            if (
                pair_event_supervision
                or self.e_pair_event_loss_weight != 0.0
                or self.mu_pair_event_loss_weight != 0.0
            ):
                raise ValueError(
                    "shared contact forbids legacy e/mu event supervision"
                )
            expected_contact_weight = 0.1 if contact_supervision else 0.0
            if self.contact_loss_weight != expected_contact_weight:
                raise ValueError(
                    "contact supervision flag and contact loss weight disagree"
                )
            if os.environ.get(
                "PHYSICAL_WM_DISABLE_GRAVITY", "0"
            ).strip().lower() in {"1", "true", "yes", "on"}:
                raise ValueError(
                    "contact-core forbids the legacy global gravity-disable override"
                )
        elif pair_event_supervision and (
            self.e_pair_event_loss_weight != 0.1
            or self.mu_pair_event_loss_weight != 0.1
        ):
            raise ValueError("pair e/mu event loss weights must both equal 0.1")
        else:
            self.contact_supervision = False
        self.dynamic_loss_weight = float(dynamic_loss_weight)
        self.temporal_loss_weight = float(temporal_loss_weight)
        self.phyparam_feature_loss_weight = float(phyparam_feature_loss_weight)
        self.phyparam_temporal_loss_weight = float(phyparam_temporal_loss_weight)
        self.sparse_object_adapter_architecture = sparse_object_adapter_architecture
        self.enable_phyparam_control_dit = bool(enable_phyparam_control_dit)
        self.phyparam_flow_only_ablation = bool(phyparam_flow_only_ablation)
        self.phyparam_max_objects = int(phyparam_max_objects)
        self.phyparam_dino_teacher_path = phyparam_dino_teacher_path
        self.phyparam_dino_teacher_sha256 = phyparam_dino_teacher_sha256
        self.phyparam_dino_feature_contract = phyparam_dino_feature_contract
        self.phyparam_dino_decoded_frames = (
            int(phyparam_dino_decoded_frames)
            if phyparam_dino_decoded_frames is not None else None
        )
        self.phyparam_dino_feature_frame_indices = (
            tuple(int(value) for value in phyparam_dino_feature_frame_indices)
            if phyparam_dino_feature_frame_indices else None
        )
        self.phyparam_training_spatial_preprocess = {
            "height": (
                int(phyparam_training_height)
                if phyparam_training_height is not None else None
            ),
            "width": (
                int(phyparam_training_width)
                if phyparam_training_width is not None else None
            ),
            "resize_mode": str(phyparam_training_resize_mode),
        }
        
    def parse_extra_inputs(self, data, extra_inputs, inputs_shared):
        for extra_input in extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data["video"][0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][-1]
            elif extra_input == "reference_image" or extra_input == "vace_reference_image":
                inputs_shared[extra_input] = data[extra_input][0]
            else:
                if extra_input == "force_control_latents":
                    control = data[extra_input]
                    if tuple(control.shape) != (48, 13, 28, 48):
                        raise ValueError(
                            "dynamic point-force latent must be [48,13,28,48], got "
                            f"{tuple(control.shape)}"
                        )
                    if control.dtype != torch.bfloat16 or not torch.isfinite(control).all():
                        raise ValueError(
                            f"invalid dynamic point-force latent: dtype={control.dtype}, "
                            f"finite={bool(torch.isfinite(control).all())}"
                        )
                elif extra_input == "force_condition_tensor":
                    condition = data[extra_input]
                    if tuple(condition.shape) != (3, 13, 28, 48):
                        raise ValueError(
                            "numeric force field must be [3,13,28,48], got "
                            f"{tuple(condition.shape)}"
                        )
                    if condition.dtype != torch.bfloat16 or not torch.isfinite(condition).all():
                        raise ValueError(
                            f"invalid force field: dtype={condition.dtype}, "
                            f"finite={bool(torch.isfinite(condition).all())}"
                        )
                    support = condition[0].float()
                    signed = condition[1:].float()
                    if (
                        float(support.min()) < 0.0
                        or float(support.max()) > 1.0
                        or float(signed.min()) < -1.0
                        or float(signed.max()) > 1.0
                    ):
                        raise ValueError("force support/vector channels violate [-1,1] contract")
                    if torch.count_nonzero(signed[:, support == 0]).item() != 0:
                        raise ValueError("force vector is nonzero outside support")
                elif extra_input == "phys_tokens_json":
                    # Manifest cell is a plain JSON string (read verbatim by
                    # pandas.read_csv, not a file path), e.g. "[]" or
                    # '[["g", 9.8], ["e", 0.35]]'. Missing physics
                    # quantities are never zero-filled: the data builder simply
                    # omits that entry, so the parsed list legitimately ranges
                    # from empty (pure-F sources) up to a handful of entries.
                    raw = data[extra_input]
                    parsed = json.loads(raw) if isinstance(raw, str) else (raw or [])
                    for item in parsed:
                        if (
                            not isinstance(item, (list, tuple))
                            or len(item) != 2
                            or item[0] not in ("g", "mu", "e")
                        ):
                            raise ValueError(f"invalid phys_tokens_json entry: {item!r}")
                        if not (isinstance(item[1], (int, float)) and math.isfinite(float(item[1]))):
                            raise ValueError(f"invalid phys_tokens_json value: {item!r}")
                    inputs_shared["global_physics_tokens"] = [(t, float(v)) for t, v in parsed]
                    continue
                elif extra_input == "oracle_slot_condition":
                    condition = data[extra_input]
                    required = {
                        "masks", "force", "force_target", "force_active",
                        "gravity", "friction", "restitution", "impact_labels",
                    }
                    missing = required - set(condition)
                    if missing:
                        raise ValueError(
                            f"oracle slot condition missing {sorted(missing)}"
                        )
                    if tuple(condition["masks"].shape) != (13, 2, 14, 24):
                        raise ValueError(
                            "oracle masks must be [13,2,14,24], got "
                            f"{tuple(condition['masks'].shape)}"
                        )
                    if tuple(condition["impact_labels"].shape) != (12, 1):
                        raise ValueError(
                            "impact labels must be [12,1], got "
                            f"{tuple(condition['impact_labels'].shape)}"
                        )
                elif extra_input == "object_time_graph_condition":
                    condition = data[extra_input]
                    required = {
                        "first_frame_masks",
                        "force",
                        "force_target",
                        "mu",
                        "restitution",
                        "object_attention_labels",
                        "contact_labels",
                    }
                    missing = required - set(condition)
                    if missing:
                        raise ValueError(
                            "object-time graph condition missing "
                            f"{sorted(missing)}"
                        )
                    expected_shapes = {
                        "first_frame_masks": (2, 14, 24),
                        "force": (3,),
                        "force_target": (2,),
                        "mu": (2,),
                        "restitution": (1,),
                        "object_attention_labels": (13, 2, 14, 24),
                        "contact_labels": (13, 1),
                    }
                    for name, expected in expected_shapes.items():
                        actual = tuple(condition[name].shape)
                        if actual != expected:
                            raise ValueError(
                                f"{name} must be {expected}, got {actual}"
                            )
                        if not torch.isfinite(condition[name]).all():
                            raise ValueError(f"{name} contains non-finite values")
                    for name in (
                        "first_frame_masks",
                        "object_attention_labels",
                        "contact_labels",
                    ):
                        tensor = condition[name]
                        if tensor.min() < 0 or tensor.max() > 1:
                            raise ValueError(f"{name} must be in [0,1]")
                elif extra_input == "sparse_object_condition":
                    condition = data[extra_input]
                    required = {
                        "first_frame_masks",
                        "object_valid_mask",
                        "force",
                        "force_present",
                        "mu",
                        "mu_present",
                        "restitution",
                        "restitution_present",
                    }
                    if not self.enable_phyparam_control_dit:
                        required.add("object_attention_labels")
                    temporal_sparse = self.sparse_object_adapter_architecture in {
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
                    scheduled_temporal = (
                        self.sparse_object_adapter_architecture
                        in {
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
                    )
                    if temporal_sparse or self.enable_phyparam_control_dit:
                        required.update({"gravity", "gravity_present"})
                    if scheduled_temporal:
                        required.add("force_schedule")
                    if self.sparse_object_adapter_architecture in {
                        "temporal_no_interaction_schedule",
                        "temporal_no_interaction_schedule_mugate",
                        "temporal_no_interaction_schedule_mugate_decoupled_writer",
                        "temporal_no_interaction_schedule_mugate_decoupled_writer_continuous",
                        "temporal_independent_edge_schedule_decoupled_writer_continuous",
                        "temporal_no_interaction_schedule_trackprev_mugate",
                        "temporal_no_interaction_schedule_ictr",
                        "temporal_pair_schedule",
                    }:
                        required.add("platform_mask")
                    if self.sparse_object_adapter_architecture == (
                        "temporal_independent_edge_schedule_decoupled_writer_continuous"
                    ):
                        expected_edge_contract = (
                            "continuous-writer-independent-undirected-edge-r4-v1"
                        )
                        if condition.get("contract") != expected_edge_contract:
                            raise ValueError(
                                "independent-edge condition contract differs from "
                                f"{expected_edge_contract!r}"
                            )
                        required.update(
                            {
                                "edge_mu",
                                "edge_mu_present",
                                "edge_restitution",
                                "edge_restitution_present",
                            }
                        )
                        if self.shared_contact_gate or self.three_contact_gates:
                            required.update({"contact_label", "contact_valid"})
                            allowed_contact_contracts = (
                                {"physical-per-head-loss-only-contact-occupancy-v3"}
                                if self.three_contact_gates
                                else {"simulator-loss-only-shared-contact-occupancy-v1"}
                            )
                            if self.three_contact_gates:
                                required.update(
                                    {
                                        "e_contact_label",
                                        "e_contact_valid",
                                        "mu_contact_label",
                                        "mu_contact_valid",
                                    }
                                )
                            if condition.get("contact_contract") not in allowed_contact_contracts:
                                raise ValueError("contact label contract differs")
                        elif self.pair_event_supervision:
                            required.update(
                                {
                                    "e_pair_event_labels",
                                    "e_pair_event_valid",
                                    "mu_pair_event_labels",
                                    "mu_pair_event_valid",
                                }
                            )
                            if condition.get("pair_event_contract") != (
                                "simulator-metadata-loss-only-pair-emu-event-v1"
                            ):
                                raise ValueError("pair event label contract differs")
                    missing = required - set(condition)
                    if missing:
                        raise ValueError(
                            f"sparse object condition missing {sorted(missing)}"
                        )
                    objects = int(condition["first_frame_masks"].shape[0])
                    native_locator = (
                        self.sparse_object_adapter_architecture
                        == "temporal_no_interaction_schedule_ictr"
                    )
                    locator_shape = (28, 48) if native_locator else (14, 24)
                    expected_shapes = {
                        "first_frame_masks": (objects, *locator_shape),
                        "object_valid_mask": (objects,),
                        "force": (objects, 3),
                        "force_present": (objects,),
                        **(
                            {"force_schedule": (13, objects)}
                            if scheduled_temporal
                            else {}
                        ),
                        **(
                            {"platform_mask": locator_shape}
                            if self.sparse_object_adapter_architecture in {
                                "temporal_no_interaction_schedule",
                                "temporal_no_interaction_schedule_mugate",
                                "temporal_no_interaction_schedule_mugate_decoupled_writer",
                                "temporal_no_interaction_schedule_mugate_decoupled_writer_continuous",
                                "temporal_independent_edge_schedule_decoupled_writer_continuous",
                                "temporal_no_interaction_schedule_trackprev_mugate",
                                "temporal_no_interaction_schedule_ictr",
                                "temporal_pair_schedule",
                            }
                            else {}
                        ),
                        **(
                            {
                                "gravity": (objects, 3),
                                "gravity_present": (objects,),
                            }
                            if temporal_sparse or self.enable_phyparam_control_dit
                            else {}
                        ),
                        "mu": (objects,),
                        "mu_present": (objects,),
                        "restitution": (objects,),
                        "restitution_present": (objects,),
                        **(
                            {
                                "edge_mu": (objects, objects + 1),
                                "edge_mu_present": (objects, objects + 1),
                                "edge_restitution": (objects, objects + 1),
                                "edge_restitution_present": (objects, objects + 1),
                                **(
                                    {
                                        "contact_label": (13, objects, objects + 1),
                                        "contact_valid": (13, objects, objects + 1),
                                        **(
                                            {
                                                "e_contact_label": (13, objects, objects + 1),
                                                "e_contact_valid": (13, objects, objects + 1),
                                                "mu_contact_label": (13, objects, objects + 1),
                                                "mu_contact_valid": (13, objects, objects + 1),
                                            }
                                            if self.three_contact_gates
                                            else {}
                                        ),
                                    }
                                    if self.shared_contact_gate or self.three_contact_gates
                                    else {}
                                ),
                                **(
                                    {
                                        "e_pair_event_labels": (13, objects, objects + 1),
                                        "e_pair_event_valid": (13, objects, objects + 1),
                                        "mu_pair_event_labels": (13, objects, objects + 1),
                                        "mu_pair_event_valid": (13, objects, objects + 1),
                                    }
                            if self.pair_event_supervision
                            and not (self.shared_contact_gate or self.three_contact_gates)
                            else {}
                                ),
                            }
                            if self.sparse_object_adapter_architecture
                            == "temporal_independent_edge_schedule_decoupled_writer_continuous"
                            else {}
                        ),
                        **(
                            {"object_attention_labels": (13, objects, *locator_shape)}
                            if not self.enable_phyparam_control_dit
                            else {}
                        ),
                    }
                    for name, expected in expected_shapes.items():
                        tensor = condition[name]
                        if tuple(tensor.shape) != expected:
                            raise ValueError(
                                f"{name} must be {expected}, got {tuple(tensor.shape)}"
                            )
                        if not torch.isfinite(tensor).all():
                            raise ValueError(f"{name} contains non-finite values")
                    for name in (
                        "first_frame_masks",
                        "object_valid_mask",
                        *(
                            ["platform_mask"]
                            if self.sparse_object_adapter_architecture in {
                                "temporal_no_interaction_schedule",
                                "temporal_no_interaction_schedule_mugate",
                                "temporal_no_interaction_schedule_mugate_decoupled_writer",
                                "temporal_no_interaction_schedule_mugate_decoupled_writer_continuous",
                                "temporal_independent_edge_schedule_decoupled_writer_continuous",
                                "temporal_no_interaction_schedule_trackprev_mugate",
                                "temporal_no_interaction_schedule_ictr",
                                "temporal_pair_schedule",
                            }
                            else []
                        ),
                        "force_present",
                        *(["force_schedule"] if scheduled_temporal else []),
                        *(
                            ["gravity_present"]
                            if temporal_sparse or self.enable_phyparam_control_dit
                            else []
                        ),
                        "mu_present",
                        "restitution_present",
                        *(
                            ["edge_mu_present", "edge_restitution_present"]
                            if self.sparse_object_adapter_architecture
                            == "temporal_independent_edge_schedule_decoupled_writer_continuous"
                            else []
                        ),
                        *(
                            [
                                "contact_label",
                                "contact_valid",
                                *(
                                    [
                                        "e_contact_label",
                                        "e_contact_valid",
                                        "mu_contact_label",
                                        "mu_contact_valid",
                                    ]
                                    if self.three_contact_gates
                                    else []
                                ),
                            ]
                            if self.shared_contact_gate or self.three_contact_gates
                            else []
                        ),
                        *(
                            [
                                "e_pair_event_labels",
                                "e_pair_event_valid",
                                "mu_pair_event_labels",
                                "mu_pair_event_valid",
                            ]
                            if self.pair_event_supervision
                            and not (self.shared_contact_gate or self.three_contact_gates)
                            else []
                        ),
                        *(
                            ["object_attention_labels"]
                            if not self.enable_phyparam_control_dit
                            else []
                        ),
                    ):
                        tensor = condition[name]
                        if tensor.min() < 0 or tensor.max() > 1:
                            raise ValueError(f"{name} must be in [0,1]")
                    if self.shared_contact_gate or self.three_contact_gates:
                        prefixes = ("", "e_", "mu_") if self.three_contact_gates else ("",)
                        for prefix in prefixes:
                            contact_label = condition[f"{prefix}contact_label"]
                            contact_valid = condition[f"{prefix}contact_valid"]
                            object_block_label = contact_label[:, :, :objects]
                            object_block_valid = contact_valid[:, :, :objects]
                            if not torch.equal(
                                object_block_label,
                                object_block_label.transpose(1, 2),
                            ):
                                raise ValueError(
                                    f"object-object {prefix}contact_label must be symmetric"
                                )
                            if not torch.equal(
                                object_block_valid,
                                object_block_valid.transpose(1, 2),
                            ):
                                raise ValueError(
                                    f"object-object {prefix}contact_valid must be symmetric"
                                )
                            if torch.any(
                                torch.diagonal(
                                    object_block_valid, dim1=1, dim2=2
                                )
                            ):
                                raise ValueError(f"self {prefix}contact must be invalid")
                            if torch.any(contact_label > contact_valid):
                                raise ValueError(
                                    f"positive {prefix}contact must be marked valid"
                                )
                    if "platform_mask" in condition and not torch.any(
                        condition["platform_mask"] > 0
                    ):
                        raise ValueError(
                            "platform_mask must contain the real physical support; "
                            "background-complement fallback is forbidden"
                        )
                elif extra_input == "phyparam_dino_features":
                    target = data[extra_input]
                    features = (
                        target.get("features") if isinstance(target, dict) else target
                    )
                    if not isinstance(features, torch.Tensor):
                        raise TypeError(
                            "phyparam_dino_features must be a tensor or a mapping "
                            "with a tensor-valued 'features' field"
                        )
                    if features.ndim not in (4, 5):
                        raise ValueError(
                            "formal PhyParam DINO features must be [T,H,W,C] or [B,T,H,W,C]"
                        )
                    if not torch.isfinite(features).all():
                        raise ValueError("phyparam_dino_features contains non-finite values")
                    if not isinstance(target, dict):
                        raise ValueError("PhyParam DINO target must include provenance metadata")
                    provenance_expected = {
                        "teacher_model_sha256": self.phyparam_dino_teacher_sha256,
                        "feature_contract": self.phyparam_dino_feature_contract,
                        "decoded_frames": self.phyparam_dino_decoded_frames,
                        "feature_frame_indices": list(
                            self.phyparam_dino_feature_frame_indices or ()
                        ),
                        "training_spatial_preprocess": self.phyparam_training_spatial_preprocess,
                    }
                    provenance_bad = {
                        key: target.get(key)
                        for key, expected in provenance_expected.items()
                        if target.get(key) != expected
                    }
                    if provenance_bad:
                        raise ValueError(
                            f"PhyParam DINO target provenance mismatch: {provenance_bad}"
                        )
                    supervisor = self.pipe.dit.phyparam_control_dit.feature_supervisor
                    expected_dim = int(supervisor.feature_dim)
                    expected_grid = tuple(supervisor.target_grid)
                    expected_shape = (
                        len(self.phyparam_dino_feature_frame_indices),
                        expected_grid[0], expected_grid[1], expected_dim,
                    )
                    feature_shape = tuple(features.shape[-4:])
                    if feature_shape != expected_shape:
                        raise ValueError(
                            f"PhyParam DINO feature shape must be {expected_shape}, got {tuple(features.shape)}"
                        )
                    if features.dtype not in (torch.bfloat16, torch.float16):
                        raise ValueError("PhyParam DINO target dtype must be bfloat16 or float16")
                    if tuple(target.get("spatial_shape", ())) != expected_grid:
                        raise ValueError("PhyParam DINO target spatial_shape mismatch")
                    expected_numel = math.prod(expected_shape) * (features.shape[0] if features.ndim == 5 else 1)
                    if features.numel() != expected_numel:
                        raise ValueError("PhyParam DINO target exceeds the CPU resource contract")
                inputs_shared[extra_input] = data[extra_input]
        if inputs_shared.get("framewise_decoding", False):
            # WanToDance global model
            inputs_shared["num_frames"] = 4 * (len(data["video"]) - 1) + 1
        return inputs_shared
    
    def get_pipeline_inputs(self, data):
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_video": data["video"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
            "impact_loss_weight": self.impact_loss_weight,
            "object_attention_loss_weight": self.object_attention_loss_weight,
            "reader_supervision_mode": getattr(
                self, "reader_supervision_mode", "balanced_bce"
            ),
            "writer_loss_weight": self.writer_loss_weight,
            "writer_centroid_loss_weight": self.writer_centroid_loss_weight,
            "e_pair_event_loss_weight": self.e_pair_event_loss_weight,
            "mu_pair_event_loss_weight": self.mu_pair_event_loss_weight,
            "contact_loss_weight": self.contact_loss_weight,
            "base_contact_loss_weight": self.base_contact_loss_weight,
            "e_contact_loss_weight": self.e_contact_loss_weight,
            "mu_contact_loss_weight": self.mu_contact_loss_weight,
            "dynamic_loss_weight": self.dynamic_loss_weight,
            "temporal_loss_weight": self.temporal_loss_weight,
            "phyparam_feature_loss_weight": self.phyparam_feature_loss_weight,
            "phyparam_temporal_loss_weight": self.phyparam_temporal_loss_weight,
        }
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        return inputs_shared, inputs_posi, inputs_nega
    
    def forward(self, data, inputs=None):
        data = dict(data)
        return_prediction = bool(data.pop("__return_denoiser_prediction__", False))
        disable_condition_modules = bool(data.pop("__disable_condition_modules__", False))
        if inputs is None: inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        # Loss controls are frozen run configuration, not sample data.  Cached
        # inputs come from several historical data-process jobs and may contain
        # stale or mutually different copies.  Always restore every loss weight
        # from the current training module after microbatch collation.
        inputs_shared, inputs_posi, inputs_nega = inputs
        inputs_shared = dict(inputs_shared)
        if os.environ.get("PHYSICAL_WM_TEXT_ONLY", "0") == "1":
            inputs_shared.pop("sparse_object_condition", None)
        inputs_shared["impact_loss_weight"] = self.impact_loss_weight
        inputs_shared["object_attention_loss_weight"] = self.object_attention_loss_weight
        inputs_shared["reader_supervision_mode"] = getattr(
            self, "reader_supervision_mode", "balanced_bce"
        )
        inputs_shared["writer_loss_weight"] = self.writer_loss_weight
        inputs_shared["writer_centroid_loss_weight"] = self.writer_centroid_loss_weight
        inputs_shared["e_pair_event_loss_weight"] = self.e_pair_event_loss_weight
        inputs_shared["mu_pair_event_loss_weight"] = self.mu_pair_event_loss_weight
        inputs_shared["contact_loss_weight"] = self.contact_loss_weight
        inputs_shared["base_contact_loss_weight"] = getattr(
            self, "base_contact_loss_weight", 0.0
        )
        inputs_shared["e_contact_loss_weight"] = getattr(
            self, "e_contact_loss_weight", 0.0
        )
        inputs_shared["mu_contact_loss_weight"] = getattr(
            self, "mu_contact_loss_weight", 0.0
        )
        inputs_shared["dynamic_loss_weight"] = self.dynamic_loss_weight
        inputs_shared["temporal_loss_weight"] = self.temporal_loss_weight
        inputs_shared["phyparam_feature_loss_weight"] = self.phyparam_feature_loss_weight
        inputs_shared["phyparam_temporal_loss_weight"] = self.phyparam_temporal_loss_weight
        inputs = (inputs_shared, inputs_posi, inputs_nega)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        if return_prediction:
            if self.task not in ("sft", "sft:train"):
                raise ValueError("denoiser prediction audit only supports SFT")
            shared, positive, _ = inputs
            shared = dict(shared)
            if disable_condition_modules:
                shared["force_condition_tensor"] = None
                shared["global_physics_tokens"] = None
                shared["oracle_slot_condition"] = None
                shared["object_time_graph_condition"] = None
                if getattr(self.pipe.dit, "phyparam_control_enabled", False):
                    if shared.get("sparse_object_condition") is None:
                        raise ValueError("PhyParam step-0 audit requires a nonempty condition")
                    shared["disable_phyparam_dense_residual"] = True
                else:
                    shared["sparse_object_condition"] = None
            return FlowMatchSFTLoss(
                self.pipe,
                return_prediction=True,
                **shared,
                **positive,
            )
        loss = self.task_to_loss[self.task](self.pipe, *inputs)
        return loss


def wan_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser = add_general_config(parser)
    parser = add_video_size_config(parser)
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer.")
    parser.add_argument(
        "--validation_cache_path",
        type=str,
        default=None,
        help="Preencoded validation cache. Required when :train has removed VAE/T5 units.",
    )
    parser.add_argument("--cache_override_manifest", type=str, default=None)
    parser.add_argument("--validation_cache_override_manifest", type=str, default=None)
    parser.add_argument(
        "--enable_prompt_context_override",
        default=False,
        action="store_true",
        help="Allow selected cached rows to replace only their positive T5 context "
             "while reusing the original VAE latents.",
    )
    parser.add_argument(
        "--drop_incomplete_global_batch",
        default=False,
        action="store_true",
        help="Shuffle without replacement and drop only the final incomplete "
             "world-size x microbatch group, preventing Accelerate padding duplicates.",
    )
    parser.add_argument("--audio_processor_path", type=str, default=None, help="Path to the audio processor. If provided, the processor will be used for Wan2.2-S2V model.")
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0, help="Max timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0, help="Min timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--initialize_model_on_cpu", default=False, action="store_true", help="Whether to initialize models on CPU.")
    parser.add_argument("--framewise_decoding", default=False, action="store_true", help="Enable it if this model is a WanToDance global model.")
    parser.add_argument(
        "--enable_friction_controlnet",
        default=False,
        action="store_true",
        help="Attach one five-block friction-map ControlNet branch after Wan loading.",
    )
    parser.add_argument(
        "--enable_force_controlnet",
        default=False,
        action="store_true",
        help="Attach one five-block official-channel-6:9 force ControlNet branch.",
    )
    parser.add_argument(
        "--force_condition_mode",
        choices=["latent_concat", "cross_attention"],
        default=None,
        help="Non-ControlNet numeric force interface attached after clean Wan loading.",
    )
    parser.add_argument(
        "--enable_global_physics_cross_attn",
        default=False,
        action="store_true",
        help="Attach the independent global g/mu/e physics cross-attention interface "
             "(variable-length scalar tokens, mounted at the same blocks as the F path "
             "but with fully independent weights).",
    )
    parser.add_argument(
        "--enable_oracle_slot_adapter",
        default=False,
        action="store_true",
        help="Attach the blocks 7/15/23 oracle-routed typed object-slot adapter.",
    )
    parser.add_argument(
        "--impact_loss_weight",
        type=float,
        default=0.1,
        help="Lambda multiplying the single balanced ImpactHead BCE.",
    )
    parser.add_argument(
        "--enable_object_time_graph_adapter",
        default=False,
        action="store_true",
        help="Attach eight learned-routing groups with two Object-Time Graph "
             "Blocks per group at Wan indices 2/6/10/14/18/22/25/27.",
    )
    parser.add_argument(
        "--object_attention_loss_weight",
        type=float,
        default=0.1,
        help="Lambda multiplying the single balanced future object localization BCE.",
    )
    parser.add_argument(
        "--reader_supervision_mode",
        choices=("balanced_bce", "mask_route_read"),
        default="balanced_bce",
        help=(
            "Reader localization objective. mask_route_read combines normalized "
            "future-mask spatial CE with actual final Top-K off-target read mass."
        ),
    )
    parser.add_argument(
        "--writer_loss_weight",
        type=float,
        default=0.0,
        help="Lambda multiplying independent Writer route/write supervision.",
    )
    parser.add_argument(
        "--writer_centroid_loss_weight",
        type=float,
        default=0.0,
        help="Lambda multiplying deterministic future-mask-centroid Writer CE.",
    )
    parser.add_argument(
        "--e_pair_event_loss_weight",
        type=float,
        default=0.0,
        help="Lambda multiplying loss-only object-object pair e event BCE.",
    )
    parser.add_argument(
        "--mu_pair_event_loss_weight",
        type=float,
        default=0.0,
        help="Lambda multiplying loss-only object-table pair mu event BCE.",
    )
    parser.add_argument(
        "--contact_loss_weight",
        type=float,
        default=0.1,
        help="Lambda multiplying the single balanced autonomous ContactHead BCE.",
    )
    parser.add_argument(
        "--base_contact_loss_weight", type=float, default=0.0,
        help="Lambda multiplying the base contact-head BCE.",
    )
    parser.add_argument(
        "--e_contact_loss_weight", type=float, default=0.0,
        help="Lambda multiplying the e-present contact-head BCE.",
    )
    parser.add_argument(
        "--mu_contact_loss_weight", type=float, default=0.0,
        help="Lambda multiplying the mu-present contact-head BCE.",
    )
    parser.add_argument(
        "--enable_sparse_object_interaction_adapter",
        default=False,
        action="store_true",
        help="Attach eight single-pass variable-object sparse interaction groups.",
    )
    parser.add_argument(
        "--enable_phyparam_control_dit",
        default=False,
        action="store_true",
        help="Attach the paper-faithful PhyParam unified Control-DiT branch.",
    )
    parser.add_argument(
        "--phyparam_num_control_blocks",
        type=int,
        default=None,
        help="Number of copied Wan blocks; default copies the full backbone.",
    )
    parser.add_argument(
        "--phyparam_harmonic_bands",
        type=int,
        default=8,
        help="Unpublished harmonic frequency count; must be frozen per run.",
    )
    parser.add_argument(
        "--phyparam_default_mass_normalized",
        type=float,
        default=0.5,
        help="Explicit constant mass token for current data, which has no mass field.",
    )
    parser.add_argument(
        "--phyparam_max_objects",
        type=int,
        default=8,
        help="Frozen upper bound for model-visible PhyParam objects.",
    )
    parser.add_argument(
        "--phyparam_dino_feature_dim",
        type=int,
        default=4096,
        help="Feature dimension of the frozen DINOv3 teacher target.",
    )
    parser.add_argument(
        "--phyparam_feature_tap_blocks",
        default="",
        help="Comma-separated zero-based Control-DiT feature taps; empty uses thirds.",
    )
    parser.add_argument(
        "--phyparam_flow_only_ablation",
        default=False,
        action="store_true",
        help="Flow-only loss ablation; not a matched-loss architecture comparison.",
    )
    parser.add_argument(
        "--phyparam_dino_target_grid",
        default="14,14",
        help="DINO target H,W used before the 4096-d projection heads.",
    )
    parser.add_argument(
        "--phyparam_temporal_min_norm",
        type=float,
        default=1e-6,
        help="Minimum teacher temporal-delta norm used by cosine supervision.",
    )
    parser.add_argument("--phyparam_dino_teacher_path", default=None)
    parser.add_argument("--phyparam_dino_teacher_sha256", default=None)
    parser.add_argument(
        "--phyparam_dino_feature_contract",
        default=None,
    )
    parser.add_argument("--phyparam_dino_decoded_frames", type=int, default=None)
    parser.add_argument(
        "--phyparam_dino_feature_frame_indices",
        default=None,
        help="Required comma-separated exact teacher frame indices in full-system mode.",
    )
    parser.add_argument(
        "--phyparam_feature_loss_weight",
        type=float,
        default=0.1,
        help="Unpublished lambda_1 for DINOv3 feature cosine loss.",
    )
    parser.add_argument(
        "--phyparam_temporal_loss_weight",
        type=float,
        default=0.1,
        help="Unpublished lambda_2 for temporal feature-difference cosine loss.",
    )
    parser.add_argument(
        "--sparse_object_adapter_architecture",
        choices=[
            "v2",
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
        ],
        default="v2",
        help="Sparse-object route: pooled v2 or same-object-temporal interaction.",
    )
    parser.add_argument(
        "--sparse_object_injection_blocks",
        default="2,6,10,14,18,22,25,27",
        help="Comma-separated zero-based Wan blocks that own sparse-object groups.",
    )
    parser.add_argument(
        "--dynamic_loss_weight",
        type=float,
        default=0.2,
        help="Lambda multiplying GT-latent dynamic-region weighted FM.",
    )
    parser.add_argument(
        "--temporal_loss_weight",
        type=float,
        default=0.2,
        help="Lambda multiplying first-order FM-residual temporal Huber.",
    )
    parser.add_argument(
        "--skip_step0_equivalence",
        default=False,
        action="store_true",
        help="Skip the clean-base step-0 equivalence probe for a non-zero weight-only warm-start.",
    )
    parser.add_argument(
        "--video_resize_mode",
        choices=["crop", "pad"],
        default="crop",
        help="Frame geometry contract. Oracle routes compiled from padded masks "
             "must use pad.",
    )
    return parser


if __name__ == "__main__":
    parser = wan_parser()
    args = parser.parse_args()
    dataloader_config = accelerate.DataLoaderConfiguration(
        use_seedable_sampler=True,
        data_seed=args.seed,
    )
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataloader_config=dataloader_config,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )
    accelerate.utils.set_seed(args.seed, device_specific=True)
    special_operator_map = {
        "animate_face_video": ToAbsolutePath(args.dataset_base_path) >> LoadVideo(args.num_frames, 4, 1, frame_processor=ImageCropAndResize(512, 512, None, 16, 16)),
        "wantodance_music_path": ToAbsolutePath(args.dataset_base_path),
        "friction_control_latents": ToAbsolutePath(args.dataset_base_path) >> LoadTorchPickle(),
        "force_control_latents": ToAbsolutePath(args.dataset_base_path) >> LoadTorchPickle(),
        "force_condition_tensor": ToAbsolutePath(args.dataset_base_path) >> LoadTorchPickle(),
        "oracle_slot_condition": ToAbsolutePath(args.dataset_base_path) >> LoadTorchPickle(),
        "object_time_graph_condition": ToAbsolutePath(args.dataset_base_path) >> LoadTorchPickle(),
        "sparse_object_condition": ToAbsolutePath(args.dataset_base_path) >> LoadTorchPickle(),
        "phyparam_dino_features": ToAbsolutePath(args.dataset_base_path) >> LoadTorchPickle(),
    }
    try:
        import librosa  # pylint: disable=unused-import,import-outside-toplevel
        special_operator_map["input_audio"] = ToAbsolutePath(args.dataset_base_path) >> LoadAudio(sr=16000)
    except ModuleNotFoundError:
        pass
    validate_phyparam_data_file_keys(
        args.enable_phyparam_control_dit, args.extra_inputs, args.data_file_keys
    )
    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=UnifiedDataset.default_video_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
            num_frames=args.num_frames,
            time_division_factor=4 if not args.framewise_decoding else 1,
            time_division_remainder=1 if not args.framewise_decoding else 0,
            resize_mode=args.video_resize_mode,
        ),
        special_operator_map=special_operator_map,
        cache_override_manifest=args.cache_override_manifest,
        cache_override_key=_select_cache_override_key(
            args.cache_override_manifest,
            enable_phyparam_control_dit=args.enable_phyparam_control_dit,
            enable_prompt_context_override=args.enable_prompt_context_override,
        ),
    )
    validation_dataset = None
    if args.validation_cache_path is not None:
        validation_dataset = UnifiedDataset(
            base_path=args.validation_cache_path,
            metadata_path=None,
            repeat=1,
            data_file_keys=(),
            cache_override_manifest=args.validation_cache_override_manifest,
            cache_override_key=_select_cache_override_key(
                args.validation_cache_override_manifest,
                enable_phyparam_control_dit=args.enable_phyparam_control_dit,
                enable_prompt_context_override=args.enable_prompt_context_override,
            ),
        )
    elif args.validation_metadata_path is not None:
        validation_dataset = UnifiedDataset(
            base_path=args.dataset_base_path,
            metadata_path=args.validation_metadata_path,
            repeat=1,
            data_file_keys=args.data_file_keys.split(","),
            main_data_operator=UnifiedDataset.default_video_operator(
                base_path=args.dataset_base_path,
                max_pixels=args.max_pixels,
                height=args.height,
                width=args.width,
                height_division_factor=16,
                width_division_factor=16,
                num_frames=args.num_frames,
                time_division_factor=4 if not args.framewise_decoding else 1,
                time_division_remainder=1 if not args.framewise_decoding else 0,
                resize_mode=args.video_resize_mode,
            ),
            special_operator_map=special_operator_map,
        )
    # CPU-offloaded PhyParam is not DDP-wrapped, so all ranks must construct
    # the same scratch Control-DiT before explicit gradient averaging begins.
    # Restore the normal device-specific stream immediately after construction.
    if args.enable_phyparam_control_dit:
        accelerate.utils.set_seed(args.seed, device_specific=False)
    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        audio_processor_path=args.audio_processor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        resume_from_checkpoint=args.resume_from_checkpoint,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        initial_sparse_object_checkpoint_dir=args.initial_sparse_object_checkpoint_dir,
        initial_sparse_object_checkpoint_sha256=args.initial_sparse_object_checkpoint_sha256,
        initial_sparse_object_checkpoint_step=args.initial_sparse_object_checkpoint_step,
        task=args.task,
        device="cpu" if (args.initialize_model_on_cpu or args.enable_model_cpu_offload) else accelerator.device,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        enable_friction_controlnet=args.enable_friction_controlnet,
        enable_force_controlnet=args.enable_force_controlnet,
        force_condition_mode=args.force_condition_mode,
        enable_global_physics_cross_attn=args.enable_global_physics_cross_attn,
        enable_oracle_slot_adapter=args.enable_oracle_slot_adapter,
        enable_object_time_graph_adapter=args.enable_object_time_graph_adapter,
        enable_sparse_object_interaction_adapter=args.enable_sparse_object_interaction_adapter,
        enable_phyparam_control_dit=args.enable_phyparam_control_dit,
        phyparam_num_control_blocks=args.phyparam_num_control_blocks,
        phyparam_harmonic_bands=args.phyparam_harmonic_bands,
        phyparam_default_mass_normalized=args.phyparam_default_mass_normalized,
        phyparam_max_objects=args.phyparam_max_objects,
        phyparam_dino_feature_dim=args.phyparam_dino_feature_dim,
        phyparam_feature_tap_blocks=args.phyparam_feature_tap_blocks,
        phyparam_flow_only_ablation=args.phyparam_flow_only_ablation,
        phyparam_dino_target_grid=args.phyparam_dino_target_grid,
        phyparam_temporal_min_norm=args.phyparam_temporal_min_norm,
        phyparam_dino_teacher_path=args.phyparam_dino_teacher_path,
        phyparam_dino_teacher_sha256=args.phyparam_dino_teacher_sha256,
        phyparam_dino_feature_contract=args.phyparam_dino_feature_contract,
        phyparam_dino_decoded_frames=args.phyparam_dino_decoded_frames,
        phyparam_dino_feature_frame_indices=(
            tuple(int(value) for value in args.phyparam_dino_feature_frame_indices.split(","))
            if args.phyparam_dino_feature_frame_indices else None
        ),
        phyparam_training_height=args.height,
        phyparam_training_width=args.width,
        phyparam_training_resize_mode=args.video_resize_mode,
        sparse_object_adapter_architecture=args.sparse_object_adapter_architecture,
        sparse_object_injection_blocks=args.sparse_object_injection_blocks,
        impact_loss_weight=args.impact_loss_weight,
        object_attention_loss_weight=args.object_attention_loss_weight,
        reader_supervision_mode=args.reader_supervision_mode,
        writer_loss_weight=args.writer_loss_weight,
        writer_centroid_loss_weight=args.writer_centroid_loss_weight,
        e_pair_event_loss_weight=args.e_pair_event_loss_weight,
        mu_pair_event_loss_weight=args.mu_pair_event_loss_weight,
        contact_loss_weight=args.contact_loss_weight,
        base_contact_loss_weight=args.base_contact_loss_weight,
        e_contact_loss_weight=args.e_contact_loss_weight,
        mu_contact_loss_weight=args.mu_contact_loss_weight,
        dynamic_loss_weight=args.dynamic_loss_weight,
        temporal_loss_weight=args.temporal_loss_weight,
        phyparam_feature_loss_weight=args.phyparam_feature_loss_weight,
        phyparam_temporal_loss_weight=args.phyparam_temporal_loss_weight,
    )
    if args.enable_phyparam_control_dit:
        accelerate.utils.set_seed(args.seed, device_specific=True)
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        enable_tensorboard_log=args.enable_tensorboard_log,
        enable_swanlab_log=args.enable_swanlab_log,
        swanlab_project=args.swanlab_project,
        enable_wandb_log=args.enable_wandb_log,
        wandb_project=args.wandb_project,
    )
    launcher_map = {
        "sft:data_process": launch_data_process_task,
        "direct_distill:data_process": launch_data_process_task,
        "sft": launch_training_task,
        "sft:train": launch_training_task,
        "direct_distill": launch_training_task,
        "direct_distill:train": launch_training_task,
    }
    launcher_map[args.task](
        accelerator,
        dataset,
        model,
        model_logger,
        validation_dataset=validation_dataset,
        args=args,
    )
