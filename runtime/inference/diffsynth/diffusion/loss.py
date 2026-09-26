from .base_pipeline import BasePipeline
from .sparse_object_losses import (
    dynamic_weighted_flow_loss,
    temporal_residual_loss,
)
import torch


PAIR_EVENT_LOSS_ONLY_KEYS = {
    "object_attention_labels",
    "pair_event_contract",
    "contact_contract",
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
    "pair_event_provenance",
    "contact_provenance",
}


def strip_pair_event_loss_only_inputs(inputs: dict) -> dict:
    """Remove every future or simulator-derived supervision tensor from forward."""
    condition = inputs.get("sparse_object_condition")
    if not isinstance(condition, dict) or not (PAIR_EVENT_LOSS_ONLY_KEYS & set(condition)):
        return inputs
    output = dict(inputs)
    output["sparse_object_condition"] = {
        key: value for key, value in condition.items() if key not in PAIR_EVENT_LOSS_ONLY_KEYS
    }
    return output


def _weight_three_contact_losses(
    base_loss,
    e_loss,
    mu_loss,
    base_valid_count,
    e_valid_count,
    mu_valid_count,
    base_weight,
    e_weight,
    mu_weight,
):
    active_base = (
        (base_valid_count > 0) & (float(base_weight) > 0.0)
    ).to(dtype=base_loss.dtype)
    active_e = ((e_valid_count > 0) & (float(e_weight) > 0.0)).to(
        dtype=e_loss.dtype
    )
    active_mu = ((mu_valid_count > 0) & (float(mu_weight) > 0.0)).to(
        dtype=mu_loss.dtype
    )
    active_head_count = (active_base + active_e + active_mu).clamp_min(1.0)
    return (
        float(base_weight) * base_loss * active_base / active_head_count,
        float(e_weight) * e_loss * active_e / active_head_count,
        float(mu_weight) * mu_loss * active_mu / active_head_count,
        active_head_count,
    )


def FlowMatchSFTLoss(
    pipe: BasePipeline,
    return_prediction: bool = False,
    impact_loss_weight: float = 0.1,
    object_attention_loss_weight: float = 0.1,
    reader_supervision_mode: str = "balanced_bce",
    writer_loss_weight: float = 0.0,
    writer_centroid_loss_weight: float = 0.0,
    e_pair_event_loss_weight: float = 0.0,
    mu_pair_event_loss_weight: float = 0.0,
    contact_loss_weight: float = 0.1,
    base_contact_loss_weight: float = 0.0,
    e_contact_loss_weight: float = 0.0,
    mu_contact_loss_weight: float = 0.0,
    dynamic_loss_weight: float = 0.2,
    temporal_loss_weight: float = 0.2,
    phyparam_feature_loss_weight: float = 0.1,
    phyparam_temporal_loss_weight: float = 0.1,
    **inputs,
):
    if "lora" in inputs:
        # Image-to-LoRA models need to load lora here.
        pipe.clear_lora(verbose=0)
        pipe.load_lora(pipe.dit, state_dict=inputs["lora"], hotload=True, verbose=0)

    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)
    
    noise = torch.randn_like(inputs["input_latents"]) * inputs.get("noise_scale", 1.0)
    inputs["latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
    training_target = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)
    
    if "first_frame_latents" in inputs:
        inputs["latents"][:, :, 0:1] = inputs["first_frame_latents"]
    
    sparse_object_condition = inputs.get("sparse_object_condition")
    model_inputs = strip_pair_event_loss_only_inputs(inputs)

    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    active_dit = models.get("dit", pipe.dit)
    noise_pred = pipe.model_fn(**models, **model_inputs, timestep=timestep)
    
    if "first_frame_latents" in inputs:
        noise_pred = noise_pred[:, :, 1:]
        training_target = training_target[:, :, 1:]
    
    flow_matching_loss = torch.nn.functional.mse_loss(
        noise_pred.float(), training_target.float()
    )
    flow_matching_loss = (
        flow_matching_loss * pipe.scheduler.training_weight(timestep)
    )
    if return_prediction:
        return flow_matching_loss, noise_pred

    phyparam_branch = getattr(active_dit, "phyparam_control_dit", None)
    if getattr(active_dit, "phyparam_control_enabled", False) and getattr(
        phyparam_branch, "feature_supervision_enabled", False
    ):
        dino_target = inputs.get("phyparam_dino_features")
        if dino_target is None:
            raise ValueError(
                "PhyParam full-system training requires phyparam_dino_features"
            )
        feature_loss, feature_temporal_loss, feature_metrics = (
            active_dit.phyparam_feature_losses(dino_target)
        )
        weighted_feature = float(phyparam_feature_loss_weight) * feature_loss
        weighted_feature_temporal = (
            float(phyparam_temporal_loss_weight) * feature_temporal_loss
        )
        total = flow_matching_loss + weighted_feature + weighted_feature_temporal
        active_dit.phyparam_last_losses = {
            "flow_matching": flow_matching_loss.detach(),
            "dino_feature_cosine": feature_loss.detach(),
            "dino_temporal_cosine": feature_temporal_loss.detach(),
            "weighted_dino_feature": weighted_feature.detach(),
            "weighted_dino_temporal": weighted_feature_temporal.detach(),
            "feature_loss_weight_effective": flow_matching_loss.new_tensor(
                float(phyparam_feature_loss_weight)
            ),
            "temporal_loss_weight_effective": flow_matching_loss.new_tensor(
                float(phyparam_temporal_loss_weight)
            ),
            "objective_total": total.detach(),
            **feature_metrics,
        }
        return total

    if sparse_object_condition is not None and not getattr(
        pipe.dit, "phyparam_control_enabled", False
    ):
        required_labels = {"object_attention_labels", "object_valid_mask"}
        missing = required_labels - set(sparse_object_condition)
        if missing:
            raise ValueError(
                f"sparse_object_condition is missing labels {sorted(missing)}"
            )
        object_loss, finder_metrics = pipe.dit.sparse_object_localization_loss(
            sparse_object_condition["object_attention_labels"],
            sparse_object_condition["object_valid_mask"],
            supervision_mode=reader_supervision_mode,
        )
        writer_loss = None
        writer_centroid_loss = None
        writer_metrics = {}
        if getattr(pipe.dit, "sparse_object_writer_outputs", None):
            writer_loss, writer_metrics = pipe.dit.sparse_object_writer_loss(
                sparse_object_condition["object_attention_labels"],
                sparse_object_condition["object_valid_mask"],
            )
            if float(writer_centroid_loss_weight) > 0.0:
                writer_centroid_loss, centroid_metrics = (
                    pipe.dit.sparse_object_writer_loss(
                        sparse_object_condition["object_attention_labels"],
                        sparse_object_condition["object_valid_mask"],
                        supervision_mode="centroid_ce",
                    )
                )
                writer_metrics.update(
                    {
                        f"writer_centroid_{key}": value
                        for key, value in centroid_metrics.items()
                    }
                )
        adapter = getattr(pipe.dit, "sparse_object_adapter", None)
        hard_support_loss = flow_matching_loss.new_zeros(())
        weighted_hard_support = flow_matching_loss.new_zeros(())
        hard_support_metrics = {}
        if getattr(adapter, "hard_support_enabled", False):
            hard_support_loss, hard_support_metrics = (
                pipe.dit.sparse_object_hard_support_loss(
                    sparse_object_condition["object_attention_labels"],
                    sparse_object_condition["object_valid_mask"],
                )
            )
            weighted_hard_support = (
                float(adapter.hard_support_loss_weight) * hard_support_loss
            )
        dynamic_loss, dynamic_stats = dynamic_weighted_flow_loss(
            noise_pred,
            training_target,
            inputs["input_latents"],
        )
        temporal_loss = temporal_residual_loss(noise_pred, training_target)
        training_weight = pipe.scheduler.training_weight(timestep)
        dynamic_loss = dynamic_loss * training_weight
        temporal_loss = temporal_loss * training_weight
        weighted_object = float(object_attention_loss_weight) * object_loss
        weighted_writer = (
            float(writer_loss_weight) * writer_loss
            if writer_loss is not None
            else flow_matching_loss.new_zeros(())
        )
        weighted_writer_centroid = (
            float(writer_centroid_loss_weight) * writer_centroid_loss
            if writer_centroid_loss is not None
            else flow_matching_loss.new_zeros(())
        )
        weighted_dynamic = float(dynamic_loss_weight) * dynamic_loss
        weighted_temporal = float(temporal_loss_weight) * temporal_loss
        e_pair_event_loss = flow_matching_loss.new_zeros(())
        mu_pair_event_loss = flow_matching_loss.new_zeros(())
        contact_loss = flow_matching_loss.new_zeros(())
        base_contact_loss = flow_matching_loss.new_zeros(())
        e_contact_loss = flow_matching_loss.new_zeros(())
        mu_contact_loss = flow_matching_loss.new_zeros(())
        pair_event_metrics = {}
        pair_event_supervision = getattr(adapter, "pair_event_supervision", False)
        shared_contact_gate = getattr(adapter, "shared_contact_gate", False)
        three_contact_gates = getattr(adapter, "three_contact_gates", False)
        if three_contact_gates:
            required_contact = {
                "contact_label",
                "contact_valid",
                "e_contact_label",
                "e_contact_valid",
                "mu_contact_label",
                "mu_contact_valid",
                "edge_restitution_present",
                "edge_mu_present",
            }
            missing_contact = required_contact - set(sparse_object_condition)
            if missing_contact:
                raise ValueError(
                    f"three-contact labels missing {sorted(missing_contact)}"
                )
            (
                base_contact_loss,
                e_contact_loss,
                mu_contact_loss,
                pair_event_metrics,
            ) = pipe.dit.sparse_object_three_contact_losses(
                sparse_object_condition["contact_label"],
                sparse_object_condition["contact_valid"],
                sparse_object_condition["e_contact_label"],
                sparse_object_condition["e_contact_valid"],
                sparse_object_condition["mu_contact_label"],
                sparse_object_condition["mu_contact_valid"],
                sparse_object_condition["object_valid_mask"],
                sparse_object_condition["edge_restitution_present"],
                sparse_object_condition["edge_mu_present"],
            )
        elif shared_contact_gate and float(contact_loss_weight) > 0.0:
            required_contact = {"contact_label", "contact_valid"}
            missing_contact = required_contact - set(sparse_object_condition)
            if missing_contact:
                raise ValueError(
                    f"shared contact labels missing {sorted(missing_contact)}"
                )
            contact_loss, pair_event_metrics = pipe.dit.sparse_object_contact_loss(
                sparse_object_condition["contact_label"],
                sparse_object_condition["contact_valid"],
                sparse_object_condition["object_valid_mask"],
            )
        elif pair_event_supervision:
            required_pair = {
                "e_pair_event_labels",
                "e_pair_event_valid",
                "mu_pair_event_labels",
                "mu_pair_event_valid",
            }
            missing_pair = required_pair - set(sparse_object_condition)
            if missing_pair:
                raise ValueError(
                    f"pair event labels missing {sorted(missing_pair)}"
                )
            e_pair_event_loss, mu_pair_event_loss, pair_event_metrics = (
                pipe.dit.sparse_object_pair_event_losses(
                    sparse_object_condition["e_pair_event_labels"],
                    sparse_object_condition["e_pair_event_valid"],
                    sparse_object_condition["mu_pair_event_labels"],
                    sparse_object_condition["mu_pair_event_valid"],
                    sparse_object_condition["object_valid_mask"],
                )
            )
        weighted_e_pair_event = float(e_pair_event_loss_weight) * e_pair_event_loss
        weighted_mu_pair_event = float(mu_pair_event_loss_weight) * mu_pair_event_loss
        weighted_contact = float(contact_loss_weight) * contact_loss
        if three_contact_gates:
            (
                weighted_base_contact,
                weighted_e_contact,
                weighted_mu_contact,
                active_head_count,
            ) = _weight_three_contact_losses(
                base_contact_loss,
                e_contact_loss,
                mu_contact_loss,
                pair_event_metrics["base_contact_valid_count"],
                pair_event_metrics["e_contact_valid_count"],
                pair_event_metrics["mu_contact_valid_count"],
                base_contact_loss_weight,
                e_contact_loss_weight,
                mu_contact_loss_weight,
            )
        else:
            active_head_count = flow_matching_loss.new_zeros(())
            weighted_base_contact = (
                float(base_contact_loss_weight) * base_contact_loss
            )
            weighted_e_contact = float(e_contact_loss_weight) * e_contact_loss
            weighted_mu_contact = float(mu_contact_loss_weight) * mu_contact_loss
        write_gates = getattr(pipe.dit, "sparse_object_write_gates", None)
        gate_metrics = {}
        if write_gates:
            flattened_gates = torch.cat(
                [gate.detach().reshape(-1).float() for gate in write_gates]
            )
            active_gates = flattened_gates[flattened_gates > 0]
            if active_gates.numel() == 0:
                active_gates = flattened_gates
            gate_metrics = {
                "writer_gate_mean": active_gates.mean(),
                "writer_gate_std": active_gates.std(unbiased=False),
            }
        total = (
            flow_matching_loss
            + weighted_object
            + weighted_writer
            + weighted_writer_centroid
            + weighted_hard_support
            + weighted_dynamic
            + weighted_temporal
            + weighted_e_pair_event
            + weighted_mu_pair_event
            + weighted_contact
            + weighted_base_contact
            + weighted_e_contact
            + weighted_mu_contact
        )
        pipe.dit.sparse_object_last_losses = {
            "flow_matching": flow_matching_loss.detach(),
            "object_localization_bce": object_loss.detach(),
            "dynamic_weighted_fm": dynamic_loss.detach(),
            "temporal_residual_huber": temporal_loss.detach(),
            "weighted_object_localization": weighted_object.detach(),
            "writer_routing": (
                writer_loss.detach()
                if writer_loss is not None
                else flow_matching_loss.new_zeros(())
            ),
            "weighted_writer_routing": weighted_writer.detach(),
            "writer_loss_weight_effective": flow_matching_loss.new_tensor(
                float(writer_loss_weight)
            ),
            "writer_centroid_routing": (
                writer_centroid_loss.detach()
                if writer_centroid_loss is not None
                else flow_matching_loss.new_zeros(())
            ),
            "weighted_writer_centroid_routing": weighted_writer_centroid.detach(),
            "writer_centroid_loss_weight_effective": flow_matching_loss.new_tensor(
                float(writer_centroid_loss_weight)
            ),
            "hard_support_gate_objective": hard_support_loss.detach(),
            "weighted_hard_support_gate": weighted_hard_support.detach(),
            "hard_support_gate_loss_weight_effective": flow_matching_loss.new_tensor(
                float(getattr(adapter, "hard_support_loss_weight", 0.0))
                if getattr(adapter, "hard_support_enabled", False)
                else 0.0
            ),
            "weighted_dynamic": weighted_dynamic.detach(),
            "weighted_temporal": weighted_temporal.detach(),
            "e_pair_event_objective": e_pair_event_loss.detach(),
            "contact_bce": contact_loss.detach(),
            "weighted_contact": weighted_contact.detach(),
            "contact_loss_weight_effective": flow_matching_loss.new_tensor(
                float(contact_loss_weight)
            ),
            "base_contact_bce": base_contact_loss.detach(),
            "e_contact_bce": e_contact_loss.detach(),
            "mu_contact_bce": mu_contact_loss.detach(),
            "weighted_base_contact": weighted_base_contact.detach(),
            "weighted_e_contact": weighted_e_contact.detach(),
            "weighted_mu_contact": weighted_mu_contact.detach(),
            "base_contact_loss_weight_effective": flow_matching_loss.new_tensor(
                float(base_contact_loss_weight)
            ),
            "e_contact_loss_weight_effective": flow_matching_loss.new_tensor(
                float(e_contact_loss_weight)
            ),
            "mu_contact_loss_weight_effective": flow_matching_loss.new_tensor(
                float(mu_contact_loss_weight)
            ),
            "three_contact_active_head_count": active_head_count.detach(),
            **(
                {"e_pair_event_expected_distance": e_pair_event_loss.detach()}
                if getattr(
                    getattr(pipe.dit, "sparse_object_adapter", None),
                    "e_pair_event_loss_mode",
                    "hard_slot",
                ) == "event_distribution_w1"
                else {"e_pair_event_bce": e_pair_event_loss.detach()}
            ),
            "mu_pair_event_bce": mu_pair_event_loss.detach(),
            "weighted_e_pair_event": weighted_e_pair_event.detach(),
            "weighted_mu_pair_event": weighted_mu_pair_event.detach(),
            "e_pair_event_loss_weight_effective": flow_matching_loss.new_tensor(
                float(e_pair_event_loss_weight)
            ),
            "mu_pair_event_loss_weight_effective": flow_matching_loss.new_tensor(
                float(mu_pair_event_loss_weight)
            ),
            "objective_total": total.detach(),
            **finder_metrics,
            **writer_metrics,
            **hard_support_metrics,
            **gate_metrics,
            **pair_event_metrics,
            **dynamic_stats,
        }
        return total

    object_time_graph_condition = inputs.get("object_time_graph_condition")
    if object_time_graph_condition is not None:
        required_labels = {"object_attention_labels", "contact_labels"}
        missing = required_labels - set(object_time_graph_condition)
        if missing:
            raise ValueError(
                "object_time_graph_condition is missing labels "
                f"{sorted(missing)}"
            )
        object_loss, contact_loss = (
            pipe.dit.object_time_graph_auxiliary_losses(
                object_time_graph_condition["object_attention_labels"],
                object_time_graph_condition["contact_labels"],
            )
        )
        pipe.dit.object_time_graph_last_losses = {
            "flow_matching": flow_matching_loss.detach(),
            "object_localization_bce": object_loss.detach(),
            "contact_bce": contact_loss.detach(),
        }
        return (
            flow_matching_loss
            + float(object_attention_loss_weight) * object_loss
            + float(contact_loss_weight) * contact_loss
        )

    oracle_condition = inputs.get("oracle_slot_condition")
    if oracle_condition is None:
        return flow_matching_loss
    if "impact_labels" not in oracle_condition:
        raise ValueError("oracle_slot_condition is missing impact_labels")
    impact_loss = pipe.dit.oracle_slot_impact_loss(
        oracle_condition["impact_labels"]
    )
    pipe.dit.oracle_slot_last_losses = {
        "flow_matching": flow_matching_loss.detach(),
        "impact_bce": impact_loss.detach(),
    }
    return flow_matching_loss + float(impact_loss_weight) * impact_loss


def FlowMatchSFTAudioVideoLoss(pipe: BasePipeline, **inputs):
    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)
    
    # video
    noise = torch.randn_like(inputs["input_latents"])
    inputs["video_latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
    training_target = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)
    
    # audio
    if inputs.get("audio_input_latents") is not None:
        audio_noise = torch.randn_like(inputs["audio_input_latents"])
        inputs["audio_latents"] = pipe.scheduler.add_noise(inputs["audio_input_latents"], audio_noise, timestep)
        training_target_audio = pipe.scheduler.training_target(inputs["audio_input_latents"], audio_noise, timestep)

    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    noise_pred, noise_pred_audio = pipe.model_fn(**models, **inputs, timestep=timestep)

    loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
    loss = loss * pipe.scheduler.training_weight(timestep)
    if inputs.get("audio_input_latents") is not None:
        loss_audio = torch.nn.functional.mse_loss(noise_pred_audio.float(), training_target_audio.float())
        loss_audio = loss_audio * pipe.scheduler.training_weight(timestep)
        loss = loss + loss_audio
    return loss


def DirectDistillLoss(pipe: BasePipeline, **inputs):
    pipe.scheduler.set_timesteps(inputs["num_inference_steps"])
    pipe.scheduler.training = True
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
        timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
        noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep, progress_id=progress_id)
        inputs["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred, **inputs)
    loss = torch.nn.functional.mse_loss(inputs["latents"].float(), inputs["input_latents"].float())
    return loss


class TrajectoryImitationLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.initialized = False
    
    def initialize(self, device):
        import lpips # TODO: remove it
        self.loss_fn = lpips.LPIPS(net='alex').to(device)
        self.initialized = True

    def fetch_trajectory(self, pipe: BasePipeline, timesteps_student, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        trajectory = [inputs_shared["latents"].clone()]

        pipe.scheduler.set_timesteps(num_inference_steps, target_timesteps=timesteps_student)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )
            inputs_shared["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred.detach(), **inputs_shared)

            trajectory.append(inputs_shared["latents"].clone())
        return pipe.scheduler.timesteps, trajectory
    
    def align_trajectory(self, pipe: BasePipeline, timesteps_teacher, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        loss = 0
        pipe.scheduler.set_timesteps(num_inference_steps, training=True)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)

            progress_id_teacher = torch.argmin((timesteps_teacher - timestep).abs())
            inputs_shared["latents"] = trajectory_teacher[progress_id_teacher]

            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )

            sigma = pipe.scheduler.sigmas[progress_id]
            sigma_ = 0 if progress_id + 1 >= len(pipe.scheduler.timesteps) else pipe.scheduler.sigmas[progress_id + 1]
            if progress_id + 1 >= len(pipe.scheduler.timesteps):
                latents_ = trajectory_teacher[-1]
            else:
                progress_id_teacher = torch.argmin((timesteps_teacher - pipe.scheduler.timesteps[progress_id + 1]).abs())
                latents_ = trajectory_teacher[progress_id_teacher]
            
            denom = sigma_ - sigma
            denom = torch.sign(denom) * torch.clamp(denom.abs(), min=1e-6)
            target = (latents_ - inputs_shared["latents"]) / denom
            loss = loss + torch.nn.functional.mse_loss(noise_pred.float(), target.float()) * pipe.scheduler.training_weight(timestep)
        return loss
    
    def compute_regularization(self, pipe: BasePipeline, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        inputs_shared["latents"] = trajectory_teacher[0]
        pipe.scheduler.set_timesteps(num_inference_steps)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )
            inputs_shared["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred.detach(), **inputs_shared)

        image_pred = pipe.vae_decoder(inputs_shared["latents"])
        image_real = pipe.vae_decoder(trajectory_teacher[-1])
        loss = self.loss_fn(image_pred.float(), image_real.float())
        return loss

    def forward(self, pipe: BasePipeline, inputs_shared, inputs_posi, inputs_nega):
        if not self.initialized:
            self.initialize(pipe.device)
        with torch.no_grad():
            pipe.scheduler.set_timesteps(8)
            timesteps_teacher, trajectory_teacher = self.fetch_trajectory(inputs_shared["teacher"], pipe.scheduler.timesteps, inputs_shared, inputs_posi, inputs_nega, 50, 2)
            timesteps_teacher = timesteps_teacher.to(dtype=pipe.torch_dtype, device=pipe.device)
        loss_1 = self.align_trajectory(pipe, timesteps_teacher, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, 8, 1)
        loss_2 = self.compute_regularization(pipe, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, 8, 1)
        loss = loss_1 + loss_2
        return loss
