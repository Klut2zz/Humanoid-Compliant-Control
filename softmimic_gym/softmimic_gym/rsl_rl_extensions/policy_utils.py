"""Minimal policy utilities for staged general-tracking fine-tuning."""

from __future__ import annotations

import torch


# new change
def configure_actor_output_row_finetune(policy, action_ids) -> dict[str, object]:
    """Freeze actor features and update only selected final-output rows.

    PyTorch cannot set ``requires_grad`` per matrix row, so the final linear
    layer remains optimizer-visible while hooks zero every unselected row.
    This strictly preserves the learned lower-body controller.
    """

    mlp = getattr(policy, "mlp", None)
    if mlp is None:
        raise TypeError("The loaded actor has no MLP for output-row fine-tuning.")
    linear_layers = [module for module in mlp.modules() if isinstance(module, torch.nn.Linear)]
    if not linear_layers:
        raise TypeError("The loaded actor MLP contains no linear output layer.")
    output_layer = linear_layers[-1]
    selected_ids = tuple(int(action_id) for action_id in action_ids)
    if not selected_ids or len(set(selected_ids)) != len(selected_ids):
        raise ValueError("Trainable actor output ids must be non-empty and unique.")
    if min(selected_ids) < 0 or max(selected_ids) >= output_layer.out_features:
        raise ValueError(
            f"Actor output ids must be in [0, {output_layer.out_features - 1}], "
            f"got {selected_ids}."
        )

    for handle in getattr(policy, "_general_tracking_output_gradient_hooks", ()):
        handle.remove()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    output_layer.weight.requires_grad_(True)
    if output_layer.bias is not None:
        output_layer.bias.requires_grad_(True)

    row_mask = torch.zeros(
        output_layer.out_features,
        device=output_layer.weight.device,
        dtype=output_layer.weight.dtype,
    )
    row_mask[list(selected_ids)] = 1.0
    hooks = [
        output_layer.weight.register_hook(
            lambda gradient, mask=row_mask: gradient * mask[:, None]
        )
    ]
    if output_layer.bias is not None:
        hooks.append(
            output_layer.bias.register_hook(
                lambda gradient, mask=row_mask: gradient * mask
            )
        )
    policy._general_tracking_output_gradient_hooks = tuple(hooks)

    frozen_ids = tuple(
        action_id
        for action_id in range(output_layer.out_features)
        if action_id not in selected_ids
    )
    return {
        "output_features": output_layer.out_features,
        "trainable_action_ids": selected_ids,
        "frozen_action_ids": frozen_ids,
        "trainable_parameter_values": len(selected_ids)
        * (output_layer.in_features + int(output_layer.bias is not None)),
    }


# new change
def configure_bilateral_structured_arm_adapter_finetune(policy) -> dict[str, object]:
    """Freeze the legacy whole-body actor and train only both arm adapters."""

    left = getattr(policy, "left_arm_residual_adapter", None)
    right = getattr(policy, "right_arm_residual_adapter", None)
    if left is None or right is None:
        raise TypeError("The actor does not contain both structured arm adapters.")
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    for adapter in (left, right):
        for parameter in adapter.parameters():
            parameter.requires_grad_(True)
    return {
        "left_trainable_values": sum(p.numel() for p in left.parameters()),
        "right_trainable_values": sum(p.numel() for p in right.parameters()),
        "left_limits": tuple(float(v) for v in policy.left_arm_adapter_limits.cpu()),
        "right_limits": tuple(float(v) for v in policy.right_arm_adapter_limits.cpu()),
    }


# new change
def configure_proximal_arm_waist_adapter_finetune(policy) -> dict[str, object]:
    """Train shoulder/elbow heads plus the tiny waist adapter only."""

    left = getattr(policy, "left_arm_residual_adapter", None)
    right = getattr(policy, "right_arm_residual_adapter", None)
    waist = getattr(policy, "waist_residual_adapter", None)
    if left is None or right is None or waist is None:
        raise TypeError(
            "The actor must contain bilateral arm adapters and a waist adapter."
        )
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    for adapter in (left, right):
        for head_name in ("shoulder", "elbow"):
            for parameter in adapter[head_name].parameters():
                parameter.requires_grad_(True)
    for parameter in waist.parameters():
        parameter.requires_grad_(True)
    return {
        "left_proximal_trainable_values": sum(
            parameter.numel()
            for head_name in ("shoulder", "elbow")
            for parameter in left[head_name].parameters()
        ),
        "right_proximal_trainable_values": sum(
            parameter.numel()
            for head_name in ("shoulder", "elbow")
            for parameter in right[head_name].parameters()
        ),
        "waist_trainable_values": sum(parameter.numel() for parameter in waist.parameters()),
        "waist_limits": tuple(float(value) for value in policy.waist_adapter_limits.cpu()),
    }


# new change
def configure_waist_support_adapter_finetune(policy) -> dict[str, object]:
    """Freeze legacy actor/arms and train only V3 waist/support heads."""

    waist = getattr(policy, "waist_residual_adapter", None)
    support = getattr(policy, "support_residual_adapter", None)
    if waist is None or support is None:
        raise TypeError("The actor must contain V3 waist and support adapters.")
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    for adapter in (waist, support):
        for parameter in adapter.parameters():
            parameter.requires_grad_(True)
    return {
        "waist_trainable_values": sum(
            parameter.numel() for parameter in waist.parameters()
        ),
        "support_trainable_values": sum(
            parameter.numel() for parameter in support.parameters()
        ),
        "waist_limits": tuple(
            float(value) for value in policy.waist_adapter_limits.cpu()
        ),
        "support_limits": tuple(
            float(value) for value in policy.support_adapter_limits.cpu()
        ),
        "support_action_ids": tuple(
            int(value) for value in policy.support_adapter_action_ids.cpu()
        ),
    }


# new change
def configure_stationary_support_adapter_finetune(policy) -> dict[str, object]:
    """Freeze model_1375 plus arm/waist heads; train only support correction."""

    support = getattr(policy, "support_residual_adapter", None)
    if support is None:
        raise TypeError("The actor must contain a support residual adapter.")
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    for parameter in support.parameters():
        parameter.requires_grad_(True)
    return {
        "support_trainable_values": sum(
            parameter.numel() for parameter in support.parameters()
        ),
        "support_limits": tuple(
            float(value) for value in policy.support_adapter_limits.cpu()
        ),
        "support_action_ids": tuple(
            int(value) for value in policy.support_adapter_action_ids.cpu()
        ),
    }


# new change
def configure_online_compliance_adapter_finetune(policy) -> dict[str, object]:
    """Freeze model_1375 and train only the online force-conditioned head."""

    adapter = getattr(policy, "compliance_residual_adapter", None)
    if adapter is None:
        raise TypeError("The actor does not contain an online compliance adapter.")
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    for parameter in adapter.parameters():
        parameter.requires_grad_(True)
    return {
        "trainable_values": sum(p.numel() for p in adapter.parameters()),
        "limits": tuple(float(v) for v in policy.compliance_adapter_limits.cpu()),
        "action_ids": tuple(int(v) for v in policy.compliance_adapter_action_ids.cpu()),
    }


# new change
def configure_online_compliance_arm_only_finetune(policy) -> dict[str, object]:
    """Freeze model_1375, support and waist; train only both arm force heads."""

    adapter = getattr(policy, "compliance_residual_adapter", None)
    if adapter is None:
        raise TypeError("The actor does not contain an online compliance adapter.")
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    trainable_heads = ("left_arm", "right_arm")
    for head_name in trainable_heads:
        for parameter in adapter[head_name].parameters():
            parameter.requires_grad_(True)
    waist_values = sum(parameter.numel() for parameter in adapter["waist"].parameters())
    trainable_values = sum(
        parameter.numel()
        for parameter in policy.parameters()
        if parameter.requires_grad
    )
    return {
        "trainable_values": trainable_values,
        "frozen_waist_values": waist_values,
        "trainable_heads": trainable_heads,
        "limits": tuple(float(v) for v in policy.compliance_adapter_limits.cpu()),
        "action_ids": tuple(int(v) for v in policy.compliance_adapter_action_ids.cpu()),
    }


# new change
def configure_online_compliance_waist_only_finetune(
    policy, *, waist_limit_rad: float = 0.002
) -> dict[str, object]:
    """Freeze the accepted force-gated arms and train only a tiny waist head."""

    if not (0.0 < waist_limit_rad <= 0.004):
        raise ValueError("waist_limit_rad must be in (0, 0.004].")
    adapter = getattr(policy, "compliance_residual_adapter", None)
    if adapter is None:
        raise TypeError("The actor does not contain an online compliance adapter.")
    if not hasattr(policy, "compliance_force_active_index"):
        raise TypeError("Waist coordination requires the force-gated actor.")
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    for parameter in adapter["waist"].parameters():
        parameter.requires_grad_(True)
    # Waist target action scale is 0.20 rad per actor unit.  Reassert the
    # physical 2-mrad authority after loading model_1394's older 4-mrad buffer.
    with torch.no_grad():
        policy.compliance_adapter_limits[:3].fill_(waist_limit_rad / 0.20)
    return {
        "trainable_values": sum(
            parameter.numel()
            for parameter in policy.parameters()
            if parameter.requires_grad
        ),
        "trainable_heads": ("waist",),
        "waist_limit_rad": waist_limit_rad,
        "frozen_arm_values": sum(
            parameter.numel()
            for name in ("left_arm", "right_arm")
            for parameter in adapter[name].parameters()
        ),
        "action_ids": (2, 5, 8),
    }


# new change
def online_compliance_adapter_zero_summary(policy) -> dict[str, float]:
    """Audit exact zero initialization before loading/training."""

    adapter = getattr(policy, "compliance_residual_adapter", None)
    if adapter is None:
        raise TypeError("The actor does not contain an online compliance adapter.")
    output_layers = [head[-1] for head in adapter.values()]
    return {
        "output_weight_abs_max": max(
            float(layer.weight.detach().abs().max()) for layer in output_layers
        ),
        "output_bias_abs_max": max(
            float(layer.bias.detach().abs().max()) for layer in output_layers
        ),
    }


# new change
def configure_structured_action_std(
    policy,
    action_ids,
    *,
    active_std: float,
    frozen_std: float = 1.0e-4,
) -> dict[str, object]:
    """Keep effective exploration only on selected physical action rows."""

    if active_std <= 0.0 or frozen_std <= 0.0:
        raise ValueError("Structured exploration std values must be positive.")
    distribution = getattr(policy, "distribution", None)
    if distribution is None:
        raise TypeError("The policy has no Gaussian output distribution.")
    if hasattr(distribution, "std_param"):
        parameter = distribution.std_param
        is_log = False
    elif hasattr(distribution, "log_std_param"):
        parameter = distribution.log_std_param
        is_log = True
    else:
        raise TypeError("Expected Gaussian std_param or log_std_param.")
    selected_ids = tuple(int(action_id) for action_id in action_ids)
    if not selected_ids or len(set(selected_ids)) != len(selected_ids):
        raise ValueError("Structured exploration ids must be non-empty and unique.")
    if min(selected_ids) < 0 or max(selected_ids) >= parameter.numel():
        raise ValueError("Structured exploration action id is out of range.")
    with torch.no_grad():
        if is_log:
            parameter.fill_(torch.log(torch.tensor(frozen_std)).item())
            parameter[list(selected_ids)] = torch.log(
                torch.tensor(active_std, device=parameter.device)
            )
            actual = parameter.detach().exp()
        else:
            parameter.fill_(frozen_std)
            parameter[list(selected_ids)] = active_std
            actual = parameter.detach()
    return {
        "active_action_ids": selected_ids,
        "active_std": float(actual[list(selected_ids)].mean().item()),
        "frozen_std": float(
            actual[
                [index for index in range(actual.numel()) if index not in selected_ids]
            ].mean().item()
        ),
    }


# new change
def waist_adapter_zero_summary(policy) -> dict[str, float]:
    """Audit the new waist output layers before any optimization."""

    waist = getattr(policy, "waist_residual_adapter", None)
    if waist is None:
        raise TypeError("The actor does not contain a waist residual adapter.")
    output_layers = [head[-1] for head in waist.values()]
    return {
        "output_weight_abs_max": max(
            float(layer.weight.detach().abs().max()) for layer in output_layers
        ),
        "output_bias_abs_max": max(
            float(layer.bias.detach().abs().max()) for layer in output_layers
        ),
    }


# new change
def support_adapter_zero_summary(policy) -> dict[str, float]:
    """Audit V3 support output layers before any optimization."""

    support = getattr(policy, "support_residual_adapter", None)
    if support is None:
        raise TypeError("The actor does not contain a support residual adapter.")
    output_layers = [head[-1] for head in support.values()]
    return {
        "output_weight_abs_max": max(
            float(layer.weight.detach().abs().max()) for layer in output_layers
        ),
        "output_bias_abs_max": max(
            float(layer.bias.detach().abs().max()) for layer in output_layers
        ),
    }


# new change
def bilateral_structured_arm_adapter_zero_summary(policy) -> dict[str, float]:
    """Audit the six output layers that must be exactly zero at bootstrap."""

    output_layers = []
    for name in ("left_arm_residual_adapter", "right_arm_residual_adapter"):
        adapter = getattr(policy, name, None)
        if adapter is None:
            raise TypeError(f"Missing {name}.")
        output_layers.extend(head[-1] for head in adapter.values())
    return {
        "output_weight_abs_max": max(
            float(layer.weight.detach().abs().max()) for layer in output_layers
        ),
        "output_bias_abs_max": max(
            float(layer.bias.detach().abs().max()) for layer in output_layers
        ),
    }


# new change
def load_checkpoint_with_zero_adapter_compat(
    runner, checkpoint_path: str, *, load_cfg: dict[str, bool] | None = None
) -> dict[str, object]:
    """Strictly load a legacy actor while allowing only new adapter entries."""

    policy = runner.alg.get_policy()
    if not hasattr(policy, "left_arm_residual_adapter"):
        runner.load(checkpoint_path, load_cfg=load_cfg)
        return {"legacy_bootstrap": False, "missing_actor_keys": ()}

    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    current_keys = set(policy.state_dict())
    checkpoint_keys = set(checkpoint["actor_state_dict"])
    missing = tuple(sorted(current_keys - checkpoint_keys))
    unexpected = tuple(sorted(checkpoint_keys - current_keys))
    allowed_missing = {
        key
        for key in current_keys
        if key.startswith("left_arm_residual_adapter.")
        or key.startswith("right_arm_residual_adapter.")
        or key.startswith("left_arm_adapter_")
        or key.startswith("right_arm_adapter_")
        # new change
        or key.startswith("waist_residual_adapter.")
        or key.startswith("waist_adapter_")
        # new change
        or key.startswith("support_residual_adapter.")
        or key.startswith("support_adapter_")
        # new change
        or key.startswith("support_candidate_")
        or key.startswith("support_action_scale_")
        # new change
        # Direct support heads normalize only their private 87D input and
        # optionally scale only their zero-initialized output.  These buffers
        # are deterministic defaults and do not alter the legacy actor.
        or key.startswith("support_feature_mean")
        or key.startswith("support_feature_scale")
        or key.startswith("support_output_scale")
        # new change
        or key.startswith("compliance_residual_adapter.")
        or key.startswith("compliance_adapter_")
        # new change
        or key.startswith("compliance_force_")
        # new change
        # Inference-only q/dq feedback buffers are deterministic constants;
        # permitting only this explicit prefix keeps legacy model_1375 loading
        # strict for every nominal actor parameter.
        or key.startswith("analytic_position_gain")
        or key.startswith("analytic_velocity_gain_s")
        or key.startswith("analytic_arm_target_limit_rad")
        or key.startswith("analytic_left_feature_ids")
        or key.startswith("analytic_right_feature_ids")
    }
    if unexpected or set(missing) - allowed_missing:
        raise RuntimeError(
            "Structured-adapter checkpoint compatibility failed: "
            f"missing={missing}, unexpected={unexpected}."
        )
    legacy_bootstrap = bool(missing)
    compliance_bootstrap = any(
        key.startswith("compliance_residual_adapter.")
        or key.startswith("compliance_adapter_")
        for key in missing
    )
    effective_load_cfg = load_cfg
    if legacy_bootstrap:
        effective_load_cfg = effective_load_cfg or {
            "actor": True,
            "critic": not compliance_bootstrap,
            "optimizer": False,
            "iteration": True,
            "rnd": False,
        }
        if compliance_bootstrap:
            # Online force context adds 20 critic inputs.  A model_1375 critic
            # cannot be shape-compatible and is intentionally reinitialized.
            effective_load_cfg = dict(effective_load_cfg)
            effective_load_cfg["critic"] = False
            effective_load_cfg["optimizer"] = False
        if effective_load_cfg.get("optimizer", False):
            raise ValueError("Legacy bootstrap requires a fresh optimizer.")
    runner.load(
        checkpoint_path,
        load_cfg=effective_load_cfg,
        strict=not legacy_bootstrap,
    )
    return {"legacy_bootstrap": legacy_bootstrap, "missing_actor_keys": missing}
