from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn


@dataclass
class LMOutput:
    logits: torch.Tensor
    auxiliary_losses: Mapping[str, torch.Tensor] = field(default_factory=dict)


Builder = Callable[[Mapping[str, Any]], nn.Module]
ParameterGroups = Callable[[nn.Module, float], list[dict[str, Any]]]
FlopEstimator = Callable[[nn.Module, int], int]


@dataclass(frozen=True)
class ModelSpec:
    builder: Builder
    parameter_groups: ParameterGroups | None = None
    flop_estimator: FlopEstimator | None = None


_REGISTRY: dict[str, ModelSpec] = {}


def register_model(
    name: str,
    builder: Builder,
    parameter_groups: ParameterGroups | None = None,
    flop_estimator: FlopEstimator | None = None,
) -> None:
    if name in _REGISTRY:
        raise ValueError(f"model {name!r} is already registered")
    _REGISTRY[name] = ModelSpec(builder, parameter_groups, flop_estimator)


def get_model_spec(name: str) -> ModelSpec:
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        choices = ", ".join(sorted(_REGISTRY)) or "<none>"
        raise KeyError(f"unknown model {name!r}; registered models: {choices}") from exc


def build_model(name: str, params: Mapping[str, Any]) -> tuple[nn.Module, ModelSpec]:
    spec = get_model_spec(name)
    model = spec.builder(params)
    if not isinstance(model, nn.Module):
        raise TypeError(f"builder for {name!r} did not return torch.nn.Module")
    return model, spec


def default_parameter_groups(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim < 2 or name.endswith("bias"):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def non_embedding_parameter_count(model: nn.Module) -> int:
    embedding_ids = {
        id(parameter)
        for module in model.modules()
        if isinstance(module, nn.Embedding)
        for parameter in module.parameters(recurse=False)
    }
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if id(parameter) not in embedding_ids
    )


def estimate_training_flops(
    model: nn.Module, spec: ModelSpec, training_tokens: int
) -> int:
    if spec.flop_estimator is not None:
        return int(spec.flop_estimator(model, training_tokens))
    return 6 * non_embedding_parameter_count(model) * training_tokens
