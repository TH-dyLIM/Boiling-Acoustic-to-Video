from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def add_unet_lora(unet: torch.nn.Module, rank: int = 32, alpha: int | None = None) -> torch.nn.Module:
    """Attach PEFT LoRA adapters to SVD UNet attention projections.

    Some diffusers UNet classes expose ``add_adapter`` directly, while the SVD
    UNetSpatioTemporalConditionModel in current Windows installs often does not.
    In that case PEFT can still wrap the module and insert LoRA layers by target
    module name.
    """
    try:
        from peft import LoraConfig, get_peft_model
    except Exception as exc:  # pragma: no cover - runtime dependency guard
        raise RuntimeError("peft is required for LoRA training. Install requirements_svd_lora.txt") from exc

    alpha = int(alpha if alpha is not None else rank)
    config = LoraConfig(
        r=int(rank),
        lora_alpha=alpha,
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    if hasattr(unet, "add_adapter"):
        unet.add_adapter(config)
        return unet
    return get_peft_model(unet, config)


def trainable_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: param.detach().cpu()
        for name, param in module.named_parameters()
        if param.requires_grad
    }


def load_trainable_state_dict(module: torch.nn.Module, state_dict: dict[str, torch.Tensor]) -> tuple[list[str], list[str]]:
    current = module.state_dict()
    filtered = {k: v for k, v in state_dict.items() if k in current}
    missing, unexpected = module.load_state_dict(filtered, strict=False)
    return list(missing), list(unexpected)


def save_checkpoint(
    path: str | Path,
    unet: torch.nn.Module,
    conditioner: torch.nn.Module,
    args: dict[str, Any],
    step: int,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": int(step),
            "args": dict(args),
            "unet_lora": trainable_state_dict(unet),
            "conditioner": conditioner.state_dict(),
        },
        path,
    )


def load_checkpoint(path: str | Path, unet: torch.nn.Module, conditioner: torch.nn.Module, map_location: str = "cpu") -> dict[str, Any]:
    ckpt = torch.load(path, map_location=map_location)
    load_trainable_state_dict(unet, ckpt["unet_lora"])
    conditioner.load_state_dict(ckpt["conditioner"], strict=False)
    return ckpt
