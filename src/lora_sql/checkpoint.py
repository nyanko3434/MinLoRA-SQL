"""LoRA checkpoints: adapter.safetensors (only lora_ keys, under our module names) plus a
lora_config.json sidecar that is enough to rebuild the adapter on a fresh base.

load_lora is the ONLY place a LoRA state dict is loaded. load_state_dict(strict=False) silently
loading nothing (a key-prefix mismatch, a wrong rank or target set) would evaluate the untrained
base while reporting success, so every way that can happen is asserted against here.
"""

import json
import subprocess
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file

from lora_sql.lora import LoRALinear, inject_lora

ADAPTER_FILE = "adapter.safetensors"
SIDECAR_FILE = "lora_config.json"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_REQUIRED_META = ("step", "config_sha256")


def _git_commit() -> str:
    try:
        def git(*args):
            return subprocess.run(
                ["git", *args], cwd=_REPO_ROOT, capture_output=True, text=True, check=True
            ).stdout.strip()

        commit = git("rev-parse", "HEAD")
        return commit + "+dirty" if git("status", "--porcelain") else commit
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _adapter_spec(model: nn.Module) -> dict:
    """target_modules / rank / alpha / dropout / adapter_dtype, read from the LoRALinear modules
    themselves rather than trusted from a caller. Every wrapped layer must agree."""
    targets, specs = set(), set()
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            targets.add(name.rsplit(".", 1)[-1])
            p = module.lora_dropout.p if isinstance(module.lora_dropout, nn.Dropout) else 0.0
            specs.add((module.r, module.alpha, p, str(module.lora_A.dtype).removeprefix("torch.")))
    assert targets, "model has no LoRALinear modules; nothing to save"
    assert len(specs) == 1, f"LoRA layers disagree on (rank, alpha, dropout, dtype): {sorted(specs)}"
    ((rank, alpha, dropout, adapter_dtype),) = specs
    return {"target_modules": sorted(targets), "rank": rank, "alpha": alpha, "dropout": dropout,
            "adapter_dtype": adapter_dtype}


def save_lora(model: nn.Module, out_dir: str | Path, meta: dict) -> Path:
    state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items() if "lora_" in k}
    assert state, "no lora_ keys in the model state dict"
    assert all("lora_" in k for k in state)
    missing_meta = [k for k in _REQUIRED_META if k not in meta]
    assert missing_meta == [], f"meta is missing {missing_meta}"

    sidecar = {"base_model": model.config.name_or_path, **_adapter_spec(model)}
    clash = sorted(set(meta) & set(sidecar))
    assert clash == [], f"meta may not override fields read from the model: {clash}"
    sidecar.update(
        git_commit=_git_commit(),
        n_keys=len(state),
        n_params=sum(v.numel() for v in state.values()),
        **meta,
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(state, out_dir / ADAPTER_FILE)
    (out_dir / SIDECAR_FILE).write_text(json.dumps(sidecar, indent=2))
    return out_dir


def load_lora(model: nn.Module, ckpt_dir: str | Path) -> nn.Module:
    """Inject adapters into an UN-injected base from the sidecar, then load and verify them."""
    ckpt_dir = Path(ckpt_dir)
    sidecar = json.loads((ckpt_dir / SIDECAR_FILE).read_text())
    assert sidecar["base_model"] == model.config.name_or_path, (
        f"checkpoint was trained on {sidecar['base_model']!r}, model is {model.config.name_or_path!r}"
    )

    # Rebuilding from the sidecar means a wrong rank or target set fails below, not silently.
    inject_lora(
        model,
        target_modules=tuple(sidecar["target_modules"]),
        rank=sidecar["rank"],
        alpha=sidecar["alpha"],
        dropout=sidecar["dropout"],
        adapter_dtype=getattr(torch, sidecar["adapter_dtype"]),
    )

    state = load_file(ckpt_dir / ADAPTER_FILE)
    current = model.state_dict()
    # Without this, a shape mismatch surfaces as torch's size-mismatch RuntimeError.
    bad = [
        f"{k}: saved {tuple(v.shape)} {v.dtype}, model {tuple(current[k].shape)} {current[k].dtype}"
        for k, v in state.items()
        if k in current and (current[k].shape != v.shape or current[k].dtype != v.dtype)
    ]
    assert bad == [], "saved tensors don't match the injected model:\n" + "\n".join(bad[:5])

    res = model.load_state_dict(state, strict=False)
    assert res.unexpected_keys == [], (
        f"{len(res.unexpected_keys)} unexpected keys (prefix mismatch?), e.g. {res.unexpected_keys[:3]}"
    )
    unconsumed = sorted(set(state) & set(res.missing_keys))
    assert unconsumed == [], f"{len(unconsumed)} saved keys were not consumed, e.g. {unconsumed[:3]}"
    model_lora = {n for n, _ in model.named_parameters() if "lora_" in n}
    diff = sorted(model_lora ^ set(state))
    assert not diff, f"model lora_ params and saved keys differ ({len(diff)}), e.g. {diff[:3]}"

    loaded = model.state_dict()
    unequal = [k for k in state if not torch.equal(loaded[k].cpu(), state[k])]
    assert unequal == [], f"{len(unequal)} tensors differ after load, first {unequal[0]}"
    assert any(v.any() for k, v in state.items() if "lora_B" in k), (
        "adapter is a no-op (every lora_B is zero)"
    )
    return model
