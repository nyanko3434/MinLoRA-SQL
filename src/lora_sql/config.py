"""Strict training-run config: YAML -> frozen dataclasses.

Unknown keys, missing keys and wrong types all raise, so a typo in a config can never silently fall
back to a default. The held-out rule is enforced here in code: only train_spider.json /
train_others.json may ever be training data.
"""

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from types import NoneType, UnionType
from typing import Union, get_args, get_origin, get_type_hints

import yaml

ALLOWED_TRAIN_FILES = frozenset({"train_spider.json", "train_others.json"})
_QUANT_MODES = frozenset({"nf4", "none"})
_DTYPES = frozenset({"float32", "bfloat16", "float16"})


@dataclass(frozen=True)
class ModelConfig:
    name_or_path: str
    quant: str  # nf4 | none
    double_quant: bool
    dtype: str
    gradient_checkpointing: bool


@dataclass(frozen=True)
class LoRAConfig:
    target_modules: tuple[str, ...]
    rank: int
    alpha: int
    dropout: float


@dataclass(frozen=True)
class DataConfig:
    spider_root: str
    train_files: tuple[str, ...]
    max_seq_len: int

    @property
    def train_paths(self) -> list[Path]:
        return [Path(self.spider_root) / f for f in self.train_files]


@dataclass(frozen=True)
class TrainConfig:
    micro_batch_size: int
    grad_accum_steps: int
    epochs: int
    max_steps: int | None
    lr: float
    weight_decay: float
    warmup_ratio: float
    max_grad_norm: float
    log_every: int
    save_every: int
    time_budget_hours: float
    out_dir: str


@dataclass(frozen=True)
class RunConfig:
    run_name: str
    seed: int
    model: ModelConfig
    lora: LoRAConfig
    data: DataConfig
    train: TrainConfig


def _coerce(value, tp, path: str):
    if dataclasses.is_dataclass(tp):
        return _build(tp, value, path)

    origin = get_origin(tp)
    if origin in (Union, UnionType):  # only `X | None` is used
        if value is None and NoneType in get_args(tp):
            return None
        (inner,) = [a for a in get_args(tp) if a is not NoneType]
        return _coerce(value, inner, path)
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{path}: expected a list, got {value!r}")
        (item_tp, _ellipsis) = get_args(tp)
        return tuple(_coerce(v, item_tp, f"{path}[{i}]") for i, v in enumerate(value))
    if tp is float:
        # YAML 1.1 parses "2e-4" (no dot) as a string, so coerce rather than type-check.
        if isinstance(value, bool):
            raise ValueError(f"{path}: expected a float, got {value!r}")
        try:
            return float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{path}: expected a float, got {value!r}") from None
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{path}: expected an int, got {value!r}")
        return value
    if tp in (bool, str):
        if not isinstance(value, tp):
            raise ValueError(f"{path}: expected {tp.__name__}, got {value!r}")
        return value
    raise TypeError(f"{path}: unsupported field type {tp!r}")


def _build(cls, data, path: str):
    if not isinstance(data, dict):
        raise ValueError(f"{path or 'config'}: expected a mapping, got {data!r}")
    hints = get_type_hints(cls)
    names = [f.name for f in dataclasses.fields(cls)]
    prefix = f"{path}." if path else ""

    unknown = sorted(set(data) - set(names))
    if unknown:
        raise ValueError(f"unknown config key(s): {', '.join(prefix + k for k in unknown)}")
    missing = [n for n in names if n not in data]
    if missing:
        raise ValueError(f"missing config key(s): {', '.join(prefix + k for k in missing)}")

    return cls(**{n: _coerce(data[n], hints[n], prefix + n) for n in names})


def _validate(cfg: RunConfig) -> None:
    if cfg.model.quant not in _QUANT_MODES:
        raise ValueError(f"model.quant must be one of {sorted(_QUANT_MODES)}, got {cfg.model.quant!r}")
    if cfg.model.dtype not in _DTYPES:
        raise ValueError(f"model.dtype must be one of {sorted(_DTYPES)}, got {cfg.model.dtype!r}")
    if cfg.model.quant == "nf4" and cfg.model.dtype != "bfloat16":
        raise ValueError(f"model.quant=nf4 computes in bfloat16; model.dtype must be bfloat16, got {cfg.model.dtype!r}")

    if not cfg.data.train_files:
        raise ValueError("data.train_files is empty")
    # Exact-name match: rejects dev.json / test.json, and also path tricks like ../dev.json.
    bad = [f for f in cfg.data.train_files if f not in ALLOWED_TRAIN_FILES]
    if bad:
        raise ValueError(
            f"data.train_files may only contain {sorted(ALLOWED_TRAIN_FILES)}; got {bad}. "
            "dev is eval-only and test is held out."
        )

    if cfg.train.micro_batch_size != 1:
        raise ValueError(
            f"train.micro_batch_size must be 1 (completion_loss assumes no padding), got {cfg.train.micro_batch_size}"
        )


def config_from_dict(data: dict) -> RunConfig:
    cfg = _build(RunConfig, data, "")
    _validate(cfg)
    return cfg


def load_config(path: str | Path) -> RunConfig:
    with open(path, "r", encoding="utf-8") as f:
        return config_from_dict(yaml.safe_load(f))
