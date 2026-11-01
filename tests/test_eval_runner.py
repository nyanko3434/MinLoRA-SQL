"""Runner end to end on the 0.5B fp32 fixture: checkpoint loading goes through load_lora (logits
equal to loading it directly), the base path injects nothing, and a smoke run_eval writes the
results JSON + per-example JSONL. GPU: do not run while a training run holds the GPU."""

import gc
import json
from pathlib import Path

import torch
import torch.nn as nn

from lora_sql.checkpoint import load_lora, save_lora
from lora_sql.eval.runner import load_config, load_model_for_eval, run_eval
from lora_sql.lora import LoRALinear, inject_lora

ROOT = Path(__file__).resolve().parents[1]


def _smoke_cfg():
    cfg = load_config(ROOT / "configs" / "week1_parity.yaml")
    for key in ("eval_path", "db_dir"):  # cwd-independent
        cfg["data"][key] = str(ROOT / cfg["data"][key])
    return cfg


def _free(*models):
    del models
    gc.collect()
    torch.cuda.empty_cache()


def _checkpoint(fresh_model, tmp_path):
    model = inject_lora(fresh_model(), target_modules=("q_proj", "v_proj"), rank=8, alpha=16, dropout=0.0)
    torch.manual_seed(0)
    for m in model.modules():
        if isinstance(m, LoRALinear):
            nn.init.normal_(m.lora_B, std=0.02)
    ckpt = save_lora(model, tmp_path / "ckpt", {"step": 0, "config_sha256": "test"})
    _free(model)
    return ckpt


def test_load_model_for_eval_uses_load_lora(fresh_model, batch, tmp_path):
    ckpt = _checkpoint(fresh_model, tmp_path)
    model, _ = load_model_for_eval(_smoke_cfg(), str(ckpt))
    reference = load_lora(fresh_model(), ckpt).eval()
    assert not model.training
    with torch.no_grad():
        assert torch.equal(model(**batch).logits, reference(**batch).logits)
    _free(model, reference)


def test_load_model_for_eval_base_injects_nothing():
    model, _ = load_model_for_eval(_smoke_cfg())
    assert not any(isinstance(m, LoRALinear) for m in model.modules())
    _free(model)


def test_run_eval_smoke(tmp_path):
    out = tmp_path / "eval.json"
    summary = run_eval(_smoke_cfg(), out=str(out), limit=3)
    assert summary["total"] == 3
    assert 0 <= summary["correct"] <= 3 and 0 <= summary["n_pred_exec_errors"] <= 3

    report = json.loads(out.read_text())
    assert len(report["results"]) == 3
    assert report["meta"]["limit"] == 3 and report["meta"]["lora_checkpoint"] is None
    lines = out.with_suffix(".jsonl").read_text().splitlines()
    assert len(lines) == 3 and all("exec_error" in json.loads(l) for l in lines)
