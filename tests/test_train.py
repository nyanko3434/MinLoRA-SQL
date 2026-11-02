"""Training loop on the 0.5B fp32 fixture (quant none): token-weighted accumulation equals one big
batch, the LR schedule and step count follow the spec, and an end-to-end smoke run logs, saves,
honours the time budget and refuses to clobber a finished run."""

import gc
import json
import math
import shutil
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import yaml
from conftest import MODEL_NAME

from lora_sql.checkpoint import load_lora
from lora_sql.config import config_from_dict, load_config
from lora_sql.data import build_train_example, load_examples
from lora_sql.lora import LoRALinear, inject_lora
from lora_sql.qlora import completion_loss, prepare_for_training
from lora_sql.train import accumulate_window, compute_total_steps, lr_lambda, main, warmup_steps

ROOT = Path(__file__).resolve().parents[1]
SPIDER_ROOT = ROOT / "data" / "spider_data"
TRAIN_PATH = SPIDER_ROOT / "train_spider.json"


def _free(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()


def _lora_grads(lora):
    return [t.grad.clone() for m in lora for t in (m.lora_A, m.lora_B)]


def test_accumulate_window_matches_single_pass(fresh_model, base_model_and_tokenizer):
    _, tokenizer = base_model_and_tokenizer
    rows = [build_train_example(tokenizer, ex) for ex in load_examples(TRAIN_PATH)[:4]]

    model = prepare_for_training(fresh_model())
    inject_lora(model, target_modules=("q_proj", "v_proj"), rank=8, alpha=16, dropout=0.0)
    lora = [m for m in model.modules() if isinstance(m, LoRALinear)]
    torch.manual_seed(0)
    for m in lora:  # with lora_B at zero, lora_A.grad is zero and the comparison is vacuous
        nn.init.normal_(m.lora_B, std=0.02)

    loss_sum, n_tokens = accumulate_window(model, rows)
    accumulated = _lora_grads(lora)
    model.zero_grad(set_to_none=True)

    # Reference: one scalar (sum of CE sums) / (sum of tokens), one backward.
    total_loss, total_n = 0.0, 0
    for ids, n in rows:
        example_loss, k = completion_loss(model, torch.tensor([ids], device="cuda"), n)
        total_loss = total_loss + example_loss
        total_n += k
    (total_loss / total_n).backward()
    reference = _lora_grads(lora)

    assert n_tokens == total_n == sum(n for _, n in rows)
    assert math.isclose(loss_sum, total_loss.item(), rel_tol=1e-6)
    assert all(g.abs().sum() > 0 for g in reference)
    for g_acc, g_ref in zip(accumulated, reference):
        assert torch.allclose(g_acc, g_ref, atol=1e-6)
    _free(model)


def test_lr_lambda():
    total = 536
    warm = warmup_steps(0.03, total)
    assert warm == 16
    assert lr_lambda(0, warm, total) == 1 / warm
    assert lr_lambda(warm - 1, warm, total) == 1.0
    assert lr_lambda(warm, warm, total) == 1.0
    expected = 0.5 * (1 + math.cos(math.pi * (total - 1 - warm) / (total - warm)))
    assert lr_lambda(total - 1, warm, total) == pytest.approx(expected, rel=1e-12)
    assert lr_lambda(total, warm, total) == pytest.approx(0.0, abs=1e-12)
    # total == warm: LambdaLR still evaluates step == total after the last step (the value is never
    # used by an optimizer step); it must not divide by zero.
    assert warmup_steps(0.03, 3) == 1
    assert math.isfinite(lr_lambda(1, 1, 1))


def test_total_steps_floor():
    assert compute_total_steps(8577, 16, 1, None) == 536
    assert compute_total_steps(8577, 16, 2, None) == 1072
    assert compute_total_steps(8577, 16, 1, 3) == 3


def _smoke_cfg(out_dir, **train_overrides):
    d = yaml.safe_load((ROOT / "configs" / "train" / "qv_r8.yaml").read_text())
    d["model"].update(name_or_path=MODEL_NAME, quant="none", dtype="float32")
    d["data"]["spider_root"] = str(SPIDER_ROOT)  # cwd-independent
    d["train"].update(grad_accum_steps=2, max_steps=3, save_every=2, log_every=1, out_dir=str(out_dir))
    d["train"].update(train_overrides)
    return config_from_dict(d)


def _log(out_dir):
    lines = [json.loads(l) for l in (Path(out_dir) / "log.jsonl").read_text().splitlines()]
    return [l for l in lines if "event" not in l], [l for l in lines if "event" in l]


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory, batch):
    out_dir = tmp_path_factory.mktemp("smoke")
    cfg = _smoke_cfg(out_dir)
    model = main(cfg, limit_examples=8)
    model.eval()  # dropout (0.05 from qv_r8) must be off for the logit comparison
    with torch.no_grad():
        logits = model(**batch).logits.cpu()
    _free(model)
    return out_dir, cfg, logits


def test_smoke_run(smoke_run, fresh_model, batch):
    out_dir, cfg, trained_logits = smoke_run

    rows, events = _log(out_dir)
    assert [r["step"] for r in rows] == [1, 2, 3]
    assert all(math.isfinite(r["loss"]) for r in rows)
    assert events == []
    for d in ("step_2", "final"):
        assert (out_dir / d / "adapter.safetensors").exists()

    report = json.loads((out_dir / "data_report.json").read_text())
    assert report["dropped_by_db_id"] == {"baseball_1": 82}
    assert report["n_kept"] == 8577 and report["n_used"] == 8
    assert load_config(out_dir / "config.yaml") == cfg

    loaded = load_lora(fresh_model(), out_dir / "final").eval()
    with torch.no_grad():
        assert torch.equal(loaded(**batch).logits.cpu(), trained_logits)
    _free(loaded)


def test_budget_guard(tmp_path):
    model = main(_smoke_cfg(tmp_path, time_budget_hours=1e-9), limit_examples=8)
    _free(model)

    rows, events = _log(tmp_path)
    assert [r["step"] for r in rows] == [1]
    assert events == [{"event": "stopped_by_budget", "step": 1}]
    assert (tmp_path / "final" / "adapter.safetensors").exists()
    assert json.loads((tmp_path / "final" / "lora_config.json").read_text())["step"] == 1


def test_no_clobber(smoke_run, tmp_path):
    out_dir = tmp_path / "run"
    shutil.copytree(smoke_run[0], out_dir)  # a finished run, final/ included
    cfg = _smoke_cfg(out_dir)

    with pytest.raises(FileExistsError, match="final"):
        main(cfg, limit_examples=8)
    model = main(cfg, limit_examples=8, overwrite=True)
    _free(model)
    rows, _ = _log(out_dir)
    assert [r["step"] for r in rows] == [1, 2, 3]
    assert (out_dir / "final" / "adapter.safetensors").exists()
