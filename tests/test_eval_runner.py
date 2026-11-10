"""Runner end to end on the 0.5B fp32 fixture: checkpoint loading goes through load_lora (logits
equal to loading it directly), the base path injects nothing, generation is true greedy, and a
smoke run_eval writes the results JSON + per-example JSONL. GPU: do not run while a training run
holds the GPU."""

import gc
import json
from pathlib import Path

import torch
import torch.nn as nn
from conftest import requires_gpu, requires_spider

from lora_sql.checkpoint import load_lora, save_lora
from lora_sql.data import encode_prompt, load_examples
from lora_sql.eval.runner import generate_sql, load_config, load_model_for_eval, run_eval
from lora_sql.lora import LoRALinear, inject_lora

pytestmark = [requires_gpu, requires_spider]

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
    assert report["meta"]["decoding"] == {"do_sample": False, "repetition_penalty": 1.0}
    rows = [json.loads(l) for l in out.with_suffix(".jsonl").read_text().splitlines()]
    assert len(rows) == 3 and all("exec_error" in r for r in rows)
    assert all(r["n_new_tokens"] > 0 and isinstance(r["truncated"], bool) for r in rows)
    assert summary["n_truncated"] == sum(r["truncated"] for r in rows)


@torch.no_grad()
def _argmax_decode(model, tokenizer, example, max_new_tokens):
    """Greedy decoding written out by hand: no generate(), so no generation_config and no logits
    processors can be involved."""
    ids = torch.tensor([encode_prompt(tokenizer, example)], device=model.device)
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    new = []
    for _ in range(max_new_tokens):
        token = model(input_ids=ids).logits[0, -1].argmax().item()
        new.append(token)
        if token in eos_ids:
            break
        ids = torch.cat([ids, torch.tensor([[token]], device=ids.device)], dim=1)
    return tokenizer.decode(new, skip_special_tokens=True).strip(), len(new)


def test_generate_sql_is_true_greedy():
    """At repetition_penalty 1.0 the runner's generation equals a hand-written argmax loop. At the
    1.1 that Qwen's generation_config ships (what generate() uses when the argument is left out)
    it does not, which is why the argument is always passed."""
    cfg = _smoke_cfg()
    model, tokenizer = load_model_for_eval(cfg)
    assert model.generation_config.repetition_penalty == 1.1  # the default this guards against
    examples = load_examples(cfg["data"]["eval_path"])[:4]

    differs = 0
    for example in examples:
        expected, n_expected = _argmax_decode(model, tokenizer, example, 64)
        sql, n_new, truncated = generate_sql(model, tokenizer, example, 64, model.device, 1.0)
        assert (sql, n_new) == (expected, n_expected)
        assert truncated == (n_new == 64)
        differs += generate_sql(model, tokenizer, example, 64, model.device, 1.1)[0] != sql
    assert differs > 0, "repetition_penalty 1.1 changed nothing on these prompts; the guard is untested"
    _free(model)
