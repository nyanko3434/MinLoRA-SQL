"""Session 4: toy LoRA training loop, ours (inject_lora) vs peft (get_peft_model),
same init/data/hyperparameters, to produce a supporting training-curve figure for the
Session 3 numerical parity result. Qwen2.5-0.5B-Instruct fixture only, never Coder-7B."""
import copy
import json

import torch
from peft import LoraConfig, get_peft_model
from peft.tuners.lora import Linear as PeftLoraLinear
from transformers import AutoModelForCausalLM, AutoTokenizer

from lora_sql.data import format_prompt, load_examples
from lora_sql.lora import LoRALinear, inject_lora

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
TARGET_MODULES = ("q_proj", "v_proj")
RANK, ALPHA = 8, 16
N_STEPS = 200
LR = 2e-4
SEED = 0

torch.manual_seed(SEED)

print("=== setup ===")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
base_model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
lora_cfg = LoraConfig(
    r=RANK, lora_alpha=ALPHA, target_modules=list(TARGET_MODULES),
    lora_dropout=0.0, use_rslora=False, bias="none",
)

toy = load_examples("data/spider_data/train_spider.json")[:N_STEPS]
print(f"toy examples: {len(toy)}")


def build_example(ex, tokenizer):
    prompt_ids = tokenizer(format_prompt(ex), add_special_tokens=False).input_ids
    completion_ids = tokenizer(ex.query, add_special_tokens=False).input_ids + [tokenizer.eos_token_id]
    input_ids = prompt_ids + completion_ids
    labels = [-100] * len(prompt_ids) + completion_ids
    return input_ids, labels


print("=== pre-tokenizing ===")
tokenized = [build_example(ex, tokenizer) for ex in toy]

# Identical adapter init: build a peft model once (CPU, no forward pass needed), snapshot
# its lora_A, then discard. Both training runs below start from a *fresh* build with this
# same snapshot copied into lora_A; lora_B is left at its default zero-init on both sides
# (NOT randomized -- that's a parity-test-only technique, see tests/test_lora_parity.py;
# real training needs lora_B=0).
print("=== snapshotting peft lora_A init ===")
snapshot_source = get_peft_model(copy.deepcopy(base_model), lora_cfg)
lora_a_snapshot = {
    name.removeprefix("base_model.model."): module.lora_A["default"].weight.data.clone()
    for name, module in snapshot_source.named_modules()
    if isinstance(module, PeftLoraLinear)
}
del snapshot_source
print(f"snapshotted {len(lora_a_snapshot)} lora_A tensors")


def build_ours():
    model = inject_lora(
        copy.deepcopy(base_model), target_modules=TARGET_MODULES,
        rank=RANK, alpha=ALPHA, dropout=0.0,
    ).cuda()
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            module.lora_A.data.copy_(lora_a_snapshot[name])
    return model


def build_peft():
    model = get_peft_model(copy.deepcopy(base_model), lora_cfg).cuda()
    for name, module in model.named_modules():
        if isinstance(module, PeftLoraLinear):
            module.lora_A["default"].weight.data.copy_(lora_a_snapshot[name.removeprefix("base_model.model.")])
    return model


BUILDERS = {"ours": build_ours, "peft": build_peft}


def train(model, tag):
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=0.0)
    model.train()
    losses = []
    for step, (input_ids, labels) in enumerate(tokenized):
        input_ids_t = torch.tensor([input_ids], device="cuda")
        labels_t = torch.tensor([labels], device="cuda")
        attention_mask = torch.ones_like(input_ids_t)
        out = model(input_ids=input_ids_t, attention_mask=attention_mask, labels=labels_t)
        out.loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(out.loss.item())
        if step % 20 == 0:
            print(f"[{tag}] step {step:3d}  loss {out.loss.item():.4f}")
    return losses


for impl in ["ours", "peft"]:
    print(f"\n=== training: {impl} ===")
    torch.manual_seed(SEED)
    model = BUILDERS[impl]()
    losses = train(model, impl)
    with open(f"results/toy_loss_{impl}.json", "w") as f:
        json.dump(losses, f)
    del model
    torch.cuda.empty_cache()

print("\n=== PASS ===")
