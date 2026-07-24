"""Session 1 environment check. Fails loudly on anything that would bite later."""
import torch
import transformers
import peft
import bitsandbytes as bnb
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# --- versions -----------------------------------------------------------
print("=== versions ===")
print(f"torch          {torch.__version__}")
print(f"transformers   {transformers.__version__}")
print(f"peft           {peft.__version__}")
print(f"bitsandbytes   {bnb.__version__}")

# --- gpu ----------------------------------------------------------------
print("\n=== gpu ===")
assert torch.cuda.is_available(), "CUDA not available"
props = torch.cuda.get_device_properties(0)
print(f"device         {props.name}")
print(f"vram           {props.total_memory / 1e9:.1f} GB")
print(f"capability     {props.major}.{props.minor}")
print(f"bf16 support   {torch.cuda.is_bf16_supported()}")

# --- bitsandbytes (the Week 2 dependency most likely to break) ----------
print("\n=== bitsandbytes ===")
layer = bnb.nn.Linear4bit(64, 64, compute_dtype=torch.bfloat16).cuda()
out = layer(torch.randn(2, 64, device="cuda", dtype=torch.bfloat16))
print(f"forward ok     {tuple(out.shape)}  dtype={out.dtype}")
print(f"layer type     {type(layer).__name__}")
print(f"isinstance nn.Linear: {isinstance(layer, torch.nn.Linear)}  (expect True)")
print(f"in/out features: {layer.in_features}/{layer.out_features}  "
      f"(trust these; weight.shape is packed: {tuple(layer.weight.shape)})")

# --- model load + generate ---------------------------------------------
print("\n=== model ===")
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).cuda()
inputs = tok.apply_chat_template(
    [{"role": "user", "content": "Say hi in five words."}],
    add_generation_prompt=True, return_tensors="pt", return_dict=True,
).to("cuda")
gen = model.generate(**inputs, max_new_tokens=24, do_sample=False)
print(tok.decode(gen[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip())

# --- target modules for inject_lora ------------------------------------
print("\n=== lora targets (layer 0) ===")
targets = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
for name, mod in model.named_modules():
    if "layers.0." in name and name.split(".")[-1] in targets:
        print(f"{name:<48} {mod.in_features:>5} -> {mod.out_features:<5}")

n_wrappable = sum(
    1 for _, m in model.named_modules()
    if isinstance(m, torch.nn.Linear)
    and any(t in n for n, mm in model.named_modules() if mm is m for t in targets)
)
print(f"\nhidden size    {model.config.hidden_size}")
print(f"layers         {model.config.num_hidden_layers}")
print(f"attn heads     {model.config.num_attention_heads} q / "
      f"{model.config.num_key_value_heads} kv  (GQA)")

print("\n=== PASS ===")