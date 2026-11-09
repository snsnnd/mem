"""Step3 Train: Hypernet that generates LoRA ΔW from habit_vec

Phase 1: For each habit (code_first / table / prose), train a tiny LoRA on Qwen2.5-0.5B FP16
        on hand-crafted SFT examples that embody the habit style.
Phase 2: Train a hypernetwork MLP habit_vec(5) -> flattened LoRA ΔW to mimic Phase1 weights.

We run on RTX 4060, FP16, frozen base, only LoRA + hypernet trained.

Run:
  uv run python scripts/train_hypernet.py
"""
from __future__ import annotations
import os, random, json
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

torch.manual_seed(0); random.seed(0); np.random.seed(0)

MODEL_ID = os.environ.get("QWEN_ID","Qwen/Qwen2.5-0.5B-Instruct")
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device=="cuda" else torch.float32
print(f"[+] Loading {MODEL_ID} {dtype} on {device}")

tok = AutoTokenizer.from_pretrained(MODEL_ID)
base = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype, device_map=device)
base.config.use_cache=False

# SFT data: each habit's ideal answers - short, style-specific
HABIT_DATA = {
    "code_first": [
        ("写个快速排序并说明一下", "```python\ndef quick_sort(a):\n    return sorted(a)\n```\n简短说明：调用 sorted。"),
        ("写个循环", "```python\nfor i in range(3):\n    print(i)\n```\n简短说明：循环打印。"),
        ("给个例子", "```python\nprint('hi')\n```\n简短说明：打印hi。"),
    ],
    "table": [
        ("介绍一下Mamba", "Mamba是一个高效模型。\n| 项目 | 说明 |\n|---|---|\n| 原理 | 选择性状态空间 |\n| 优势 | 长上下文 |\n| 对比 | 优于Transformer |\n详细展开：..."),
        ("总结今天内容", "总结如下：\n| 项目 | 内容 |\n|---|---|\n| 进展 | 完成 |\n| 待办 | 下一步 |"),
        ("用表格讲一遍这道题", "过程：\n| 步骤 | 结果 |\n|---|---|\n| 1 | 10 |\n| 2 | 20 |"),
    ],
    "prose": [
        ("解释一下这段循环", "这段循环是先选一个中间点作枢轴，把大的拆成小的，分两边依次理顺，整体上是从大到小排。"),
        ("解释一下概念", "这个概念的意思是说，我们先把问题拆开，再把拆开的小问题一个一个理顺，最后汇总起来。"),
        ("讲讲原理", "原理就是先选一个基准，分组推进，每组都比上一组小，慢慢把乱的变有序。"),
    ],
}

def habit_to_vec(habit: str) -> torch.Tensor:
    mapping = {"code_first": [0,0,1,0,0], "table":[0,0,0,1,0], "prose":[0,0,0,0,1],
               "long_table_emoji": [1,1,0,1,0]}
    return torch.tensor(mapping.get(habit, [0,0,0,0,1]), dtype=torch.float32, device=device)

# Phase 1: train per-habit LoRA (quick, 30 steps) to get "ideal" ΔW
lora_cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj","v_proj"], lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
model = get_peft_model(base, lora_cfg)

# Collect ideal weights
ideal_weights = {}

print("\n=== Phase 1: Train per-habit LoRA ===")
for habit, examples in HABIT_DATA.items():
    # reset LoRA to zero-ish
    for n, p in model.named_parameters():
        if "lora_" in n:
            nn.init.zeros_(p) if "lora_B" in n else nn.init.kaiming_uniform_(p, a=5**0.5)
    opt = torch.optim.AdamW([p for n,p in model.named_parameters() if "lora_" in n], lr=2e-4)
    for step in range(30):
        prompt, answer = random.choice(examples)
        msgs = [{"role":"user","content":prompt}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + answer + tok.eos_token
        enc = tok(text, return_tensors="pt").to(device)
        # mask prompt part
        prompt_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        prompt_len = len(tok(prompt_text)["input_ids"])
        labels = enc["input_ids"].clone()
        labels[0, :prompt_len] = -100
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], labels=labels)
        loss = out.loss
        loss.backward()
        opt.step(); opt.zero_grad()
        if step % 10 == 0:
            print(f"  {habit} step {step} loss {loss.item():.3f}")
    # collect ideal flattened weights (last layer only for hypernet regression, small)
    flat = []
    for n,p in model.named_parameters():
        if "layers.23" in n and "lora_A" in n:  # last layer approx
            flat.append(p.detach().cpu().numpy().flatten())
        if "layers.23" in n and "lora_B" in n:
            flat.append(p.detach().cpu().numpy().flatten())
    # If not found (layer count diff), take all
    if not flat:
        for n,p in model.named_parameters():
            if "lora_" in n: flat.append(p.detach().cpu().numpy().flatten())
    ideal_weights[habit]=np.concatenate(flat)
    print(f"  -> {habit}: ideal ΔW size {ideal_weights[habit].size}, norm {np.linalg.norm(ideal_weights[habit]):.3f}")

# Phase2: Train Hypernetwork to predict ideal ΔW from habit_vec
print("\n=== Phase 2: Train Hypernet MLP ===")
# Build dataset
X = torch.stack([habit_to_vec(h) for h in HABIT_DATA.keys()])
Y = torch.tensor(np.stack([ideal_weights[h] for h in HABIT_DATA.keys()]), dtype=torch.float32, device=device)
print(f"X {X.shape} Y {Y.shape}")

class HyperNet(nn.Module):
    def __init__(self, in_dim=5, hidden=64, out_dim=None):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, out_dim))
    def forward(self,x): return self.net(x)

hyper = HyperNet(in_dim=5, hidden=64, out_dim=Y.shape[1]).to(device)
opt = torch.optim.AdamW(hyper.parameters(), lr=1e-3, weight_decay=1e-4)
for step in range(300):
    opt.zero_grad()
    pred = hyper(X)
    loss = nn.MSELoss()(pred, Y)
    loss.backward(); opt.step()
    if step % 50 == 0:
        print(f"  hyper step {step} mse {loss.item():.4f}")

print("\nDone. Hypernet trained from habit -> ideal ΔW. You can now generate weights zero-shot.")
print("Next: plug hyper output into live Qwen to compare style shift (code vs table vs prose).")
