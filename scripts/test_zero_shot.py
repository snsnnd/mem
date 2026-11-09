"""Zero-shot test: unseen habit combo -> Hypernet generates ΔW -> inject into Qwen

We re-train quick LoRA per habit (30 steps each) + Hypernet regression, then test
on unseen habit: long + always emoji + code_first (code_first+emoji+long combo never trained as single)
"""
from __future__ import annotations
import os, random
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

torch.manual_seed(0); random.seed(0); np.random.seed(0)
MODEL_ID="Qwen/Qwen2.5-0.5B-Instruct"
device="cuda" if torch.cuda.is_available() else "cpu"
dtype=torch.float16 if device=="cuda" else torch.float32
print(f"[+] {MODEL_ID} {dtype} on {device}")

tok=AutoTokenizer.from_pretrained(MODEL_ID)
base=AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype, device_map=device)
base.config.use_cache=False

HABIT_DATA={
    "code_first": [("写个快速排序并说明一下","```python\ndef quick_sort(a):\n    return sorted(a)\n```\n简短说明：调用 sorted。"),
                   ("写个循环","```python\nfor i in range(3):\n    print(i)\n```\n简短说明：循环打印。"),
                   ("给个例子","```python\nprint('hi')\n```\n简短说明：打印hi。")],
    "table": [("介绍一下Mamba","Mamba是一个高效模型。\n| 项目 | 说明 |\n|---|---|\n| 原理 | 选择性状态空间 |\n| 优势 | 长上下文 |"),
              ("总结今天内容","总结如下：\n| 项目 | 内容 |\n|---|---|\n| 进展 | 完成 |\n| 待办 | 下一步 |"),
              ("用表格讲一遍","过程：\n| 步骤 | 结果 |\n|---|---|\n| 1 | 10 |\n| 2 | 20 |")],
    "prose": [("解释一下这段循环","这段循环是先选一个中间点作枢轴，把大的拆成小的，分两边依次理顺。"),
              ("解释一下概念","这个概念的意思是说，我们先把问题拆开，再把拆开的小问题一个一个理顺。"),
              ("讲讲原理","原理就是先选一个基准，分组推进，每组都比上一组小。")],
}

def habit_to_vec(habit:str):
    m={"code_first":[0,0,1,0,0],"table":[0,0,0,1,0],"prose":[0,0,0,0,1],
       "unseen_code_long_emoji":[1,1,1,0,0]}
    return torch.tensor(m.get(habit,[0,0,0,0,1]), dtype=torch.float32, device=device)

lora_cfg=LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj","v_proj"], lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
model=get_peft_model(base, lora_cfg)
# record param order for last layer
target_names=[]
target_shapes=[]
for n,p in model.named_parameters():
    if "layers.23" in n and "lora_" in n:
        target_names.append(n); target_shapes.append(p.shape)
print("target lora params:", target_names)
# quick train per habit
ideal_weights={}
for habit, examples in HABIT_DATA.items():
    for n,p in model.named_parameters():
        if "lora_" in n:
            nn.init.zeros_(p) if "lora_B" in n else nn.init.kaiming_uniform_(p, a=5**0.5)
    opt=torch.optim.AdamW([p for n,p in model.named_parameters() if "lora_" in n], lr=2e-4)
    for step in range(30):
        prompt, answer = random.choice(examples)
        msgs=[{"role":"user","content":prompt}]
        prompt_text=tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        text=prompt_text+answer+tok.eos_token
        enc=tok(text, return_tensors="pt").to(device)
        prompt_len=len(tok(prompt_text)["input_ids"])
        labels=enc["input_ids"].clone(); labels[0,:prompt_len]=-100
        out=model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], labels=labels)
        out.loss.backward(); opt.step(); opt.zero_grad()
    flat = np.concatenate([p.detach().cpu().numpy().flatten() for n,p in model.named_parameters() if n in target_names])
    ideal_weights[habit]=flat
    print(f"  trained {habit} loss {out.loss.item():.3f} norm {np.linalg.norm(flat):.2f}")

# Hypernet
X=torch.stack([habit_to_vec(h) for h in HABIT_DATA.keys()])
Y=torch.tensor(np.stack([ideal_weights[h] for h in HABIT_DATA.keys()]), dtype=torch.float32, device=device)
class HyperNet(nn.Module):
    def __init__(self, in_dim=5, hidden=64, out_dim=None):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(in_dim,hidden), nn.ReLU(), nn.Linear(hidden,hidden), nn.ReLU(), nn.Linear(hidden,out_dim))
    def forward(self,x): return self.net(x)
hyper=HyperNet(in_dim=5, hidden=64, out_dim=Y.shape[1]).to(device)
opt=torch.optim.AdamW(hyper.parameters(), lr=1e-3, weight_decay=1e-4)
for step in range(300):
    opt.zero_grad(); loss=nn.MSELoss()(hyper(X), Y); loss.backward(); opt.step()
print(f"hyper trained mse {loss.item():.6f}")

# Zero-shot: unseen habit long+emoji+code_first
unseen_vec=habit_to_vec("unseen_code_long_emoji")
print("\n=== Zero-shot unseen: long + always emoji + code_first ===")
print("This combo never trained as single habit. Hypernet will interpolate.")
with torch.no_grad():
    pred_flat=hyper(unseen_vec[None,:])[0].cpu().numpy()

def inject_flat(flat_np):
    offset=0
    with torch.no_grad():
        for (n, shape) in zip(target_names, target_shapes):
            # find param
            for nn_name, param in model.named_parameters():
                if nn_name==n:
                    size=int(np.prod(shape))
                    chunk=flat_np[offset:offset+size]
                    param.data.copy_(torch.tensor(chunk, dtype=param.dtype, device=device).view(shape))
                    offset+=size
                    break

def generate(prompt: str, pack_style=None):
    msgs=[{"role":"user","content":prompt}]
    text=tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs=tok([text], return_tensors="pt").to(device)
    with torch.no_grad():
        out=model.generate(**inputs, max_new_tokens=140, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

def style_of(text:str):
    return {"code":"```" in text, "table":"|" in text, "emoji":any(ord(c)>0xFFFF or c in "✨😊" for c in text), "len":len(text)}

prompt="写个快速排序，并详细说明，记得用emoji"
# Base (zero LoRA)
for n,p in model.named_parameters():
    if "lora_B" in n: nn.init.zeros_(p)
base_txt=generate(prompt)
print(f"[Base:zeroLoRA] {base_txt[:220]}")
print(f" style {style_of(base_txt)}")

# PromptPack as text (inject habit as text, not weights)
# manually add instruction
msgs=[{"role":"user","content":prompt+"（用户习惯：详细，长，emoji，代码先行）"}]
text=tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
inputs=tok([text], return_tensors="pt").to(device)
with torch.no_grad():
    out=model.generate(**inputs, max_new_tokens=140, do_sample=False, pad_token_id=tok.eos_token_id)
pack_txt=tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(f"\n[PromptPack] {pack_txt[:220]}")
print(f" style {style_of(pack_txt)}")

# Hypernet ΔW injection
inject_flat(pred_flat)
hyper_txt=generate(prompt)
print(f"\n[HypernetΔW unseen combo] {hyper_txt[:260]}")
print(f" style {style_of(hyper_txt)}")

print("\nCheck: Hypernet should have code+table/emoji? Expect code block + maybe emoji + longer verbose.")
