"""LLM-as-encoder parser: reuse Qwen2.5-0.5B hidden states as frozen encoder.

We compare:
 Baseline: hash bag-of-words encoder (previous run_value_parser)
 Proposed: Qwen last-layer mean-pooled embedding -> MLP head (frozen LLM)

Test generalization to unseen paraphrases. If LLM encoder works, unseen phrasing will still map.

Splits:
 train_phrases: phrases from PHRASES bank
 test_paraphrases: unseen paraphrases per (dim,value) - never seen in training
"""
from __future__ import annotations
import os, random
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

PHRASES_TRAIN = {
    ("verbosity","short"): ["短一点","别啰嗦","简洁些"],
    ("verbosity","long"): ["详细一点","讲透","多说点"],
    ("emoji","never"): ["别用 emoji","删掉 emoji","不要表情"],
    ("emoji","always"): ["加个 emoji","用 emoji","带点表情"],
    ("format","code_first"): ["先给代码","先上代码","代码先行"],
    ("format","table"): ["用表格讲","换成表格","表格形式"],
    ("format","prose"): ["用话讲","文字讲一下","用文字"],
}
# Unseen paraphrases (never in training)
PHRASES_TEST = {
    ("verbosity","short"): ["说短一点","压缩一下","别长篇大论"],
    ("verbosity","long"): ["展开讲","讲详细些","多讲一些"],
    ("emoji","never"): ["去掉表情符号","不要加表情符号","别出现表情符号"],
    ("emoji","always"): ["配上表情符号","加上 emoji","用表情点缀"],
    ("format","code_first"): ["先来程序","程序先行","直接贴代码"],
    ("format","table"): ["用表讲","列表格出来","表格呈现"],
    ("format","prose"): ["用散文讲","口头描述一下","文字描述"],
}
LABELS = list(PHRASES_TRAIN.keys())
IDX = {k:i for i,k in enumerate(LABELS)}

MODEL_ID="Qwen/Qwen2.5-0.5B-Instruct"
device="cuda" if torch.cuda.is_available() else "cpu"
dtype=torch.float16 if device=="cuda" else torch.float32
print(f"[+] {MODEL_ID} on {device}")

tok=AutoTokenizer.from_pretrained(MODEL_ID)
base=AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype, device_map=device)
base.eval()
# freeze base
for p in base.parameters(): p.requires_grad=False

def llm_embed(text: str) -> torch.Tensor:
    """Mean-pooled last hidden state as embedding."""
    enc=tok(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out=base.model(**enc)  # base model body
        hidden=out.last_hidden_state  # (1, seq, hidden)
    # mean pool excluding padding (simple mean)
    emb=hidden.mean(dim=1)[0]  # (hidden,)
    return emb

class Head(nn.Module):
    def __init__(self, in_dim, hidden=128, out_dim=7):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(in_dim,hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))
    def forward(self,x): return self.net(x)

# Build datasets
def build_dataset(phrases_dict):
    X=[]; Y=[]
    for (dim,val), ps in phrases_dict.items():
        for p in ps:
            # repeat for stability
            for _ in range(20):
                emb=llm_embed(p).float()  # to float32 for MLP
                X.append(emb); Y.append(IDX[(dim,val)])
    return torch.stack(X), torch.tensor(Y, device=device)

# Quick check embedding dim
tmp=llm_embed("测试")
print("LLM embed dim:", tmp.shape)

Xtr, Ytr = build_dataset(PHRASES_TRAIN)
Xte, Yte = build_dataset(PHRASES_TEST)

head=Head(in_dim=Xtr.shape[1]).to(device)
opt=torch.optim.AdamW(head.parameters(), lr=1e-3)
for epoch in range(80):
    head.train()
    perm=torch.randperm(len(Xtr), device=device)
    for i in range(0,len(Xtr),64):
        idx=perm[i:i+64]
        opt.zero_grad()
        loss=nn.CrossEntropyLoss()(head(Xtr[idx]), Ytr[idx])
        loss.backward(); opt.step()
    if epoch%20==0:
        head.eval()
        with torch.no_grad():
            tr_acc=(head(Xtr).argmax(1)==Ytr).float().mean().item()
            te_acc=(head(Xte).argmax(1)==Yte).float().mean().item()
        print(f" epoch {epoch} train_acc {tr_acc:.3f} test_paraphrase_acc {te_acc:.3f}")

# Final
head.eval()
with torch.no_grad():
    tr_acc=(head(Xtr).argmax(1)==Ytr).float().mean().item()
    te_acc=(head(Xte).argmax(1)==Yte).float().mean().item()
print(f"\nFinal: train phrase acc {tr_acc:.3f} | unseen paraphrase acc {te_acc:.3f} (chance 0.14)")

print("\nInterpretation: If LLM encoder generalizes to synonyms ('不要表情符号' vs '别用emoji'),")
print("unseen accuracy stays high; hash encoder earlier was 100% on seen but would drop on unseen.")
