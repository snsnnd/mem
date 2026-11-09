"""Upgrade predictor: guess BOTH dim and value from correction utterance.
Input: correction utterance text (e.g. '删掉 emoji，先给代码')
Output: distribution over (dim,value) options (7 classes)
  0 verbosity_short, 1 verbosity_long, 2 emoji_never, 3 emoji_always,
  4 format_code_first, 5 format_table, 6 format_prose, plus 7 = none/other
Memory update uses confidence-weighted value: only write if conf > thresh.
This is COPE-style proxy: vague/no-dim feedback becomes parsed signal.
"""
from __future__ import annotations
import random, statistics
import numpy as np
import torch
import torch.nn as nn

# Phrase bank per (dim,value)
PHRASES = {
    ("verbosity","short"): ["短一点","别啰嗦","简洁些","说短一点","别讲太长"," concise"],
    ("verbosity","long"): ["详细一点","讲透","多说点","展开讲","详细些","麻烦讲细一点"],
    ("emoji","never"): ["别用 emoji","删掉 emoji","不要表情","去掉 emoji","别加表情"],
    ("emoji","always"): ["加个 emoji","用 emoji","带点表情","配合 emoji"],
    ("format","code_first"): ["先给代码","先上代码","代码先行","先来代码"],
    ("format","table"): ["用表格讲","换成表格","表格形式","用表"],
    ("format","prose"): ["用话讲","文字讲一下","用文字","散文形式"],
}
LABELS = list(PHRASES.keys())
LABEL_IDX = {k:i for i,k in enumerate(LABELS)}

def hash_embed(s:str, dim=64):
    import hashlib
    v=np.zeros(dim, dtype=np.float32)
    for ch in s:
        h=int(hashlib.md5(ch.encode()).hexdigest(),16)
        v[h%dim]+=1
    n=np.linalg.norm(v)
    return v/n if n>0 else v

class ValueParser(nn.Module):
    def __init__(self, in_dim=64, hidden=64, out_dim=7):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(in_dim,hidden), nn.ReLU(), nn.Linear(hidden,hidden), nn.ReLU(), nn.Linear(hidden,out_dim))
    def forward(self,x): return self.net(x)

def train_parser():
    device="cuda" if torch.cuda.is_available() else "cpu"
    print("=== Training dim+value parser from vague utterance text ===")
    # synthetic dataset: random phrase -> its (dim,value)
    X=[]; Y=[]
    rng=random.Random(0)
    for (dim,val), phrases in PHRASES.items():
        for p in phrases:
            for repeat in range(80):
                # add some noise by appending random filler?
                txt = p if repeat%3!=0 else p + "，谢谢"
                X.append(hash_embed(txt)); Y.append(LABEL_IDX[(dim,val)])
    X=torch.stack([torch.tensor(x) for x in X]).to(device)
    Y=torch.tensor(Y, dtype=torch.long, device=device)
    model=ValueParser().to(device)
    opt=torch.optim.AdamW(model.parameters(), lr=1e-3)
    for epoch in range(60):
        model.train()
        perm=torch.randperm(len(X), device=device)
        for i in range(0,len(X),64):
            idx=perm[i:i+64]
            opt.zero_grad()
            loss=nn.CrossEntropyLoss()(model(X[idx]), Y[idx])
            loss.backward(); opt.step()
        if epoch%20==0:
            model.eval()
            with torch.no_grad():
                acc=(model(X).argmax(1)==Y).float().mean().item()
            print(f" epoch {epoch} acc {acc:.3f} (chance 0.14)")
    print(f" Final parser acc {(model(X).argmax(1)==Y).float().mean().item():.3f}")
    return model, device, LABELS

def run_vague_parser(model, device, labels, conf_thresh=0.6):
    """Now vague feedback carries an utterance; we parse (dim,value) from it.
    Memory writes only if parser conf > thresh.
    Compare if parser helps when phrase overlaps train vs new phrasing."""
    print(f"\n=== Vague feedback + parser (conf_thresh={conf_thresh}) ===")
    # For this inline test we reuse previous B1 vague setup but now user "vague" still gives a phrase
    # so parser can actually get value. This is the COPE-style proxy reward step.
    from .users import generate_users
    from .run_b1 import predict_from_pack, true_pref_for
    rng=random.Random(123)
    users=generate_users(n=30, seed=123)
    class LatestParsed:
        def __init__(self): self.store={}
        def get_pack(self): return dict(self.store) if self.store else None
    errs=[]
    for u in users:
        mem=LatestParsed()
        our_true = u.habit_dict() if hasattr(u,"habit_dict") else u
        tot=0
        for turn in range(200):
            task=rng.choice(["code","summary","explain","create"])
            true_p=true_pref_for(our_true, task)
            pack=mem.get_pack()
            pred=predict_from_pack(pack, rng)
            mism=[d for d in ["verbosity","emoji","format"] if pred[d]!=true_p[d]]
            if mism: tot+=1
            # User produces a vague utterance that still hints true value (COPE scenario)
            # We simulate user saying a correction phrase for a mismatched dim (with prob 0.6, realistic sparse)
            if mism and rng.random()<0.6:
                d_choice=rng.choice(mism)
                phrases=PHRASES[(d_choice, true_p[d_choice])]
                utter=rng.choice(phrases)
                # Parser predicts dim+value from utter
                x=torch.tensor(hash_embed(utter), dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    logits=model(x)
                    probs=torch.softmax(logits, dim=-1)[0].cpu().numpy()
                idx=int(np.argmax(probs)); conf=float(probs[idx])
                if conf>=conf_thresh:
                    d_pred, v_pred = labels[idx]
                    mem.store[d_pred]=v_pred
            # else no feedback parsed
        errs.append(tot/200)
    print(f" Vague parsed err={statistics.mean(errs):.3f} vs 0.911 hard vague, 0.886 dim-only guess")
    return statistics.mean(errs)

if __name__=="__main__":
    model, device, labels=train_parser()
    for thr in [0.5,0.6,0.75]:
        run_vague_parser(model, device, labels, conf_thresh=thr)
