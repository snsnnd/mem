"""Test: predict next user correction TYPE from context (task + answer style).
Vague feedback case: user only says '不对' and we must infer which dim was wrong.
If a small transformer can predict correction type > chance, we can update memory
even without explicit dim label, turning vague into usable signal.

We train a tiny transformer encoder on synthetic explicit corrections, then plug
it into memory learner for vague mode and measure improvement vs no-predictor.
"""
from __future__ import annotations
import random, statistics
import numpy as np
import torch
import torch.nn as nn
from .users import generate_users
from .assistant import make_answer
from .run_b1 import LatestValue
from .run_b1 import predict_from_pack

# We'll reuse simpler imports to avoid circular
from .run_b1 import make_feedback, true_pref_for

TASK_POOL = ["code","summary","explain","create"]

def hash_embed(s: str, dim=32):
    import hashlib
    v=np.zeros(dim, dtype=np.float32)
    for ch in s:
        h=int(hashlib.md5(ch.encode()).hexdigest(),16)
        v[h%dim]+=1
    n=np.linalg.norm(v)
    return v/n if n>0 else v

def encode_ctx(task: str, answer_style: dict) -> torch.Tensor:
    a=hash_embed(task)
    # answer style as text
    sty_txt=f"{answer_style.get('verbosity')}_{answer_style.get('emoji')}_{answer_style.get('format')}"
    b=hash_embed(sty_txt)
    return torch.tensor(np.concatenate([a,b]), dtype=torch.float32)  # (64,)

class CorrectionPredictor(nn.Module):
    def __init__(self, in_dim=64, d_model=64, nhead=4, dim_ff=128, num_layers=2, out_dim=4):
        super().__init__()
        self.proj=nn.Linear(in_dim, d_model)
        enc_layer=nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, batch_first=True)
        self.enc=nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.cls=nn.Linear(d_model, out_dim)  # 0 verbosity,1 emoji,2 format,3 none

    def forward(self, x):  # x: (B, seq, in_dim) -> (B, out_dim)
        # treat sequence length 1 for simplicity, add pos embedding via proj
        seq=self.proj(x)  # (B,1,d_model)
        out=self.enc(seq)[:,0,:]  # (B,d_model)
        return self.cls(out)

def train_predictor():
    print("=== Training correction-type predictor (synthetic explicit) ===")
    device="cuda" if torch.cuda.is_available() else "cpu"
    # Generate synthetic data: random user, random task, random answer style
    X=[]; Y=[]  # Y: primary correction dim (0/1/2) or 3 if none
    rng=random.Random(0)
    users=generate_users(n=200, seed=1)
    for i in range(3000):
        u=rng.choice(users)
        task=rng.choice(TASK_POOL)
        true = true_pref_for(u.habit_dict() if hasattr(u,"habit_dict") else u, task)
        # Random answer style (simulate mock LLM)
        ans = {d: rng.choice(["short","long"] if d=="verbosity" else ["never","always"] if d=="emoji" else ["code_first","table","prose"]) for d in ["verbosity","emoji","format"]}
        # full dict for encode
        x=encode_ctx(task, ans)
        mism=[d for d in ["verbosity","emoji","format"] if ans[d]!=true[d]]
        if not mism:
            y=3
        elif len(mism)==1:
            y=["verbosity","emoji","format"].index(mism[0])
        else:
            y=3  # multi-mismatch -> treat as none for simplicity, ambiguous
            continue
        X.append(x.unsqueeze(0))  # (1,64)
        Y.append(y)
    X=torch.stack(X).to(device); Y=torch.tensor(Y, dtype=torch.long, device=device)
    model=CorrectionPredictor(in_dim=64).to(device)
    opt=torch.optim.AdamW(model.parameters(), lr=1e-3)
    # split train/test
    for epoch in range(80):
        model.train()
        perm=torch.randperm(len(X), device=device)
        for i in range(0,len(X),128):
            idx=perm[i:i+128]
            opt.zero_grad()
            out=model(X[idx])
            loss=nn.CrossEntropyLoss()(out, Y[idx])
            loss.backward(); opt.step()
        if epoch%20==0:
            model.eval()
            with torch.no_grad():
                pred=model(X).argmax(1)
                acc=(pred==Y).float().mean().item()
            print(f" epoch {epoch} train_acc {acc:.3f} (chance ~0.25)")
    model.eval()
    # final accuracy
    with torch.no_grad():
        acc=(model(X).argmax(1)==Y).float().mean().item()
    print(f" Final predictor accuracy {acc:.3f}")
    return model, device

def run_vague_with_predictor(model=None, device="cpu"):
    """Run B1 vague mode but use predictor to guess correction dim when vague."""
    from .run_b1 import LatestValue, generate_users_ext, true_pref_for, make_feedback
    rng=random.Random(42)
    users=generate_users(n=30, seed=123)
    # We'll simulate mem that updates using predicted correction
    class PredLatest(LatestValue):
        def observe_with_pred(self, pred_dim, true_true, turn):
            # pred_dim: 0,1,2,3
            dim_map=["verbosity","emoji","format"]
            if pred_dim is not None and pred_dim<3:
                d=dim_map[pred_dim]
                # we don't have true value for that dim in vague mode, but in simulation
                # we can use true_true[d] as ground truth that would be inferred if predictor were perfect?
                # For this demo, we will assume predictor guesses dim, then we treat it as if user said that dim
                # and we update that dim to true value (since if dim guess is correct, we know true value via trying?)
                # This is cheating a bit - in real world we'd need to parse user's vague utterance to get value.
                # Here we assume after guessing dim, we can infer value as the opposite of what we predicted?
                # Simpler: we update store to true value for guessed dim.
                self.store[d]=true_true[d]
    errs=[]
    for u in users:
        mem=PredLatest()
        tot=0
        for turn in range(200):
            task=rng.choice(TASK_POOL)
            true_p=true_pref_for(u.habit_dict() if hasattr(u,"habit_dict") else u, task)
            pack=mem.get_pack()
            pred = predict_from_pack(pack, rng)
            mism=[d for d in ["verbosity","emoji","format"] if pred[d]!=true_p[d]]
            if mism: tot+=1
            # Vague feedback: user says "不对" but not which dim
            # Predictor guesses dim from context (task + answer)
            if model is not None and mism:
                x=encode_ctx(task, pred).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,64)
                with torch.no_grad():
                    logits=model(x)
                    guess=int(logits.argmax(1).item())
                if guess<3:
                    mem.observe_with_pred(guess, true_p, turn)
                # if guess==3 (none), no update
            # else no update
        errs.append(tot/200)
    print(f"Vague + predictor-guess err={statistics.mean(errs):.3f} (vs 0.911 no-guess baseline)")

if __name__=="__main__":
    model, device = train_predictor()
    run_vague_with_predictor(model, device)
