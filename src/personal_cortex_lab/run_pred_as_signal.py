"""Integrate Predictor as Feedback Signal: explicit sparse + predicted vague

We replace the naive feedback encoder with:
  FeedbackEncoder = explicit sparse signal + transformer-predicted correction distribution

When user corrects, we now have two sources:
  - explicit dict (if user said it) - high precision
  - predicted logits (if vague or sparse) - lower precision but available when explicit missing

Memory update uses predicted dim if explicit missing AND confidence > thresh.

We test: does adding predicted signal improve vague and sparse cases?
"""
from __future__ import annotations
import random, statistics, torch
import torch.nn as nn
import numpy as np

from .users import generate_users
from .run_b1 import true_pref_for, predict_from_pack, make_feedback
from .run_vague_predictor import CorrectionPredictor, encode_ctx

# Minimal LatestValue that accepts predicted signal
class LatestWithPred:
    name="Latest+Pred"
    def __init__(self):
        self.store={}
        self.conf_thresh=0.6
    def get_pack(self):
        return dict(self.store) if self.store else None
    def observe(self, explicit: dict | None, predicted: dict | None, turn: int):
        # explicit first (ground truth)
        if explicit:
            self.store.update(explicit)
        elif predicted:
            # predicted is {dim: (value, conf)} where value is guessed true value? 
            # In real world we would parse utterance to get value; here we simulate:
            # predicted contains dim guesses with true value revealed if guess right?
            # For this experiment we assume if predicted dim is correct, we get its true value
            # (this simulates parsing the vague utterance to find dimension).
            # If guess wrong, we would write wrong value -> false update.
            for dim, (val, conf) in predicted.items():
                if conf >= self.conf_thresh:
                    self.store[dim]=val

def train_and_test():
    device="cuda" if torch.cuda.is_available() else "cpu"
    # 1) Train predictor quickly on synthetic explicit (same as before)
    from .run_vague_predictor import train_predictor
    model, device = train_predictor()
    # 2) Benchmark vague with predictor-as-signal vs no-predictor
    users=generate_users(n=30, seed=123)
    for mode in ["vague", "sparse_explicit"]:
        for use_pred in [False, True]:
            rng=random.Random(42)
            errs=[]
            for u in users:
                mem=LatestWithPred()
                tot=0
                for turn in range(300):
                    task=rng.choice(["code","summary","explain","create"])
                    true_p=true_pref_for(u.habit_dict() if hasattr(u,"habit_dict") else u, task)
                    pack=mem.get_pack()
                    pred=predict_from_pack(pack, rng)
                    mism=[d for d in ["verbosity","emoji","format"] if pred.get(d)!=true_p.get(d)]
                    if mism: tot+=1
                    # Generate feedback
                    explicit=None; predicted=None
                    # Determine if user corrects at all
                    if mism and rng.random()<0.9:  # user notices 90% when mistaken (vague or not)
                        if mode=="vague":
                            explicit=None  # user says only "不对"
                        else:
                            # sparse explicit: user says dim only when present
                            explicit={d:true_p[d] for d in mism}
                        # Predictor guesses dim from context (task+pred)
                        if use_pred:
                            x=encode_ctx(task, pred).unsqueeze(0).unsqueeze(0).to(device)
                            with torch.no_grad():
                                logits=model(x)  # (1,4)
                                probs=torch.softmax(logits, dim=-1).cpu().numpy()[0]
                            # Translate probs to per-dim targeted predictions
                            # Highest prob dim -> we guess that dim is wrong.
                            # For simulation, if guess matches a true mism dim, we get its true value, else we guess wrong dim and flip.
                            dim_map=["verbosity","emoji","format"]
                            guess_idx=int(np.argmax(probs))
                            conf=float(probs[guess_idx])
                            if guess_idx<3:
                                d_guess=dim_map[guess_idx]
                                # If guess correct, value = true; else we would parse wrong dim and get wrong value snapshot
                                # In this simulation, if guess dim is a true mismatched dim, we get correct value; else we get flipped value (noise)
                                if d_guess in mism:
                                    val=true_p[d_guess]
                                else:
                                    # guessed a dim that wasn't mistaken -> we'd flip a correct dim to wrong -> bad update
                                    # To simulate, we set val to the opposite of what's in pack (or true, whichever)
                                    opts={"verbosity":["short","long"],"emoji":["never","always"],"format":["code_first","table","prose"]}[d_guess]
                                    val=rng.choice([o for o in opts if o!=true_p[d_guess]])
                                predicted={d_guess:(val, conf)}
                    mem.observe(explicit, predicted, turn)
                errs.append(tot/300)
            print(f"mode={mode} use_pred={use_pred} err={statistics.mean(errs):.3f}")

if __name__=="__main__":
    train_and_test()
