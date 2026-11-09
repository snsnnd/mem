"""Step3: Habit vector -> Hypernetwork -> LoRA ΔW (instead of text pack).

We test with a mock base LLM: logits = 0 (generic) + ΔW_from_hypernet.
Hypernet maps habit_vec (slot pack) to 5 bias logits (verb, emoji, fmt3).
Training is online on corrections (true habit revealed).

This proves: same habit, but injected as weights (0 tokens) not prompts.
"""
from __future__ import annotations
import numpy as np

class HyperLoRA:
    def __init__(self, in_dim=5, hidden=32, out_dim=5, lr=0.08, seed=0, weight_decay=0.01):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, np.sqrt(1.0/in_dim), (in_dim, hidden)); self.b1=np.zeros(hidden)
        self.W2 = rng.normal(0, np.sqrt(1.0/hidden), (hidden, out_dim)); self.b2=np.zeros(out_dim)
        self.lr=lr; self.wd=weight_decay; self.t=0
        self.mW1=np.zeros_like(self.W1); self.vW1=np.zeros_like(self.W1)
        self.mb1=np.zeros_like(self.b1); self.vb1=np.zeros_like(self.b1)
        self.mW2=np.zeros_like(self.W2); self.vW2=np.zeros_like(self.W2)
        self.mb2=np.zeros_like(self.b2); self.vb2=np.zeros_like(self.b2)
        self.beta1, self.beta2, self.eps=0.9,0.999,1e-8

    def forward(self, x):
        h=np.maximum(0, x@self.W1+self.b1)
        dW = h@self.W2+self.b2  # ΔW bias (LoRA merged effect)
        return dW, h

    def predict_style(self, x):
        dW,_=self.forward(x)
        p_v=1/(1+np.exp(-dW[0])); p_e=1/(1+np.exp(-dW[1]))
        e=np.exp(dW[2:5]-np.max(dW[2:5])); p_f=e/np.sum(e)
        return {"verbosity":"long" if p_v>0.5 else "short",
                "emoji":"always" if p_e>0.5 else "never",
                "format":["code_first","table","prose"][int(np.argmax(p_f))]}, (p_v,p_e,p_f)

    def train_step(self, x, true_habit: dict, corrections: list[str]):
        dW, h = self.forward(x)
        # build dlogits like before, only for corrected dims
        dlogits=np.zeros(5); 
        if "verbosity" in corrections:
            y=1.0 if true_habit["verbosity"]=="long" else 0.0
            p=1/(1+np.exp(-dW[0])); dlogits[0]=p-y
        if "emoji" in corrections:
            y=1.0 if true_habit["emoji"]=="always" else 0.0
            p=1/(1+np.exp(-dW[1])); dlogits[1]=p-y
        if "format" in corrections:
            yi={"code_first":0,"table":1,"prose":2}[true_habit["format"]]
            e=np.exp(dW[2:5]-np.max(dW[2:5])); pf=e/np.sum(e)
            dlogits[2:5]=pf.copy(); dlogits[2+yi]-=1.0
        # backprop
        dW2=np.outer(h, dlogits)+self.wd*self.W2; db2=dlogits
        dh=(dlogits@self.W2.T)*( (x@self.W1+self.b1)>0 )
        dW1=np.outer(x, dh)+self.wd*self.W1; db1=dh
        self.t+=1
        for param,grad,m,v in [(self.W1,dW1,self.mW1,self.vW1),(self.b1,db1,self.mb1,self.vb1),
                               (self.W2,dW2,self.mW2,self.vW2),(self.b2,db2,self.mb2,self.vb2)]:
            m[:]=self.beta1*m+(1-self.beta1)*grad; v[:]=self.beta2*v+(1-self.beta2)*(grad**2)
            mhat=m/(1-self.beta1**self.t); vhat=v/(1-self.beta2**self.t)
            param[:]-=self.lr*mhat/(np.sqrt(vhat)+self.eps)

def habit_to_vec(pack: dict | None) -> np.ndarray:
    if pack is None:
        return np.zeros(5)
    v=np.zeros(5)
    v[0]=1.0 if pack.get("verbosity")=="long" else 0.0
    v[1]=1.0 if pack.get("emoji")=="always" else 0.0
    fmt=pack.get("format")
    if fmt=="code_first": v[2]=1.0
    elif fmt=="table": v[3]=1.0
    elif fmt=="prose": v[4]=1.0
    return v
