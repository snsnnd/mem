"""Hebbian + Slot Routing: multiple Hebbian regions, gating by context.

Design (hippocampal style):
- Bank of slots: each slot is a Hebbian associative memory W_i (8x5)
- Router: distance of current input x to slot prototype c_i measures familiarity.
  softmax(-dist/T) gives activation. Only high-activation slots are updated
  (selective plasticity) -> less interference than single global W.
- Growth: if surprise (correction) fires AND min_dist > threshold, create new slot
  with one-shot Hebbian association (neurogenesis, pattern separation).
- Decay: all slots decay per turn, prune if importance < thresh.

Difference vs single Hebbian (G):
 single:   one W, all contexts share weights -> interference
 routed:   N Ws, contexts separated -> each region can be fast without overwriting others
"""
from __future__ import annotations
import numpy as np

class Slot:
    __slots__ = ("prototype","W","importance","last_used","count")
    def __init__(self, prototype, W):
        self.prototype = prototype.copy()
        self.W = W.copy()
        self.importance = 1.0
        self.last_used = 0
        self.count = 1

class HebbianSlotRouted:
    name = "H_hebbian_slot_routing"
    def __init__(self, in_dim=11, eta=0.6, decay=0.99, gamma=0.97,
                 max_slots=5, dist_thresh=1.2, temp=0.5, seed=0):
        self.in_dim=in_dim; self.out_dim=5
        self.eta=eta; self.decay=decay; self.gamma=gamma
        self.max_slots=max_slots; self.dist_thresh=dist_thresh; self.temp=temp
        self.slots: list[Slot]=[]
        self.rng=np.random.default_rng(seed)
        self.turn=0
        self.corr_ema={"verbosity":0.0,"emoji":0.0,"format":0.0}
        self.last_correct=0.0
        self.growth=0; self.prune=0

    def _task_type_of(self, task: str | None) -> str:
        if not task: return "explain"
        if "代码" in task: return "code"
        if "总结" in task: return "summary"
        return "explain"

    def _feat_raw(self, turn:int, task: str | None = None):
        x=np.zeros(self.in_dim)
        x[0]=1.0
        x[1]=self.corr_ema["verbosity"]; x[2]=self.corr_ema["emoji"]; x[3]=self.corr_ema["format"]
        x[4]=turn/100.0; x[5]=self.last_correct
        x[6]=1.0 if turn%2==0 else 0.0
        x[7]=1.0
        tt = self._task_type_of(task)
        if tt == "code": x[8]=1.0
        elif tt == "summary": x[9]=1.0
        else: x[10]=1.0
        return x

    def _sparsify(self, x):
        # same pattern separation as single hebbian: topk keep
        if self.in_dim>4:
            idx=np.argsort(np.abs(x))[::-1]
            keep=set(idx[:3].tolist()); keep.add(0)
            y=x.copy()
            for i in range(self.in_dim):
                if i not in keep: y[i]=0.0
            return y
        return x

    def _weights(self, x):
        if not self.slots: return np.array([])
        dists=np.array([np.linalg.norm(x - s.prototype) for s in self.slots])
        w=np.exp(-dists/self.temp)
        w=w/np.sum(w)
        return w, dists

    def _forward(self, x):
        if not self.slots:
            return 0.5,0.5,np.array([1/3,1/3,1/3])
        w,_=self._weights(x)
        # weighted sum of logits
        logits=np.zeros(self.out_dim)
        for wi, s in zip(w, self.slots):
            logits += wi * (x @ s.W)
        p_v=1/(1+np.exp(-logits[0])); p_e=1/(1+np.exp(-logits[1]))
        e=np.exp(logits[2:5]-np.max(logits[2:5])); p_f=e/np.sum(e)
        return p_v,p_e,p_f

    def get_pack(self, task: str | None = None):
        x=self._sparsify(self._feat_raw(self.turn, task))
        p_v,p_e,p_f=self._forward(x)
        return {"verbosity":"long" if p_v>0.5 else "short",
                "emoji":"always" if p_e>0.5 else "never",
                "format":["code_first","table","prose"][int(np.argmax(p_f))]}

    def observe(self, corrections, true_value, turn, rng, task: str | None = None):
        self.turn=turn
        x_raw=self._feat_raw(turn, task); x=self._sparsify(x_raw)
        for d in self.corr_ema:
            self.corr_ema[d]=0.9*self.corr_ema[d]+0.1*(1.0 if d in corrections else 0.0)
        self.last_correct=1.0 if not corrections else 0.0

        # decay all slots
        for s in self.slots:
            s.W*=self.decay; s.importance*=self.gamma

        # Determine activations
        if self.slots:
            w,dists=self._weights(x)
            min_dist=float(np.min(dists))
        else:
            w=np.array([]); min_dist=999.0

        # Growth: surprise + novelty + capacity
        need_new = bool(corrections) and (min_dist > self.dist_thresh)
        if need_new and len(self.slots)<self.max_slots:
            W0=self.rng.normal(0,0.01,(self.in_dim,self.out_dim))
            ns=Slot(prototype=x_raw, W=W0); ns.last_used=turn
            self.slots.append(ns); self.growth+=1
            # recompute weights after growth
            w,dists=self._weights(x)

        # Update only activated slots (top 2 most weighted)
        if self.slots and len(w)>0:
            order=np.argsort(w)[::-1]
            active=order[:2] if len(order)>1 else order[:1]
            p_v,p_e,p_f=self._forward(x)
            for idx in active:
                s=self.slots[idx]
                # prototype moving (cortical consolidation slowly)
                s.prototype=0.9*s.prototype+0.1*x_raw
                s.importance=min(2.0,s.importance+0.05); s.count+=1; s.last_used=turn
                # Hebbian delta only on corrected dims, modulated by weight
                wi=float(w[idx])
                if "verbosity" in corrections:
                    y=1.0 if true_value["verbosity"]=="long" else 0.0
                    s.W[:,0]+=self.eta*wi*x*(y-p_v)
                if "emoji" in corrections:
                    y=1.0 if true_value["emoji"]=="always" else 0.0
                    s.W[:,1]+=self.eta*wi*x*(y-p_e)
                if "format" in corrections:
                    yi={"code_first":0,"table":1,"prose":2}[true_value["format"]]
                    tgt=np.zeros(3); tgt[yi]=1.0
                    s.W[:,2:5]+=self.eta*wi*np.outer(x, tgt-p_f)
                s.W=np.clip(s.W,-5,5)
        # prune useless slots
        self.slots=[s for s in self.slots if not (s.importance<0.05 and (turn-s.last_used)>20)]
        # count prune roughly
        # (simplified: prune event not tracked precisely in this MVP)

    def stats(self):
        return {"slots":len(self.slots),"growth":self.growth}
