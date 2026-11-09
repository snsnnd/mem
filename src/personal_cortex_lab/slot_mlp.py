"""Slot-partitioned MLP: each slot is a region with its own tiny MLP.

Design:
- Network is partitioned into regions = slots. One slot per habit dimension
  (verbosity / emoji / format) in this MVP - 3 regions.
- Each region has its own MLP: in_dim=8 -> hidden=16 -> out (1 or 3 logits)
- No shared hidden layer => no cross-dim interference. Trained only on its dim.
- Growth: slot MLP doesn't exist until first correction for that dim (neurogenesis)
- Degradation: per-slot importance decay + weight decay (synaptic decay)
  If importance stays low, slot can be pruned (deleted) then regrown later.

This is the hybrid between Rule DynamicCortex (explicit slot value) and
global MLP (shared hidden). Expect: static ~ same as global,
drift better than global because per-dim forgetting is isolated.
"""
from __future__ import annotations
import numpy as np

class MiniMLP:
    def __init__(self, in_dim=8, hidden=16, out_dim=1, lr=0.08, seed=0, weight_decay=0.02):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, np.sqrt(1.0/in_dim), (in_dim, hidden))
        self.b1 = np.zeros(hidden)
        self.out_dim = out_dim
        self.W2 = rng.normal(0, np.sqrt(1.0/hidden), (hidden, out_dim))
        self.b2 = np.zeros(out_dim)
        self.lr = lr
        self.wd = weight_decay
        self.mW1 = np.zeros_like(self.W1); self.vW1 = np.zeros_like(self.W1)
        self.mb1 = np.zeros_like(self.b1); self.vb1 = np.zeros_like(self.b1)
        self.mW2 = np.zeros_like(self.W2); self.vW2 = np.zeros_like(self.W2)
        self.mb2 = np.zeros_like(self.b2); self.vb2 = np.zeros_like(self.b2)
        self.t = 0
        self.beta1, self.beta2, self.eps = 0.9, 0.999, 1e-8

    def forward(self, x):
        h = np.maximum(0, x @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        return logits, h

    def predict(self, x):
        logits, _ = self.forward(x)
        if self.out_dim == 1:
            p = 1/(1+np.exp(-logits[0]))
            return p
        else:
            e = np.exp(logits - np.max(logits))
            return e / np.sum(e)

    def train_step(self, x, y):
        logits, h = self.forward(x)
        if self.out_dim == 1:
            p = 1/(1+np.exp(-logits[0]))
            dlogits = np.array([p - y])
        else:
            e = np.exp(logits - np.max(logits)); p = e/np.sum(e)
            dlogits = p.copy(); dlogits[int(y)] -= 1.0
        dW2 = np.outer(h, dlogits) + self.wd * self.W2
        db2 = dlogits
        dh = dlogits @ self.W2.T
        dh_pre = dh * ((x @ self.W1 + self.b1) > 0)
        dW1 = np.outer(x, dh_pre) + self.wd * self.W1
        db1 = dh_pre
        self.t += 1
        for param, grad, m, v in [
            (self.W1, dW1, self.mW1, self.vW1),
            (self.b1, db1, self.mb1, self.vb1),
            (self.W2, dW2, self.mW2, self.vW2),
            (self.b2, db2, self.mb2, self.vb2),
        ]:
            m[:] = self.beta1*m + (1-self.beta1)*grad
            v[:] = self.beta2*v + (1-self.beta2)*(grad**2)
            mhat = m/(1-self.beta1**self.t); vhat = v/(1-self.beta2**self.t)
            param[:] -= self.lr * mhat/(np.sqrt(vhat)+self.eps)

class SlotMLPMemory:
    name = "E_slot_mlp_hybrid"
    def __init__(self, seed=0, in_dim=8, hidden=16, lr=0.08, weight_decay=0.02, gamma=0.97):
        self.seed = seed
        self.in_dim = in_dim
        self.hidden = hidden
        self.lr = lr
        self.wd = weight_decay
        self.gamma = gamma
        # slots: dim -> MiniMLP (None until grown)
        self.slots: dict[str, MiniMLP] = {}
        self.importance: dict[str, float] = {}
        self.growth = 0; self.prune = 0; self.steps = 0
        self.turn = 0
        self.corr_ema = {"verbosity":0.0,"emoji":0.0,"format":0.0}
        self.last_correct = 0.0

    def _features(self, turn: int):
        x = np.zeros(self.in_dim)
        x[0] = 1.0
        x[1] = self.corr_ema["verbosity"]
        x[2] = self.corr_ema["emoji"]
        x[3] = self.corr_ema["format"]
        x[4] = turn/100.0
        x[5] = self.last_correct
        x[6] = 1.0 if turn % 2 == 0 else 0.0
        x[7] = 1.0
        return x

    def get_pack(self):
        x = self._features(self.turn)
        pack = {}
        if "verbosity" in self.slots:
            p = self.slots["verbosity"].predict(x)
            pack["verbosity"] = "long" if p > 0.5 else "short"
        if "emoji" in self.slots:
            p = self.slots["emoji"].predict(x)
            pack["emoji"] = "always" if p > 0.5 else "never"
        if "format" in self.slots:
            p = self.slots["format"].predict(x)
            pack["format"] = ["code_first","table","prose"][int(np.argmax(p))]
        return pack if pack else None

    def _decay_and_prune(self):
        for dim in list(self.importance.keys()):
            self.importance[dim] *= self.gamma
            if self.importance[dim] < 0.1:
                # Degradation: prune region (delete MLP)
                del self.slots[dim]; del self.importance[dim]
                self.prune += 1

    def observe(self, corrections, true_value, turn, rng):
        self.turn = turn
        x = self._features(turn)
        for dim in ["verbosity","emoji","format"]:
            is_corr = 1.0 if dim in corrections else 0.0
            self.corr_ema[dim] = 0.9*self.corr_ema[dim] + 0.1*is_corr
        self.last_correct = 1.0 if not corrections else 0.0
        # decay
        self._decay_and_prune()
        # growth + training
        for dim in corrections:
            if dim not in self.slots:
                # Neurogenesis: new region
                out_dim = 3 if dim == "format" else 1
                self.slots[dim] = MiniMLP(in_dim=self.in_dim, hidden=self.hidden, out_dim=out_dim,
                                          lr=self.lr, seed=self.seed + hash(dim) % 1000,
                                          weight_decay=self.wd)
                self.importance[dim] = 1.0
                self.growth += 1
            # Train only this region
            y = 0
            if dim == "verbosity": y = 1.0 if true_value[dim]=="long" else 0.0
            elif dim == "emoji": y = 1.0 if true_value[dim]=="always" else 0.0
            elif dim == "format": y = {"code_first":0,"table":1,"prose":2}[true_value[dim]]
            self.slots[dim].train_step(x, y)
            self.importance[dim] = min(2.0, self.importance.get(dim,1.0)+0.4)
            self.steps += 1
        # positive feedback reinforces
        if not corrections:
            for dim in self.importance:
                self.importance[dim] = min(2.0, self.importance[dim]+0.02)

    def stats(self):
        return {"slots": len(self.slots), "growth": self.growth, "prune": self.prune, "steps": self.steps}
