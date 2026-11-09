"""Hybrid: explicit fast slot value + MLP slow manager

- Explicit layer (fast): slot.value = true value immediately on correction.
  Solves the SlotMLP slowness (0.524 drift) because forgetting is a jump, not gradient.
- MLP layer (slow): per-dim MiniMLP predicts utility (will slot be correct next turn?)
  Output is a trust score used to adjust importance / pruning.
  It learns per-dim volatility: volatile dims -> decay faster, stable -> keep longer.
  This replaces fixed gamma=0.97 with learned importance.

Intuition: value = fast plasticity, MLP = slow stability manager.
"""
from __future__ import annotations
import numpy as np
from .slot_mlp import MiniMLP  # reuse tiny MLP for manager

class HybridMemory:
    name = "F_hybrid_exp_value_mlp_manager"
    def __init__(self, seed=0, hidden=12, lr=0.06):
        self.slots: dict[str,str] = {}      # dim -> explicit value
        self.importance: dict[str,float] = {}
        self.managers: dict[str,MiniMLP] = {}  # dim -> MLP predicting utility (binary)
        self.seed = seed
        self.hidden = hidden
        self.lr = lr
        self.turn = 0
        self.corr_ema = {"verbosity":0.0,"emoji":0.0,"format":0.0}
        self.last_correct = 0.0
        self.growth = 0; self.prune = 0; self.steps = 0

    def _features(self, turn:int) -> np.ndarray:
        x = np.zeros(8)
        x[0]=1.0
        x[1]=self.corr_ema["verbosity"]
        x[2]=self.corr_ema["emoji"]
        x[3]=self.corr_ema["format"]
        x[4]=turn/100.0
        x[5]=self.last_correct
        x[6]=1.0 if turn%2==0 else 0.0
        x[7]=1.0
        return x

    def get_pack(self):
        return dict(self.slots) if self.slots else None

    def _ensure_manager(self, dim):
        if dim not in self.managers:
            self.managers[dim] = MiniMLP(in_dim=8, hidden=self.hidden, out_dim=1,
                                         lr=self.lr, seed=self.seed + hash(dim)%997,
                                         weight_decay=0.01)

    def observe(self, corrections, true_value, turn, rng):
        self.turn = turn
        x = self._features(turn)
        for d in ["verbosity","emoji","format"]:
            is_c = 1.0 if d in corrections else 0.0
            self.corr_ema[d] = 0.9*self.corr_ema[d] + 0.1*is_c
        self.last_correct = 1.0 if not corrections else 0.0

        # Fast layer: explicit overwrite on correction (surprise jump)
        for dim in corrections:
            if dim not in self.slots:
                self.growth += 1
            self.slots[dim] = true_value[dim]
            # Importance gets a bump, but manager will later modulate
            self.importance[dim] = min(2.0, self.importance.get(dim,1.0)+0.3)
            self._ensure_manager(dim)

        # Slow layer: train manager to predict utility (1 if dim not corrected next observed)
        # For each dim, label = 1 if dim not in corrections (slot was useful), 0 if corrected (slot stale)
        for dim in ["verbosity","emoji","format"]:
            if dim not in self.slots:
                continue
            self._ensure_manager(dim)
            y = 0.0 if dim in corrections else 1.0
            self.managers[dim].train_step(x, y)
            self.steps += 1
            # Use manager output to adjust importance: trust = predicted utility prob
            trust = float(self.managers[dim].predict(x))
            # importance decay is trust-modulated: high trust -> slow decay, low trust -> fast decay
            # Base gamma 0.97, modulated: gamma_eff = 0.97 + 0.02*(trust-0.5)  range [0.96,0.98]
            gamma_eff = 0.97 + 0.02*(trust-0.5)
            self.importance[dim] = float(self.importance.get(dim,1.0) * gamma_eff + (0.05 if y==1 else 0.0))

        # Prune if importance very low (MLP says not useful for long)
        for dim in list(self.importance.keys()):
            if self.importance[dim] < 0.05:
                del self.slots[dim]; del self.importance[dim]; del self.managers[dim]
                self.prune += 1

    def stats(self):
        return {"slots": len(self.slots), "growth": self.growth, "prune": self.prune, "steps": self.steps}
