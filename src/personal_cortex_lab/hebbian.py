"""Hebbian / hippocampal-style learning - no backprop, local rules only.

Design:
- Direct associative memory: W is input(8) x output(5). Forward is matrix multiply.
- Update is local three-factor: ΔW = eta * surprise(dim) * x ⊗ error(dim)
  where error = y_target - y_pred, surprise = 1 if that dim was corrected.
  This is Hebbian modulated by prediction error (predictive coding style).
- Hippocampal pattern separation: input is sparsified (top-k) so different
  contexts activate different ensembles -> less interference.
- Forgetting: W *= decay (0.99) each turn = synaptic decay.

This is a fast one-shot system: if surprise fires, association strengthens
in one step, unlike gradient MLP which needs many steps to move boundary.
"""
from __future__ import annotations
import numpy as np

class HebbianMemory:
    name = "G_hebbian_local"
    def __init__(self, in_dim=8, hidden=None, eta=0.4, decay=0.99, sparsify_k=3, seed=0):
        self.in_dim = in_dim
        self.out_dim = 5  # verb, emoji, fmt0, fmt1, fmt2
        self.W = np.zeros((in_dim, self.out_dim))  # associative weights
        self.eta = eta
        self.decay = decay
        self.k = sparsify_k
        self.turn = 0
        self.corr_ema = {"verbosity":0.0,"emoji":0.0,"format":0.0}
        self.last_correct = 0.0
        self.rng = np.random.default_rng(seed)
        self._init_random()

    def _init_random(self):
        # small random to break symmetry, but keep near zero for fast start
        self.W += self.rng.normal(0, 0.01, self.W.shape)

    def _features_raw(self, turn:int) -> np.ndarray:
        x = np.zeros(self.in_dim)
        x[0]=1.0
        x[1]=self.corr_ema["verbosity"]
        x[2]=self.corr_ema["emoji"]
        x[3]=self.corr_ema["format"]
        x[4]=turn/100.0
        x[5]=self.last_correct
        x[6]=1.0 if turn%2==0 else 0.0
        x[7]=1.0
        return x

    def _sparsify(self, x: np.ndarray) -> np.ndarray:
        # Pattern separation: keep top-k by absolute value, zero others
        # Keep bias (dim0) always
        y = x.copy()
        if self.k < self.in_dim:
            idx = np.argsort(np.abs(x))[::-1]
            keep = set(idx[:self.k].tolist()); keep.add(0)
            for i in range(self.in_dim):
                if i not in keep:
                    y[i] = 0.0
        return y

    def _forward(self, x: np.ndarray):
        logits = x @ self.W  # (5,)
        p_verb = 1/(1+np.exp(-logits[0]))
        p_emoji = 1/(1+np.exp(-logits[1]))
        # softmax for format
        e = np.exp(logits[2:5] - np.max(logits[2:5])); p_fmt = e/np.sum(e)
        return p_verb, p_emoji, p_fmt, logits

    def get_pack(self):
        x = self._sparsify(self._features_raw(self.turn))
        p_v, p_e, p_f, _ = self._forward(x)
        return {
            "verbosity": "long" if p_v>0.5 else "short",
            "emoji": "always" if p_e>0.5 else "never",
            "format": ["code_first","table","prose"][int(np.argmax(p_f))],
        }

    def observe(self, corrections, true_value, turn, rng):
        self.turn = turn
        x_raw = self._features_raw(turn)
        x = self._sparsify(x_raw)
        # Update EMA for features (this is not gradient, just counting)
        for d in self.corr_ema:
            self.corr_ema[d] = 0.9*self.corr_ema[d] + 0.1*(1.0 if d in corrections else 0.0)
        self.last_correct = 1.0 if not corrections else 0.0

        # Synaptic decay (forgetting) - applied every turn, module-local
        self.W *= self.decay

        # Hebbian delta update only for corrected dims (surprise-gated)
        p_v, p_e, p_f, _ = self._forward(x)
        if "verbosity" in corrections:
            y_target = 1.0 if true_value["verbosity"]=="long" else 0.0
            err = y_target - p_v
            # ΔW[:,0] ∝ x * err  (post error modulated Hebbian)
            self.W[:,0] += self.eta * x * err
        if "emoji" in corrections:
            y_target = 1.0 if true_value["emoji"]=="always" else 0.0
            err = y_target - p_e
            self.W[:,1] += self.eta * x * err
        if "format" in corrections:
            y_idx = {"code_first":0,"table":1,"prose":2}[true_value["format"]]
            target = np.zeros(3); target[y_idx]=1.0
            err_vec = target - p_f  # (3,)
            # Outer product x (8,) x err (3,)
            self.W[:,2:5] += self.eta * np.outer(x, err_vec)

        # Optional: heterosynaptic plasticity - small L2 to keep weights bounded
        # (equivalent to stability term)
        self.W = np.clip(self.W, -5, 5)

    def stats(self):
        return {"W_norm": float(np.linalg.norm(self.W)), "sparsify": self.k}
