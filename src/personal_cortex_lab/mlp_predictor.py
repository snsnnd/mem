"""Trainable small MLP habit predictor (numpy-only, no torch).

Architecture:
 Input (dim=10) -> Hidden(32, ReLU) -> Output 5 logits:
   logit_verbosity(1), logit_emoji(1), logits_format(3)
Loss: cross-entropy on dimensions that were corrected (true value revealed).
Training: online SGD/Adam step per correction event. This replaces rule-based slot update.

Input features (10):
 0: bias 1.0
 1-3: one-hot last task type (we have 3 task types simplified)
 4: recent verbosity correction rate (exp moving avg)
 5: recent emoji correction rate
 6: recent format correction rate
 7: turn / 100 (time)
 8: last answer was correct? (1/0)
 9: count of total turns normalized (log)

This is tiny (~500 params), runs on CPU, no external deps.
"""
from __future__ import annotations
import numpy as np
import random

class NumpyMLP:
    def __init__(self, in_dim=10, hidden=32, out_dim=5, lr=0.08, seed=0, weight_decay=0.05):
        rng = np.random.default_rng(seed)
        self.in_dim = in_dim
        self.hidden = hidden
        self.out_dim = out_dim
        self.lr = lr
        self.weight_decay = weight_decay
        # Xavier init
        self.W1 = rng.normal(0, np.sqrt(1.0/in_dim), (in_dim, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, np.sqrt(1.0/hidden), (hidden, out_dim))
        self.b2 = np.zeros(out_dim)
        # Adam state
        self.mW1 = np.zeros_like(self.W1); self.vW1 = np.zeros_like(self.W1)
        self.mb1 = np.zeros_like(self.b1); self.vb1 = np.zeros_like(self.b1)
        self.mW2 = np.zeros_like(self.W2); self.vW2 = np.zeros_like(self.W2)
        self.mb2 = np.zeros_like(self.b2); self.vb2 = np.zeros_like(self.b2)
        self.t = 0
        self.beta1, self.beta2, self.eps = 0.9, 0.999, 1e-8

    def _relu(self, x): return np.maximum(0, x)
    def _softmax(self, x):
        e = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e / np.sum(e, axis=-1, keepdims=True)

    def forward(self, x: np.ndarray):
        # x: (in_dim,)
        h_pre = x @ self.W1 + self.b1
        h = self._relu(h_pre)
        logits = h @ self.W2 + self.b2
        # split
        p_verb = 1/(1+np.exp(-logits[0]))  # sigmoid for verbosity: 0=short,1=long
        p_emoji = 1/(1+np.exp(-logits[1]))
        p_fmt = self._softmax(logits[2:5])
        return {"p_verb": p_verb, "p_emoji": p_emoji, "p_fmt": p_fmt, "logits": logits, "h": h, "h_pre": h_pre}

    def predict_pack(self, x: np.ndarray) -> dict:
        out = self.forward(x)
        verbosity = "long" if out["p_verb"] > 0.5 else "short"
        emoji = "always" if out["p_emoji"] > 0.5 else "never"
        fmt = ["code_first", "table", "prose"][int(np.argmax(out["p_fmt"]))]
        return {"verbosity": verbosity, "emoji": emoji, "format": fmt}

    def train_step(self, x: np.ndarray, true_habit: dict | None, corrected_dims: list[str]):
        """One Adam step on corrected dims only. true_habit needed for supervision."""
        if not corrected_dims or true_habit is None:
            return 0.0
        fwd = self.forward(x)
        logits = fwd["logits"]
        h = fwd["h"]; h_pre = fwd["h_pre"]

        # Build target and dL/dlogits
        dlogits = np.zeros_like(logits)
        loss = 0.0
        # verbosity: y=1 if long
        if "verbosity" in corrected_dims:
            y = 1.0 if true_habit["verbosity"] == "long" else 0.0
            p = fwd["p_verb"]
            loss += -(y*np.log(p+1e-9) + (1-y)*np.log(1-p+1e-9))
            dlogits[0] = p - y
        if "emoji" in corrected_dims:
            y = 1.0 if true_habit["emoji"] == "always" else 0.0
            p = fwd["p_emoji"]
            loss += -(y*np.log(p+1e-9) + (1-y)*np.log(1-p+1e-9))
            dlogits[1] = p - y
        if "format" in corrected_dims:
            y_idx = {"code_first":0, "table":1, "prose":2}[true_habit["format"]]
            p_fmt = fwd["p_fmt"]
            loss += -np.log(p_fmt[y_idx]+1e-9)
            dlogits[2:5] = p_fmt.copy()
            dlogits[2+y_idx] -= 1.0

        # Backprop
        # dW2 = h outer dlogits
        dW2 = np.outer(h, dlogits)
        db2 = dlogits
        dh = dlogits @ self.W2.T
        dh_pre = dh * (h_pre > 0)  # relu grad
        dW1 = np.outer(x, dh_pre)
        db1 = dh_pre

        # Weight decay (forgetting) - stronger to handle habit drift
        dW1 += self.weight_decay * self.W1
        dW2 += self.weight_decay * self.W2

        # Adam
        self.t += 1
        for name, param, grad, m, v in [
            ("W1", self.W1, dW1, self.mW1, self.vW1),
            ("b1", self.b1, db1, self.mb1, self.vb1),
            ("W2", self.W2, dW2, self.mW2, self.vW2),
            ("b2", self.b2, db2, self.mb2, self.vb2),
        ]:
            m[:] = self.beta1 * m + (1-self.beta1) * grad
            v[:] = self.beta2 * v + (1-self.beta2) * (grad**2)
            m_hat = m / (1 - self.beta1**self.t)
            v_hat = v / (1 - self.beta2**self.t)
            param[:] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        return float(loss)

class MLPHabitMemory:
    """Memory wrapper using NumpyMLP to predict habit from history features.
    get_pack() returns argmax prediction; observe() trains on corrections.
    """
    name = "D_mlp_predictor"

    def __init__(self, seed=0):
        self.mlp = NumpyMLP(seed=seed)
        self.stats = {"steps": 0, "total_loss": 0.0, "avg_loss_hist": []}
        self.corr_ema = {"verbosity": 0.0, "emoji": 0.0, "format": 0.0}
        self.turn = 0
        self.last_correct = 0.0

    def _features(self, turn: int) -> np.ndarray:
        # Simple engineered features mirroring recent correction rates
        x = np.zeros(10)
        x[0] = 1.0
        x[1] = self.corr_ema["verbosity"]
        x[2] = self.corr_ema["emoji"]
        x[3] = self.corr_ema["format"]
        x[4] = turn / 100.0
        x[5] = self.last_correct
        # task type pseudo: turn % 3 one-hot
        x[6 + (turn % 3)] = 1.0 if turn % 3 < 3 else 0.0
        # cap 10 handled; we use 1-3 as ema, 4 time,5 last,6-8 task one-hot, 9 bias-like denom
        x[9] = 1.0
        return x

    def get_pack(self):
        x = self._features(self.turn)
        return self.mlp.predict_pack(x)

    def observe(self, corrections, true_value, turn, rng):
        self.turn = turn
        x = self._features(turn)
        # Update EMA
        for dim in ["verbosity", "emoji", "format"]:
            is_corr = 1.0 if dim in corrections else 0.0
            self.corr_ema[dim] = 0.9 * self.corr_ema[dim] + 0.1 * is_corr
        self.last_correct = 1.0 if not corrections else 0.0
        loss = self.mlp.train_step(x, true_value, corrections)
        if corrections:
            self.stats["steps"] += 1
            self.stats["total_loss"] += loss

    def stats_dict(self):
        avg_loss = (self.stats["total_loss"] / max(1, self.stats["steps"]))
        return {"steps": self.stats["steps"], "avg_loss": avg_loss}
