"""Lightweight text embedding without torch: hash bag-of-words, 64-dim.

For Chinese, split into characters and 2-grams, hash each token into slots.
This fakes a frozen sentence encoder dependent only on numpy+hashlib.
Sufficient to separate "先给代码" vs "用表格讲" contexts for demo.
"""
from __future__ import annotations
import hashlib, numpy as np

def tokenize(s: str):
    toks = list(s)  # chars
    # add bigrams for better separation
    toks += [s[i:i+2] for i in range(max(0, len(s)-1))]
    return [t for t in toks if t.strip()]

def hash_embed(s: str, dim=64) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    for tok in tokenize(s):
        h = int(hashlib.md5(tok.encode("utf-8", errors="ignore")).hexdigest(), 16)
        idx = h % dim
        sign = 1 if (h//dim)%2==0 else -1
        v[idx] += sign
    # L2 normalize
    n = np.linalg.norm(v)
    if n>0: v/=n
    return v

def project(x: np.ndarray, out_dim=11, seed=0):
    rng = np.random.default_rng(seed)
    # random projection matrix fixed
    if not hasattr(project, "_M"):
        proj = rng.normal(0, 0.1, (len(x), out_dim))
        project._M = proj
    return x @ project._M
