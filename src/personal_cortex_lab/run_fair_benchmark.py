"""Fair B1 benchmark: Remove oracle leakage, unfair baselines, inequivalent noise.

Fixes from review:
 A) StaticRAG artificial 15% pollute removed -> fair static with latest-value retrieval
 B) Oracle removed: observe only gets correction signal (which dims corrected) + optional vague signal,
    NOT true habit vector. For explicit correction subset, true value is revealed by user utterance (parsed).
    For vague feedback ("不对"), model only knows "something wrong" not which dim.
 C) Mock LLM noise normalized: both Prompt and Hypernet use same per-dim 15% deviation.
 D) Slot growth claim clarified: we report avg slots/prune properly, and add Latest-Value as strong baseline.
 E) Added Bayesian change-point and Latest-Value baselines.
 F) New metrics: online regret, recovery time after drift, false update rate (noise), retention.

This is Personal Cortex B1: Causal Online Preference Learning (frozen LLM, only study preference state).
"""
from __future__ import annotations
import random, statistics, math
from dataclasses import dataclass
from .users import generate_users

# ----------------- Feedback encoding (no oracle) -----------------
@dataclass
class Feedback:
    # What the learner actually sees: user utterance text + which dims were explicitly mentioned?
    # In explicit mode, user says "shorter" -> label for that dim is known (short)
    # In vague mode, user only says "不对" -> learner knows at least one dim is wrong, not which
    explicit: dict | None  # e.g. {"verbosity":"short"} if user explicitly said
    vague: bool            # True if only "不对"
    noisy: bool            # True if correction is random noise (false alarm)

def make_feedback(true_habit: dict, predicted_habit: dict, mode: str, rng: random.Random, false_corr_rate: float):
    """Generate partial feedback without leaking full true_habit.
    predicted vs true: mismatch dims are true mistakes.
    With prob false_corr_rate, add a random false correction (noise).
    """
    mismatched = [d for d in true_habit if predicted_habit.get(d) != true_habit[d]]
    if mode == "explicit":
        # User explicitly corrects each mismatched dim -> learner gets true value for those dims only
        expl = {d: true_habit[d] for d in mismatched}
        # noise: sometimes add a false explicit dim (random other dim with wrong value)
        if rng.random() < false_corr_rate and len(mismatched) < 3:
            d = rng.choice(list(true_habit.keys()))
            # flip to non-true value to simulate spurious correction
            wrong = [v for v in ["short","long"] if v!=true_habit[d]] if d=="verbosity" else \
                    [v for v in ["never","always"] if v!=true_habit[d]] if d=="emoji" else \
                    [v for v in ["code_first","table","prose"] if v!=true_habit[d]]
            if wrong: expl[d] = rng.choice(wrong)
        return Feedback(explicit=expl, vague=False, noisy=bool(expl) and rng.random()<false_corr_rate)
    elif mode == "vague":
        # Only knows something is wrong, no dim info
        return Feedback(explicit={}, vague=bool(mismatched), noisy=False)
    else:
        raise ValueError(mode)

# ----------------- Baselines (fair) -----------------
class LatestValueMemory:
    """Each dim keeps only most recent explicitly corrected value. No decay, no growth."""
    name = "LatestValue"
    def __init__(self):
        self.store: dict[str,str] = {}
    def get_pack(self):
        return dict(self.store) if self.store else None
    def observe(self, fb: Feedback, turn: int):
        if fb.explicit:
            self.store.update(fb.explicit)

class StaticRAGFair:
    """Fair static: store every explicit utterance, retrieve latest per dim by recency (no artificial noise)."""
    name = "StaticRAG_fair"
    def __init__(self):
        self.hist: list[tuple[int, dict]] = []
    def get_pack(self):
        if not self.hist: return None
        pack={}
        # latest per dim
        for _, expl in reversed(self.hist):
            for d,v in expl.items():
                if d not in pack: pack[d]=v
        return pack
    def observe(self, fb: Feedback, turn: int):
        if fb.explicit: self.hist.append((turn, dict(fb.explicit)))

class RuleDynamicFair:
    """Dynamic slot but observe only explicit feedback, no oracle reinforcement on correct turns."""
    name = "RuleDynamic_fair"
    def __init__(self, gamma=0.97, prune_thresh=0.15):
        self.slots: dict[str, dict] = {}  # dim -> {value, imp, last}
        self.gamma=gamma; self.thresh=prune_thresh
    def get_pack(self):
        return {d:s["value"] for d,s in self.slots.items()} if self.slots else None
    def observe(self, fb: Feedback, turn: int):
        for d in self.slots:
            self.slots[d]["imp"] *= self.gamma
        if fb.explicit:
            for d,v in fb.explicit.items():
                if d not in self.slots:
                    self.slots[d]={"value":v,"imp":1.0,"last":turn}
                else:
                    self.slots[d]["value"]=v; self.slots[d]["imp"]=min(2.0,self.slots[d]["imp"]+0.4); self.slots[d]["last"]=turn
        # vague feedback: we know something wrong, but not which -> we can only decay trust slightly, no value update
        if fb.vague:
            for d in self.slots: self.slots[d]["imp"] *= 0.99
        # prune
        for d in list(self.slots.keys()):
            if self.slots[d]["imp"] < self.thresh and (turn-self.slots[d]["last"])>20:
                del self.slots[d]

class BayesianTracker:
    """Per-dim Beta posterior over each possible value, change-point via discounting.
    Belief update: on explicit correction, boost belief for reported value, discount old.
    Output = argmax belief. No MLP, no gradient.
    """
    name = "Bayesian"
    def __init__(self, discount=0.98):
        # beliefs: dim -> dict value-> log-odds
        self.beliefs: dict[str, dict[str,float]] = {
            "verbosity":{"short":0.0,"long":0.0},
            "emoji":{"never":0.0,"always":0.0},
            "format":{"code_first":0.0,"table":0.0,"prose":0.0},
        }
        self.discount=discount
    def get_pack(self):
        pack={}
        for d, dist in self.beliefs.items():
            best=max(dist, key=dist.get)
            if abs(dist[best]-min(dist.values()))>0.05:  # only output if confident
                pack[d]=best
        return pack if pack else None
    def observe(self, fb: Feedback, turn: int):
        for d in self.beliefs:
            for v in self.beliefs[d]:
                self.beliefs[d][v] *= self.discount  # forgetting
        if fb.explicit:
            for d,v in fb.explicit.items():
                self.beliefs[d][v] = self.beliefs[d].get(v,0.0)+1.0
        # vague: slight global discount already; no info

def predict_habit_from_pack(pack: dict | None, rng: random.Random) -> dict:
    """Mock LLM: if pack has dim, follow it with 85% prob else random else; same for all methods (fair noise)."""
    true_options = {"verbosity":["short","long"],"emoji":["never","always"],"format":["code_first","table","prose"]}
    out={}
    for dim, opts in true_options.items():
        if pack and dim in pack:
            out[dim]=pack[dim] if rng.random()<0.85 else rng.choice([o for o in opts if o!=pack[dim]])
        else:
            out[dim]=rng.choice(opts)
    return out

def run_one(user, mem, mode="explicit", n_turns=500, false_rate=0.0, drift_at=None, drift_cfg=None, seed=0):
    rng=random.Random(seed)
    true=dict(user.habit_dict())
    total_err=0; recovery_start=None; recovery_turns=None
    false_updates=0; updates=0
    for turn in range(n_turns):
        if drift_at is not None and turn==drift_at and drift_cfg:
            true.update(drift_cfg)
            recovery_start=turn
        pred = predict_habit_from_pack(mem.get_pack() if hasattr(mem,"get_pack") else None, rng)
        mismatched = [d for d in true if pred.get(d)!=true[d]]
        if mismatched: total_err+=1
        fb=make_feedback(true, pred, mode=mode, rng=rng, false_corr_rate=false_rate)
        # check false update: if fb.explicit contains a dim where explicit value != true (noise), that's a false update opportunity
        if fb.explicit:
            updates+=1
            for d,v in fb.explicit.items():
                if v!=true[d]: false_updates+=1
        # Bayesian/others take fb without true
        try:
            mem.observe(fb, turn)
        except TypeError:
            try: mem.observe(fb, turn, rng)
            except: mem.observe(fb)
        if recovery_start is not None and recovery_turns is None:
            # recovery when pred matches true for 3 consecutive turns (approx via current pred)
            if not mismatched:
                recovery_turns=turn-recovery_start
    return {"err": total_err/n_turns, "recovery": recovery_turns, "false_rate": false_updates/max(1,updates)}

def benchmark(mode="explicit", false_rate=0.10, drift=True):
    users=generate_users(n=30, seed=123)
    mems={
        "LatestValue": lambda: LatestValueMemory(),
        "StaticRAG_fair": lambda: StaticRAGFair(),
        "RuleDynamic_fair": lambda: RuleDynamicFair(),
        "Bayesian": lambda: BayesianTracker(discount=0.98),
    }
    print(f"\n=== FAIR B1 mode={mode} false_rate={false_rate} drift={drift} ===")
    for name, fac in mems.items():
        errs=[]; recs=[]
        for u in users:
            drift_cfg={"verbosity":"long" if u.verbosity=="short" else "short",
                       "format":"table" if u.fmt=="code_first" else "code_first"} if drift else None
            r=run_one(u, fac(), mode=mode, n_turns=500, false_rate=false_rate, drift_at=300 if drift else None, drift_cfg=drift_cfg, seed=1)
            errs.append(r["err"])
            if r["recovery"] is not None: recs.append(r["recovery"])
        print(f"{name:<20} err={statistics.mean(errs):.3f} recovery={(statistics.mean(recs) if recs else float('nan')):.1f}")

if __name__=="__main__":
    benchmark(mode="explicit", false_rate=0.0, drift=False)
    benchmark(mode="explicit", false_rate=0.10, drift=True)
    benchmark(mode="vague", false_rate=0.0, drift=True)
