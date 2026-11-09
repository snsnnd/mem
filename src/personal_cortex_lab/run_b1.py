"""B1: Causal Online Preference Learning - Fair, partial-observable, sparse/noisy feedback.

Core question: Can a plastic small network track a hidden user preference z_t from sparse noisy
partial observations, trade off fast adaptation vs stability, under bounded memory?

We freeze the LLM (mock) and only study preference state estimation h_t.
True preference is NEVER given to the learner, only via feedback observations.

Variables:
  feedback_prob: how often user corrects when mistaken (0.1,0.3,0.6,1.0)
  false_corr_rate: noise corrections (0,0.1,0.2)
  drift_mode: none | abrupt (flip at 300) | gradual (linear over 100 turns) | aba (flip then back)
  context: 4 tasks, each has its own expected format (+1 unseen task for generalization)
Baselines: NoMemory, LatestValue, TimeDecay, ContextKV, Bayesian, RuleDynamic, Hebbian
Metrics: online error, recovery_time, false_update, retention (after 100 idle turns), context_gen
"""
from __future__ import annotations
import random, statistics, math
from dataclasses import dataclass

# ---------- Extended user ----------
DIMS = {
    "verbosity": ["short","long"],
    "emoji": ["never","always"],
    "format": ["code_first","table","prose"],
    "tone": ["direct","polite"],
    "examples": ["with","without"],
    "citation": ["with","without"],
    "code_style": ["pythonic","pseudo"],
    "humor": ["never","sometimes"],
}
CORE3 = ["verbosity","emoji","format"]  # keep core for comparability; other 5 are extended

TASK_FORMAT = {
    "code": "code_first",
    "summary": "table",
    "explain": "prose",
    "create": "prose",   # create will have long verbosity as well
    "translate": "prose", # unseen task in train, used for generalization test
    "brainstorm": "table", # unseen
}

def generate_users_ext(n=50, seed=0, correlation=True):
    rng = random.Random(seed)
    users=[]
    for i in range(n):
        # sample core 3
        u = {
            "verbosity": rng.choice(DIMS["verbosity"]),
            "emoji": rng.choice(DIMS["emoji"]),
            "format": rng.choice(DIMS["format"]),
        }
        # correlated extras: if verbosity long -> examples more likely with, etc.
        if correlation:
            u["tone"] = rng.choice(DIMS["tone"])
            u["examples"] = "with" if (u["verbosity"]=="long" and rng.random()<0.8) else rng.choice(DIMS["examples"])
            u["citation"] = rng.choice(DIMS["citation"])
            u["code_style"] = rng.choice(DIMS["code_style"])
            u["humor"] = "sometimes" if (u["emoji"]=="always" and rng.random()<0.5) else "never"
        else:
            for d in DIMS:
                if d not in u: u[d]=rng.choice(DIMS[d])
        users.append(u)
    return users

# ---------- Feedback (partial, sparse, noisy) ----------
@dataclass
class Feedback:
    explicit: dict | None  # only dims user explicitly mentioned
    vague: bool

def true_pref_for(user, task):
    pref = dict(user)
    # context overrides format for task
    if task in TASK_FORMAT:
        pref["format"] = TASK_FORMAT[task]
    if task == "create":
        pref["verbosity"] = "long"
    return pref

def make_feedback(true_pref, pred_pref, feedback_prob, false_rate, mode, rng):
    # Determine mismatched dims (real mistakes)
    mism = [d for d in CORE3 if pred_pref.get(d) != true_pref.get(d)]
    # Sparse: user only corrects with prob feedback_prob when mistaken
    if not mism:
        # No mistake, no feedback (unless false alarm noise)
        if rng.random() < false_rate*0.2:
            # spurious vague
            return Feedback(explicit={}, vague=True)
        return Feedback(explicit={}, vague=False)
    # Mistake exists
    if rng.random() > feedback_prob:
        # User didn't correct this turn - learner sees nothing
        return Feedback(explicit={}, vague=False)
    # User corrects
    if mode == "explicit":
        expl = {d: true_pref[d] for d in mism}
        # noise: with prob false_rate add a false dim
        if rng.random() < false_rate:
            d = rng.choice(CORE3)
            # flip to wrong value
            wrong = [v for v in DIMS[d] if v != true_pref[d]]
            if wrong and d not in expl:
                expl[d] = rng.choice(wrong)
        return Feedback(explicit=expl, vague=False)
    else: # vague
        return Feedback(explicit={}, vague=True)

# ---------- Learners (all only see Feedback, never true_pref) ----------
class NoMem:
    name="NoMemory"
    def get_pack(self): return None
    def observe(self, fb, turn, task=None): pass

class LatestValue:
    name="LatestValue"
    def __init__(self): self.store={}
    def get_pack(self): return dict(self.store) if self.store else None
    def observe(self, fb, turn, task=None):
        if fb.explicit:
            self.store.update(fb.explicit)

class TimeDecay:
    name="TimeDecay"
    def __init__(self, gamma=0.95):
        self.gamma=gamma; self.store={}; self.imp={}
    def get_pack(self):
        return dict(self.store) if any(v>0.2 for v in self.imp.values()) else None
    def observe(self, fb, turn, task=None):
        for d in list(self.imp.keys()):
            self.imp[d]*=self.gamma
            if self.imp[d]<0.2:
                self.store.pop(d,None); self.imp.pop(d,None)
        if fb.explicit:
            for d,v in fb.explicit.items():
                self.store[d]=v; self.imp[d]=1.0

class ContextKV:
    """Keyed by task: learns format per task separately."""
    name="ContextKV"
    def __init__(self):
        self.by_task: dict[str, dict] = {}
        self.cur_task="explain"
    def get_pack(self, task=None):
        if task and task in self.by_task:
            return dict(self.by_task[task])
        # fallback to global latest
        if self.by_task:
            # merge all, latest wins (approx)
            merged={}
            for _,d in self.by_task.items(): merged.update(d)
            return merged
        return None
    def observe(self, fb, turn, task=None):
        tt = task if task else self.cur_task
        if fb.explicit:
            self.by_task.setdefault(tt, {}).update(fb.explicit)

class BayesianTracker:
    name="Bayesian"
    def __init__(self, discount=0.98):
        self.d=discount
        self.beliefs={d:{v:0.0 for v in DIMS[d]} for d in CORE3}
    def get_pack(self):
        pack={}
        for d, dist in self.beliefs.items():
            best=max(dist, key=dist.get)
            if abs(dist[best]-min(dist.values()))>0.1:
                pack[d]=best
        return pack if pack else None
    def observe(self, fb, turn, task=None):
        for d in self.beliefs:
            for v in self.beliefs[d]:
                self.beliefs[d][v]*=self.d
        if fb.explicit:
            for d,v in fb.explicit.items():
                if d in self.beliefs and v in self.beliefs[d]:
                    self.beliefs[d][v]+=1.0
        # vague gives no dim info, only decay already done

def predict_from_pack(pack, rng):
    out={}
    for d, opts in DIMS.items():
        if d not in CORE3: continue
        if pack and d in pack:
            out[d]=pack[d] if rng.random()<0.85 else rng.choice([o for o in opts if o!=pack[d]])
        else:
            out[d]=rng.choice(opts)
    # fill extras randomly (not tracked for core metric)
    for d in DIMS:
        if d not in out: out[d]=rng.choice(DIMS[d])
    return out

def run_one(user, mem, feedback_prob=1.0, false_rate=0.0, mode="explicit",
            drift_mode="abrupt", n_turns=500, seed=0):
    rng=random.Random(seed)
    # drift config
    base=dict(user)
    drifted=dict(base)
    if drift_mode not in ("none",):
        drifted.update({"verbosity":"long" if base["verbosity"]=="short" else "short",
                        "format":"table" if base["format"]=="code_first" else "code_first"})
    total_err=0
    recov_start=None; recov_time=None
    false_updates=0; updates=0
    consec_correct=0
    for turn in range(n_turns):
        # gradual drift: linear interpolation over 100 turns (for simplicity, flip gradually by prob)
        cur_true = dict(base)
        if drift_mode == "abrupt":
            if turn>=300: cur_true.update(drifted)
        elif drift_mode == "gradual":
            if turn>=300 and turn<400:
                # with increasing prob, true becomes drifted
                if rng.random() < (turn-300)/100:
                    cur_true.update(drifted)
            elif turn>=400: cur_true.update(drifted)
        elif drift_mode == "aba":
            if 300 <= turn < 400: cur_true.update(drifted)
            # after 400 back to base
        task=rng.choice(["code","summary","explain","create"])
        cur_true = true_pref_for(cur_true, task)

        pack = mem.get_pack(task) if mem.name=="ContextKV" else mem.get_pack()
        pred = predict_from_pack(pack, rng)
        mism = [d for d in CORE3 if pred.get(d)!=cur_true.get(d)]
        if mism: total_err+=1

        fb = make_feedback(cur_true, pred, feedback_prob, false_rate, mode, rng)
        if fb.explicit:
            updates+=1
            for d,v in fb.explicit.items():
                if v!=cur_true.get(d): false_updates+=1
        try: mem.observe(fb, turn, task)
        except TypeError: mem.observe(fb, turn)

        if drift_mode=="abrupt" and turn==300:
            recov_start=300
        if recov_start is not None and recov_time is None and not mism:
            consec_correct+=1
            if consec_correct>=3:
                recov_time=turn-recov_start
        elif mism:
            consec_correct=0

    return {"err": total_err/n_turns, "recovery": recov_time if recov_time is not None else n_turns,
            "false_rate": false_updates/max(1,updates)}

def sweep():
    factories = {
        "NoMemory": lambda: NoMem(),
        "LatestValue": lambda: LatestValue(),
        "TimeDecay": lambda: TimeDecay(gamma=0.95),
        "ContextKV": lambda: ContextKV(),
        "Bayesian": lambda: BayesianTracker(discount=0.98),
    }
    configs = [
        ("dense explicit, no drift", dict(feedback_prob=1.0, false_rate=0.0, mode="explicit", drift_mode="none")),
        ("sparse 30%, noise10%, abrupt", dict(feedback_prob=0.3, false_rate=0.10, mode="explicit", drift_mode="abrupt")),
        ("sparse 30%, noise10%, gradual", dict(feedback_prob=0.3, false_rate=0.10, mode="explicit", drift_mode="gradual")),
        ("sparse 30%, noise10%, aba", dict(feedback_prob=0.3, false_rate=0.10, mode="explicit", drift_mode="aba")),
        ("vague, noise10%, abrupt", dict(feedback_prob=0.3, false_rate=0.10, mode="vague", drift_mode="abrupt")),
    ]
    users = generate_users_ext(n=30, seed=123)
    for cfg_name, cfg in configs:
        print(f"\n--- {cfg_name} ---")
        for name, fac in factories.items():
            errs=[]; recs=[]; frs=[]
            for u in users:
                r=run_one(u, fac(), **cfg, n_turns=500, seed=42)
                errs.append(r["err"]); recs.append(r["recovery"]); frs.append(r["false_rate"])
            print(f"{name:<14} err={statistics.mean(errs):.3f}±{statistics.pstdev(errs):.3f} recov={statistics.mean(recs):.1f} false={statistics.mean(frs):.3f}")

if __name__=="__main__":
    sweep()
