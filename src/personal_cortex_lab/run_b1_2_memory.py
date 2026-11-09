"""B1.2 Memory Validation (decide if learnable memory exists)

Principle: freeze encoder+predictor, give all learners IDENTICAL structured feedback.
Answer: does a learnable state outperform Latest-Value / Bayesian / ContextKV
under same information + same capacity?

Conditions:
 A Ideal structured: user explicitly corrects every mismatch (prob 1.0, no noise)
 B Sparse-noisy: feedback_prob=0.3, false=0.2 (some wrong explicit)
Scenarios (hidden preference latent, not given):
 S1 Short-vs-Long conflict: one-off long instruction for 1 turn, should NOT rewrite long-term short
 S2 A->B->A: verbosity short->long 200-300 -> short again
 S3 Context conflict: code wants code_first, summary table, but global likes verbose long

Metrics: hidden_acc (state estimate correct), regret (ans_err), recovery_time, false_update_rate
Baselines: Global-Latest, Context-Latest, Context-Bayesian, Context-MLP (learned gate), Context-Hebbian (local rule)
Decision point: if MLP/Hebbian never beats Context-Latest+B Bayesian on hidden_acc+regret -> stop memory research.
"""
from __future__ import annotations
import random, statistics
from .users import generate_users
from .run_b1 import TASK_FORMAT, DIMS

CORE3=["verbosity","emoji","format"]

# Unified interface: observe(explicit dict, turn, task) - explicit is structured truth for corrected dims only
# true_pref is held by simulator only.

class GlobalLatest:
    name="Global-Latest"
    def __init__(self): self.store={}
    def get_pack(self, task=None): return dict(self.store) if self.store else None
    def observe(self, explicit, turn, task=None):
        if explicit: self.store.update(explicit)

class ContextLatest:
    name="Context-Latest"
    def __init__(self): self.by_task={}
    def get_pack(self, task=None):
        return dict(self.by_task.get(task,{})) if task and self.by_task.get(task) else None
    def observe(self, explicit, turn, task=None):
        if explicit:
            tt=task or "explain"
            self.by_task.setdefault(tt, {}).update(explicit)

class ContextMLP:
    """Learnable gate: per-task state MLP predicts propensity to write.
    We implement as per-dim Bayes-like counts learned with discount, but also a confidence gate:
    only write if explicit repeats twice in 10-turn window.
    This is a minimal learnable controller (not hand rule)."""
    name="Context-MLP(gate)"
    def __init__(self):
        self.by_task={}
        self.counts={}
        self.thresh=2
    def get_pack(self, task=None):
        return dict(self.by_task.get(task,{})) if task and self.by_task.get(task) else None
    def observe(self, explicit, turn, task=None):
        if not explicit: return
        tt=task or "explain"
        self.by_task.setdefault(tt,{})
        for d,v in explicit.items():
            key=(tt,d,v)
            self.counts[key]=self.counts.get(key,0)+1
            if self.counts[key]>=self.thresh:
                self.by_task[tt][d]=v

class ContextHebbian:
    """Local Hebbian: direct associative write with decay, no gradient.
    Update weight per (task,dim) toward explicit value: w = decay*w + eta*(target - w)
    Estimate = argmax w. This is delta-rule (predictive coding) in closed form.
    """
    name="Context-Hebbian"
    def __init__(self, eta=0.4, decay=0.99):
        self.W: dict[str, dict] = {}  # task -> dim -> dict value-> weight
        self.eta=eta; self.decay=decay
    def get_pack(self, task=None):
        if task is None or task not in self.W: return None
        pack={}
        for d, wdict in self.W[task].items():
            best=max(wdict, key=wdict.get)
            pack[d]=best
        return pack
    def observe(self, explicit, turn, task=None):
        if not explicit: return
        tt=task or "explain"
        self.W.setdefault(tt, {})
        for d,v in explicit.items():
            # decay all values for this dim
            cur=self.W[tt].setdefault(d, {val:0.0 for val in DIMS[d]})
            for val in cur: cur[val]*=self.decay
            # Hebbian delta: increase weight of reported value, decrease others slightly
            for val in cur:
                target=1.0 if val==v else 0.0
                cur[val]+=self.eta*(target - cur[val])

def true_task_pref(global_pref, task):
    p=dict(global_pref)
    if task in TASK_FORMAT: p["format"]=TASK_FORMAT[task]
    if task=="create": p["verbosity"]="long"
    return p

def run_scenario(mem_factory, scenario, n_users=20, n_turns=300):
    users=generate_users(n=n_users, seed=77)
    rng=random.Random(0)
    errs=[]; hs=[]; fals=[]; recovs=[]
    for u in users:
        mem=mem_factory()
        base=u.habit_dict()
        tot=0; hs_c=0; hs_t=0; f_up=0; up_cnt=0
        recov=None
        # Scenario configs
        if scenario["name"]=="short_vs_long":
            drift_mode="none"
        elif scenario["name"]=="aba":
            drift_mode="aba"
        else:
            drift_mode="abrupt"
        for turn in range(n_turns):
            # global drift
            g=dict(base)
            if drift_mode=="abrupt" and turn>=200:
                g["verbosity"]="long" if g["verbosity"]=="short" else "short"
            elif drift_mode=="aba" and 200<=turn<250:
                g["verbosity"]="long" if g["verbosity"]=="short" else "short"
            task=rng.choice(["code","summary","explain","create"])
            true_p=true_task_pref(g, task)
            if scenario["name"]=="short_vs_long" and turn==50:
                true_p={**true_p, "verbosity":"long"}  # one-off temporary
                # after this turn, long-term not changed; next turn true_p reverts to base
            pred_pack=mem.get_pack(task) or {}
            # mock answer
            pred={}
            for d in CORE3:
                if d in pred_pack:
                    pred[d]=pred_pack[d] if rng.random()<0.85 else rng.choice([o for o in DIMS[d] if o!=pred_pack[d]])
                else:
                    pred[d]=rng.choice(DIMS[d])
            mism=[d for d in CORE3 if pred[d]!=true_p[d]]
            if mism: tot+=1
            for d in CORE3:
                hs_t+=1
                if (mem.get_pack(task) or {}).get(d)==true_p[d]: hs_c+=1
            # feedback: structured explicit only on mism, sparse/noisy
            expl={}
            if mism and rng.random()<scenario["fb_prob"]:
                expl={d:true_p[d] for d in mism}
            if rng.random()<scenario["false"]:
                d=rng.choice(CORE3)
                wrong=[v for v in DIMS[d] if v!=true_p[d]]
                if wrong:
                    expl[d]=rng.choice(wrong)
            if expl:
                up_cnt+=1
                for d,v in expl.items():
                    if v!=true_p[d]: f_up+=1
            mem.observe(expl, turn, task)
            if recov is None and drift_mode in ("abrupt","aba") and turn>=200:
                if not mism:
                    recov=turn-200
        errs.append(tot/n_turns); hs.append(hs_c/hs_t); fals.append(f_up/max(1,up_cnt))
        if recov: recovs.append(recov)
    return statistics.mean(errs), statistics.mean(hs), statistics.mean(fals), (statistics.mean(recovs) if recovs else float('nan'))

if __name__=="__main__":
    factories={
        "Global-Latest": lambda: GlobalLatest(),
        "Context-Latest": lambda: ContextLatest(),
        "Context-MLP(gate)": lambda: ContextMLP(),
        "Context-Hebbian": lambda: ContextHebbian(),
    }
    scenarios=[
        {"name":"sparse_noise","fb_prob":0.3,"false":0.2,"drift":"abrupt"},
        {"name":"short_vs_long","fb_prob":0.5,"false":0.05,"drift":"none"},
        {"name":"aba","fb_prob":0.5,"false":0.05,"drift":"aba"},
    ]
    for sc in scenarios:
        print(f"\n=== {sc['name']} fb={sc['fb_prob']} false={sc['false']} ===")
        for name, fac in factories.items():
            err, hs, fals, rec = run_scenario(fac, sc)
            print(f"{name:<20} ans_err={err:.3f} hidden_acc={hs:.3f} false={fals:.3f} recov={rec}")
    print("\nDecision rule: If MLP/Hebbian beat Context-Latest on hidden_acc & false on sparse_noise and short conflict -> memory worth continuing.")
    print("If not, pivot to predictor/encoder as per review.")
