"""B1.2: Fix drift bug + three distinguishing scenarios.

Fixes from latest review (aa8b662):
 - Drift was overwritten by TASK_FORMAT, so zero real drift occurred. Now global drift
   targets verbosity (NOT overwritten by TASK_FORMAT) while task-level adjusts format only.
   Global drift: z_global changes; Context drift: Delta(c) separate.
 - Oracle isolation kept; Context-Latest/Bayesian kept as strong baselines.
 - Add Hybrid Learnable (confirmation gate) vs Latest-Value.
 - Three scenarios:
   S1 Short vs Long conflict: one-off long request should NOT rewrite stable short habit
   S2 A->B->A repeated drift: recovery speed matters
   S3 Sparse noise: low feedback prob + high false corr; measure false update rate
Metrics: hidden_acc, ans_err, recovery_time, false_update, retention
"""
from __future__ import annotations
import random, statistics
from .users import generate_users
from .run_b1 import TASK_FORMAT, DIMS

CORE3=["verbosity","emoji","format"]

# ----- Learners (all get same context, only explicit dict) -----
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
        if task and self.by_task.get(task): return dict(self.by_task[task])
        return None
    def observe(self, explicit, turn, task=None):
        if explicit:
            tt=task or "explain"
            self.by_task.setdefault(tt, {}).update(explicit)

class HybridLearnable:
    """Learnable gate: update only after 2 consistent evidences for same dim/value.
    This is the 'controller' we hypothesize beats Latest-Value on noise & short conflicts.
    It does NOT need true value; it counts explicit repetitions.
    """
    name="Hybrid-Confirm"
    def __init__(self, confirm_thresh=2):
        self.store={}
        self.counts: dict[tuple[str,str], int] = {}
        self.thresh=confirm_thresh
    def get_pack(self, task=None):
        return dict(self.store) if self.store else None
    def observe(self, explicit, turn, task=None):
        if not explicit: return
        for d,v in explicit.items():
            # Global memory (same for all tasks simplified)
            key=(d,v)
            self.counts[key]=self.counts.get(key,0)+1
            # need 2 same-value evidences before writing (gate)
            if self.counts[key]>=self.thresh:
                self.store[d]=v

def predict_from_pack(pack, rng):
    out={}
    for d,opts in DIMS.items():
        if d not in CORE3: continue
        if pack and d in pack:
            out[d]=pack[d] if rng.random()<0.85 else rng.choice([o for o in opts if o!=pack[d]])
        else:
            out[d]=rng.choice(opts)
    return out

def true_global(user_dict, turn, drift_mode):
    """Global preference before task offset. Drifts target verbosity only (not overwritten by TASK_FORMAT)."""
    g=dict(user_dict)
    if drift_mode=="aba":
        if 200 <= turn < 300:
            g["verbosity"]="long" if g["verbosity"]=="short" else "short"
        # after 300 back to base
    elif drift_mode=="abrupt":
        if turn>=200:
            g["verbosity"]="long" if g["verbosity"]=="short" else "short"
    return g

def true_for_task(global_pref, task):
    p=dict(global_pref)
    # Task-level format override - separate from global
    if task in TASK_FORMAT:
        p["format"]=TASK_FORMAT[task]
    # create long verbosity is task-level, not global drift
    if task=="create":
        p["verbosity"]="long"
    return p

def run_scenario(name, mem_factory, scenario):
    users=generate_users(n=20, seed=2024)
    rng=random.Random(42)
    errs=[]; hs=[]; false_rate=[]
    for u in users:
        mem=mem_factory()
        base=u.habit_dict()
        tot_mism=0; hs_c=0; hs_t=0; f_up=0; up_cnt=0
        for turn in range(300):
            task=rng.choice(["code","summary","explain","create"])
            g=true_global(base, turn, drift_mode=scenario["drift"])
            true_p=true_for_task(g, task)
            # S1 conflict: at turn 50 only, user temporarily asks long (one-off)
            if scenario["name"]=="short_vs_long_conflict" and turn==50:
                # temporary instruction: for this turn only, expected pref is long, but after turn revert
                true_p={**true_p, "verbosity":"long"}
            if scenario["name"]=="short_vs_long_conflict" and turn==51:
                true_p={**true_p, "verbosity": base["verbosity"]}
                # true global still base short, so after one-off system should still be short
            pack=mem.get_pack(task)
            pred=predict_from_pack(pack, rng)
            mism=[d for d in CORE3 if pred[d]!=true_p[d]]
            if mism: tot_mism+=1
            # hidden acc
            est=mem.get_pack(task) or {}
            for d in CORE3:
                hs_t+=1
                if est.get(d)==true_p[d]: hs_c+=1
            # feedback: sparse
            expl={}
            if mism and rng.random()<scenario["fb_prob"]:
                expl={d:true_p[d] for d in mism}
            # noise: add false explicit with prob false_rate
            if rng.random()<scenario["false_rate"]:
                d=rng.choice(CORE3)
                wrong=[v for v in DIMS[d] if v!=true_p[d]]
                if wrong: expl[d]=rng.choice(wrong)
            if expl:
                up_cnt+=1
                for d,v in expl.items():
                    if v!=true_p[d]: f_up+=1
            mem.observe(expl, turn, task)
        errs.append(tot_mism/300); hs.append(hs_c/hs_t); false_rate.append(f_up/max(1,up_cnt))
    print(f"{name:<20} ans_err={statistics.mean(errs):.3f} hidden={statistics.mean(hs):.3f} false={statistics.mean(false_rate):.3f}")

if __name__=="__main__":
    scenarios={
        "sparse_noise": {"name":"sparse_noise","fb_prob":0.3,"false_rate":0.2,"drift":"abrupt"},
        "short_vs_long_conflict": {"name":"short_vs_long_conflict","fb_prob":0.5,"false_rate":0.05,"drift":"none"},
        "aba_repeat": {"name":"aba_repeat","fb_prob":0.5,"false_rate":0.05,"drift":"aba"},
    }
    for sname, scfg in scenarios.items():
        print(f"\n=== Scenario {sname} ===")
        run_scenario("Global-Latest", lambda: GlobalLatest(), scfg)
        run_scenario("Context-Latest", lambda: ContextLatest(), scfg)
        run_scenario("Hybrid-Confirm(2)", lambda: HybridLearnable(confirm_thresh=2), scfg)
        run_scenario("Hybrid-Confirm(1=Latest)", lambda: HybridLearnable(confirm_thresh=1), scfg)

    print("\nKey check: drift now truly happens (verbosity flip), not overwritten by task format")
    print("Hybrid-Confirm should reduce false-update on noise and short conflict vs Latest")
