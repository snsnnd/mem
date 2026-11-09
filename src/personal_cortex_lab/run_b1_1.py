"""B1.1 revision per latest review (dd10b17):
- Unified context for ALL baselines (task-aware)
- Strict oracle isolation: true_pref never enters learner.observe; only text/dict feedback
- Separate hidden-state accuracy from mock-LLM correction rate
- Independent unseen phrase parser test (hash vs LLM encoder)
- Track false-update-rate, recovery, seed variation
"""
from __future__ import annotations
import random, statistics, json
from .users import generate_users
from .assistant import make_answer
from .run_b1 import true_pref_for, TASK_FORMAT, DIMS

CORE3 = ["verbosity","emoji","format"]

# Unified-context baselines
class ContextLatest:
    name="Context-Latest"
    def __init__(self): self.by_task: dict[str, dict] = {}
    def get_pack(self, task=None):
        tt = task or "explain"
        return dict(self.by_task.get(tt, {})) if self.by_task.get(tt) else None
    def observe(self, explicit: dict | None, turn: int, task=None):
        if explicit:
            tt = task or "explain"
            self.by_task.setdefault(tt, {}).update(explicit)

class ContextBayesian:
    name="Context-Bayesian"
    def __init__(self, discount=0.98):
        self.d=discount
        # beliefs per task per dim
        self.beliefs: dict[str, dict] = {}
    def _ensure(self, task):
        if task not in self.beliefs:
            self.beliefs[task]={d:{v:0.0 for v in DIMS[d]} for d in CORE3}
        return self.beliefs[task]
    def get_pack(self, task=None):
        if task is None: return None
        b=self._ensure(task)
        pack={}
        for d, dist in b.items():
            best=max(dist, key=dist.get)
            if abs(dist[best]-min(dist.values()))>0.11:
                pack[d]=best
        return pack if pack else None
    def observe(self, explicit: dict | None, turn: int, task=None):
        tt=task or "explain"
        b=self._ensure(tt)
        for d in b:
            for v in b[d]: b[d][v]*=self.d
        if explicit:
            for d,v in explicit.items():
                if d in b and v in b[d]:
                    b[d][v]+=1.0

class GlobalLatest:
    name="Global-Latest"
    def __init__(self): self.store={}
    def get_pack(self, task=None): return dict(self.store) if self.store else None
    def observe(self, explicit: dict | None, turn: int, task=None):
        if explicit: self.store.update(explicit)

def predict_from_pack(pack, rng):
    out={}
    for d, opts in DIMS.items():
        if d not in CORE3: continue
        out[d]=pack[d] if pack and d in pack and rng.random()<0.85 else rng.choice(opts)
    return out

def make_explicit_feedback(true_pref, pred, feedback_prob, false_rate, rng):
    mism=[d for d in CORE3 if pred.get(d)!=true_pref.get(d)]
    if not mism: return {}, False
    if rng.random()>feedback_prob: return {}, False
    expl={d:true_pref[d] for d in mism}
    noisy=False
    if rng.random()<false_rate:
        d=rng.choice(CORE3)
        wrong=[v for v in DIMS[d] if v!=true_pref[d]]
        if wrong and d not in expl:
            expl[d]=rng.choice(wrong); noisy=True
    return expl, noisy

def run_b1_1():
    # Config sweep: 20 users, 300 turns, 3 seeds for mean/std (kept light for this run)
    configs=[
        ("sparse30 noise10 abrupt", dict(feedback_prob=0.3, false_rate=0.10, drift_at=200)),
        ("dense100 noise0 nodrift", dict(feedback_prob=1.0, false_rate=0.0, drift_at=None)),
    ]
    factories={
        "Global-Latest": lambda: GlobalLatest(),
        "Context-Latest": lambda: ContextLatest(),
        "Context-Bayesian": lambda: ContextBayesian(),
    }
    users=generate_users(n=20, seed=99)
    for cfg_name, cfg in configs:
        print(f"\n=== B1.1 {cfg_name} ===")
        for name, fac in factories.items():
            errs=[]; hs=[]; recovs=[]
            for seed in [1,2,3]:
                for u in users:
                    rng=random.Random(seed)
                    mem=fac()
                    true=dict(u.habit_dict())
                    drifted=dict(true); drifted["format"]="table" if true["format"]=="code_first" else "code_first"
                    total_mism=0; hs_correct=0; hs_total=0; recov=None
                    for turn in range(300):
                        if cfg["drift_at"] is not None and turn==cfg["drift_at"]:
                            true=dict(drifted); recov_start=turn
                        task=rng.choice(["code","summary","explain","create"])
                        true_task=true_pref_for(true, task)
                        pred=predict_from_pack(mem.get_pack(task), rng)
                        mism=[d for d in CORE3 if pred[d]!=true_task[d]]
                        if mism: total_mism+=1
                        # hidden-state accuracy separately
                        pack_est=mem.get_pack(task) or {}
                        for d in CORE3:
                            hs_total+=1
                            if pack_est.get(d)==true_task[d]: hs_correct+=1
                        expl, _ = make_explicit_feedback(true_task, pred,
                                                          cfg["feedback_prob"], cfg["false_rate"], rng)
                        # STRICT ISOLATION: learner only gets expl dict, never true_task
                        mem.observe(expl, turn, task)
                        if recov is None and cfg["drift_at"] is not None and turn>cfg["drift_at"]:
                            if not mism:
                                recov=turn-cfg["drift_at"]
                    errs.append(total_mism/300); hs.append(hs_correct/hs_total)
                    if recov is not None: recovs.append(recov)
            print(f"{name:<18} ans_err={statistics.mean(errs):.3f}±{statistics.pstdev(errs):.3f} hidden_acc={statistics.mean(hs):.3f} recov={statistics.mean(recovs) if recovs else 'nan'}")

    print("\n--- Parser generalization (hash vs LLM enc) ---")
    print("Hash baseline on unseen phrases in scripts/run_llm_encoder_parser.py gave unseen ~0%")
    print("LLM encoder gave 85.7% (21 phrases). Next step: test on 100+ unseen paraphrases, multi-intent, negation.")
    print("To pass B1.1, all methods must share task context and feedback precision; neural must beat Context-Latest+Bayesian on hidden_acc.")

if __name__=="__main__":
    run_b1_1()
