from __future__ import annotations
import random, statistics
from .users import generate_users
from .assistant import make_answer, correction_event
from .cortex import DynamicCortex, StaticRAG
from .mlp_predictor import MLPHabitMemory
from .slot_mlp import SlotMLPMemory
from .hybrid import HybridMemory

def run_one(user, mem, rng, n_turns=80):
    tot=0
    for turn in range(n_turns):
        task=rng.choice(["解释","代码","总结","例子","概念"])
        ans=make_answer(task, mem.get_pack(), rng)
        corr=correction_event(ans, user.habit_dict())
        if corr: tot+=1
        mem.observe(corr, user.habit_dict(), turn, rng)
    pack=mem.get_pack() or {}
    rec=sum(1 for d,v in user.habit_dict().items() if pack.get(d)==v)/3
    return tot/n_turns, rec

def cond(name, users, factory):
    rng=random.Random(1); cs=[]; rs=[]
    for u in users:
        c,r=run_one(u, factory(), rng); cs.append(c); rs.append(r)
    print(f"{name}: corr={statistics.mean(cs):.3f} recall={statistics.mean(rs):.3f}")

def drift(name, factory):
    users=generate_users(n=30, seed=42); rng=random.Random(1); errs=[]
    for u in users:
        mem=factory(); true=u.habit_dict(); e=0; t=0
        for turn in range(80):
            if turn==40:
                true={**true, "format": "table" if true["format"]=="code_first" else "code_first",
                              "verbosity": "long" if true["verbosity"]=="short" else "short"}
            task=rng.choice(["解释","代码","总结","例子","概念"])
            ans=make_answer(task, mem.get_pack(), rng)
            corr=correction_event(ans, true)
            if turn>=40:
                t+=1
                if corr: e+=1
            mem.observe(corr, true, turn, rng)
        errs.append(e/t)
    print(f"{name} drift_err={statistics.mean(errs):.3f}")

if __name__=="__main__":
    users=generate_users(n=30, seed=42)
    print("=== STATIC 30x80 ===")
    cond("C Rule", users, lambda: DynamicCortex(seed=0))
    cond("D GlobalMLP", users, lambda: MLPHabitMemory(seed=0))
    cond("E SlotMLP", users, lambda: SlotMLPMemory(seed=0))
    cond("F Hybrid (explicit+MLP manager)", users, lambda: HybridMemory(seed=0))
    mem=HybridMemory(seed=0); rng=random.Random(1)
    run_one(users[1], mem, rng)
    print(f"Hybrid example stats {mem.stats()} pack={mem.get_pack()}")
    print("\n=== DRIFT flip at 40 ===")
    drift("C Rule", lambda: DynamicCortex(seed=0))
    drift("D GlobalMLP", lambda: MLPHabitMemory(seed=0))
    drift("E SlotMLP", lambda: SlotMLPMemory(seed=0))
    drift("F Hybrid", lambda: HybridMemory(seed=0))
    print("\nHypothesis: F should match C on static (explicit flash) and be >= C on drift")
    print("because MLP manager learns per-dim volatility, prune faster when needed.")
