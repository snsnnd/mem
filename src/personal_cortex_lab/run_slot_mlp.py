from __future__ import annotations
import random, statistics
from .users import generate_users, TASKS
from .assistant import make_answer, correction_event
from .cortex import DynamicCortex, StaticRAG, NoneMemory
from .mlp_predictor import MLPHabitMemory
from .slot_mlp import SlotMLPMemory

def run_one(user, mem, rng, n_turns=80):
    total=0
    for turn in range(n_turns):
        task=rng.choice(TASKS)
        ans=make_answer(task, mem.get_pack(), rng)
        corr=correction_event(ans, user.habit_dict())
        if corr: total+=1
        mem.observe(corr, user.habit_dict(), turn, rng)
    pack=mem.get_pack() or {}
    recall=sum(1 for d,v in user.habit_dict().items() if pack.get(d)==v)/3
    return total/n_turns, recall

def cond(name, users, factory):
    rng=random.Random(1); cr=[]; rec=[]
    for u in users:
        c,r=run_one(u, factory(), rng); cr.append(c); rec.append(r)
    print(f"{name}: corr={statistics.mean(cr):.3f} recall={statistics.mean(rec):.3f}")

def drift(name, factory):
    users=generate_users(n=30, seed=42); rng=random.Random(1); errs=[]
    for u in users:
        mem=factory(); true=u.habit_dict(); e=0; t=0
        for turn in range(80):
            if turn==40:
                true={**true, "format": "table" if true["format"]=="code_first" else "code_first",
                              "verbosity": "long" if true["verbosity"]=="short" else "short"}
            task=rng.choice(TASKS)
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
    print("=== STATIC (30x80) ===")
    cond("A None", users, lambda: NoneMemory())
    cond("B Static", users, lambda: StaticRAG())
    cond("C RuleSlot", users, lambda: DynamicCortex(seed=0))
    cond("D GlobalMLP", users, lambda: MLPHabitMemory(seed=0))
    cond("E SlotMLP (partitioned regions)", users, lambda: SlotMLPMemory(seed=0))
    # show slot stats for one user
    mem=SlotMLPMemory(seed=0); rng=random.Random(1)
    run_one(users[1], mem, rng)
    print(f"Example SlotMLP stats: {mem.stats()} pack={mem.get_pack()} true={users[1].habit_dict()}")
    print("\n=== DRIFT (flip at 40) ===")
    drift("B Static", lambda: StaticRAG())
    drift("C RuleSlot", lambda: DynamicCortex(seed=0))
    drift("D GlobalMLP", lambda: MLPHabitMemory(seed=0))
    drift("E SlotMLP", lambda: SlotMLPMemory(seed=0))
