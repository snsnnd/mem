"""Long-run (1000 turns) + noise robustness to prove growth/degradation.

Scenarios:
 1) Long-run 1000 turns, static habits -> does memory blow up? prune keeps slots bounded.
 2) Task rotation 4-cycle code/summary/explain/create, with idle -> prune then regrow.
    We track time-to-recover after task returns.
 3) Noise: 10% random false corrections. Hybrid (MLP trust) should resist noise better than fixed gamma rule.

We compare: None / StaticRAG / RuleDynamic / Hybrid
"""
from __future__ import annotations
import random, statistics
from .users import generate_users, UserPersona
from .assistant import make_answer, correction_event
from .cortex import NoneMemory, StaticRAG, DynamicCortex
from .hybrid import HybridMemory
from .run_slot_detail import TaskPartitionedMemory

# ---------- 1) Long-run static ----------
def run_long_static(name, factory, n_turns=1000, n_users=10):
    users = generate_users(n=n_users, seed=7)
    rng = random.Random(42)
    all_corr = []
    final_slots = []
    for u in users:
        mem = factory()
        tot=0
        for turn in range(n_turns):
            task = rng.choice(["解释","代码","总结","例子","概念"])
            try:
                pack = mem.get_pack(task)
            except TypeError:
                pack = mem.get_pack()
            ans = make_answer(task, pack, rng)
            corr = correction_event(ans, u.habit_dict())
            if corr: tot+=1
            try:
                mem.observe(corr, u.habit_dict(), turn, rng, task)
            except TypeError:
                mem.observe(corr, u.habit_dict(), turn, rng)
        all_corr.append(tot/n_turns)
        if hasattr(mem, "stats"): final_slots.append(mem.stats().get("slots",0))
        elif hasattr(mem, "slots"): final_slots.append(len(mem.slots))
    print(f"{name:<28} corr={statistics.mean(all_corr):.3f} avg_final_slots={statistics.mean(final_slots) if final_slots else 0:.1f}")
    return statistics.mean(all_corr)

# ---------- 2) Task rotation with prune ----------
def run_rotation(name, factory):
    users = generate_users(n=10, seed=99)
    rng = random.Random(2024)
    # 4 phases of 100 turns each = 400 turns, rotating pairs
    phases = [["code","summary"], ["explain","create"], ["code","summary"], ["explain","create"]]
    TASK_TO_FORMAT = {"code":"code_first","summary":"table","explain":"prose","create":"prose"}
    PHASE_TASKS = {"code":"写代码并说明一下","summary":"总结今天内容","explain":"解释一下这段循环","create":"帮我写个创作并展开"}
    per_phase_err = [[] for _ in phases]
    for u in users:
        mem = factory()
        base = u.habit_dict()
        for pi, phase in enumerate(phases):
            tot=0
            for turn in range(100):
                tt = rng.choice(phase)
                task = PHASE_TASKS[tt]
                try: pack = mem.get_pack(task)
                except: pack = mem.get_pack()
                ans = make_answer(task, pack, rng)
                expected = {**base, "format": TASK_TO_FORMAT[tt]}
                if tt=="create": expected["verbosity"]="long"
                corr=[]
                if ans["verbosity"]!=expected["verbosity"]: corr.append("verbosity")
                if ans["emoji"]!=expected["emoji"]: corr.append("emoji")
                if ans["format"]!=expected["format"]: corr.append("format")
                if corr: tot+=1
                try: mem.observe(corr, expected, pi*100+turn, rng, task)
                except: mem.observe(corr, expected, pi*100+turn, rng)
            per_phase_err[pi].append(tot/100)
    print(f"{name:<28} phase errs: "+ " | ".join(f"{statistics.mean(x):.2f}" for x in per_phase_err))
    # Recovery gain: phase 2 (return) vs phase 0 (first seen) - lower means remembered / fast regrow
    return [statistics.mean(x) for x in per_phase_err]

# ---------- 3) Noise ----------
def run_noise(name, factory, noise_rate=0.10):
    users = generate_users(n=10, seed=5)
    rng = random.Random(777)
    cs=[]
    for u in users:
        mem=factory(); tot=0
        for turn in range(200):
            task=rng.choice(["解释","代码","总结"])
            pack=mem.get_pack(task) if "Partitioned" in name else mem.get_pack()
            ans=make_answer(task, pack, rng)
            corr=correction_event(ans, u.habit_dict())
            # inject noise: with prob noise_rate, flip a random dim correction (false)
            if rng.random()<noise_rate:
                # add spurious correction to a random dim not in corr
                dim=rng.choice(["verbosity","emoji","format"])
                if dim not in corr: corr=corr+[dim]
            if corr: tot+=1
            try: mem.observe(corr, u.habit_dict(), turn, rng, task)
            except: mem.observe(corr, u.habit_dict(), turn, rng)
        cs.append(tot/200)
    print(f"{name:<28} noise={noise_rate} corr={statistics.mean(cs):.3f} (higher = more noise-sensitive)")
    return statistics.mean(cs)

if __name__=="__main__":
    print("=== 1) Long-run static 1000 turns ===")
    run_long_static("None", lambda: NoneMemory())
    run_long_static("StaticRAG", lambda: StaticRAG())
    run_long_static("Rule DynamicCortex", lambda: DynamicCortex(seed=0))
    run_long_static("Hybrid+MLP", lambda: HybridMemory(seed=0))

    print("\n=== 2) Task rotation (4 tasks, 400 turns, prune test) ===")
    run_rotation("Rule single-value", lambda: DynamicCortex(seed=0))
    run_rotation("TaskPart WITH prune", lambda: TaskPartitionedMemory(use_pruning=True, gamma=0.90))
    run_rotation("TaskPart NO prune", lambda: TaskPartitionedMemory(use_pruning=False))

    print("\n=== 3) Noise 10% false corrections (200 turns) ===")
    run_noise("StaticRAG", lambda: StaticRAG(), 0.10)
    run_noise("Rule DynamicCortex", lambda: DynamicCortex(seed=0), 0.10)
    run_noise("Hybrid+MLP", lambda: HybridMemory(seed=0), 0.10)
    print("\nIf Hybrid < Rule < Static on noise, MLP trust is filtering noise, not just memorizing.")
