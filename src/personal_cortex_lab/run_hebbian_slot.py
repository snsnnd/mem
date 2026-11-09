from __future__ import annotations
import random, statistics
from .users import generate_users, UserPersona
from .assistant import make_answer, correction_event
from .cortex import DynamicCortex
from .hebbian import HebbianMemory
from .hebbian_slot import HebbianSlotRouted
from .mlp_predictor import MLPHabitMemory

# --- context-dependent habits: format depends on task type ---
# task contains "代码" -> code_first, "总结" -> table, else prose
def task_type(task: str) -> str:
    if "代码" in task: return "code_first"
    if "总结" in task: return "table"
    return "prose"

def make_answer_ctx(task, pack, rng):
    # pack may have format fixed, but answer's format is chosen by mock based on pack or random
    # so correction depends on task type
    ans = make_answer(task, pack, rng)
    return ans

def correction_ctx(ans, user_true_ctx: dict, task):
    expected_fmt = task_type(task)
    corr=[]
    if ans["verbosity"] != user_true_ctx["verbosity"]: corr.append("verbosity")
    if ans["emoji"] != user_true_ctx["emoji"]: corr.append("emoji")
    if ans["format"] != expected_fmt: corr.append("format")
    return corr

def run_ctx_one(user, mem, rng, n_turns=80, use_task=False):
    base = user.habit_dict(); tot=0
    for turn in range(n_turns):
        task = rng.choice(["写代码并说明一下","总结今天内容","解释一下这段循环","用方式讲一遍这道题","再讲一遍知识点","给我一个例子","把刚才解释换个格式"])
        if use_task:
            try:
                pack = mem.get_pack(task)
            except TypeError:
                pack = mem.get_pack()
        else:
            pack = mem.get_pack()
        # for ctx, pack's format is treated as generic; mock answer uses it
        ans = make_answer(task, pack, rng)
        corr = correction_ctx(ans, base, task)
        if corr: tot+=1
        # True value for update: format is contextual
        true_ctx = {**base, "format": task_type(task)}
        if use_task:
            try:
                mem.observe(corr, true_ctx, turn, rng, task)
            except TypeError:
                mem.observe(corr, true_ctx, turn, rng)
        else:
            mem.observe(corr, true_ctx, turn, rng)
    return tot/n_turns

def cond_ctx(name, users, factory, use_task=False):
    rng=random.Random(99); cs=[]
    for u in users:
        cs.append(run_ctx_one(u, factory(), rng, use_task=use_task))
    print(f"CTX {name}: corr={statistics.mean(cs):.3f}")

def run_standard(name, factory):
    users=generate_users(n=30, seed=42); rng=random.Random(1); cs=[]
    for u in users:
        mem=factory(); tot=0
        for turn in range(80):
            task=rng.choice(["解释","代码","总结"])
            ans=make_answer(task, mem.get_pack(), rng)
            corr=correction_event(ans, u.habit_dict())
            if corr: tot+=1
            mem.observe(corr, u.habit_dict(), turn, rng)
        cs.append(tot/80)
    print(f"STD {name}: corr={statistics.mean(cs):.3f}")

if __name__=="__main__":
    print("=== STANDARD static habits (任务无关习惯) ===")
    run_standard("C Rule", lambda: DynamicCortex(seed=0))
    run_standard("G Hebbian single", lambda: HebbianMemory(eta=0.8, decay=0.98))
    run_standard("H Hebbian+SlotRouting", lambda: HebbianSlotRouted(eta=0.6, decay=0.99))
    print("\n=== CONTEXT-DEPENDENT habits (格式随任务变) ===")
    users=generate_users(n=30, seed=42)
    cond_ctx("C Rule single-value", users, lambda: DynamicCortex(seed=0))
    cond_ctx("G Hebbian single", users, lambda: HebbianMemory(eta=0.8, decay=0.98))
    # For HebbianSlotRouted, routing should separate contexts into different slots
    cond_ctx("H Hebbian+SlotRouting (no task)", users, lambda: HebbianSlotRouted(eta=0.6, decay=0.99), use_task=False)
    cond_ctx("H+task routing (with task)", users, lambda: HebbianSlotRouted(eta=0.6, decay=0.99), use_task=True)
    mem=HebbianSlotRouted(eta=0.6, decay=0.99); rng=random.Random(99)
    run_ctx_one(users[1], mem, rng)
    print(f"Example Routed stats: {mem.stats()}")
