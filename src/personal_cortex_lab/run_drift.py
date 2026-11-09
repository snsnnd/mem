"""Drift test: habits flip at turn 40. Static RAG will be slow to forget old habit,
Dynamic Cortex should decay old slot and grow/update faster.
"""
from __future__ import annotations
import random, statistics
from .users import generate_users, TASKS
from .assistant import make_answer, correction_event
from .cortex import NoneMemory, StaticRAG, DynamicCortex

def run_user_with_drift(user, memory, rng, n_turns=80):
    true = user.habit_dict()
    total_corr = 0
    errors_after_drift = 0
    turns_after_drift = 0
    # At turn 40, flip format and verbosity for 50% users (drift)
    drifted = {}
    for turn in range(n_turns):
        if turn == 40:
            # Flip: code_first <-> table, short <-> long
            if user.fmt == "code_first":
                drifted["format"] = "table"
            elif user.fmt == "table":
                drifted["format"] = "code_first"
            drifted["verbosity"] = "long" if true["verbosity"] == "short" else "short"
            true = {**true, **drifted}
        task = rng.choice(TASKS)
        pack = memory.get_pack()
        ans = make_answer(task, pack, rng)
        corr = correction_event(ans, true)
        if corr:
            total_corr += 1
        if turn >= 40:
            turns_after_drift += 1
            if corr:
                errors_after_drift += 1
        memory.observe(corr, true, turn, rng)
    # Adapt speed: how many turns after drift until correction rate drops back
    return {
        "corr_rate": total_corr / n_turns,
        "drift_error_rate": errors_after_drift / max(1, turns_after_drift),
        "final_pack": memory.get_pack(),
        "true_after": true,
    }

def run(name, factory, seed=1):
    users = generate_users(n=30, seed=42)
    rng = random.Random(seed)
    res = []
    for u in users:
        m = factory()
        r = run_user_with_drift(u, m, rng)
        r["uid"] = u.uid
        res.append(r)
    avg_corr = statistics.mean(r["corr_rate"] for r in res)
    avg_drift_err = statistics.mean(r["drift_error_rate"] for r in res)
    recall_after = statistics.mean(
        sum(1 for d,v in r["true_after"].items() if (r["final_pack"] or {}).get(d) == v) / 3
        for r in res
    )
    print(f"{name}: corr={avg_corr:.3f} drift_err_after_40={avg_drift_err:.3f} recall_after_drift={recall_after:.3f}")
    return avg_corr, avg_drift_err, recall_after

if __name__ == "__main__":
    print("=== DRIFT TEST (habit flips at turn 40) ===")
    print("This separates Static (slow to forget) vs Dynamic (decays old + grows new)")
    run("A NoMemory", lambda: NoneMemory())
    run("B StaticRAG", lambda: StaticRAG())
    run("C DynamicCortex", lambda: DynamicCortex(seed=0))
    print("\nExpectation: Dynamic should adapt faster (lower drift_err) because decay+prune releases old habit")
