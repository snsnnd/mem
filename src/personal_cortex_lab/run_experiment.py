"""Experiment runner: A vs B vs C bulk simulation.

Uses rule-based user simulator for speed and reproducibility.
Subagent-based user simulation is provided as optional qualitative mode
(see subagent_users.py) - slower but closer to real LLM behaviour.
"""
from __future__ import annotations
import random, json, statistics
from dataclasses import dataclass, asdict
from .users import generate_users, TASKS
from .assistant import make_answer, correction_event
from .cortex import NoneMemory, StaticRAG, DynamicCortex

@dataclass
class TurnLog:
    turn: int
    task: str
    answer: dict
    corrections: list[str]
    true_habit: dict
    memory_pack: dict | None

def run_one_user(user, memory, rng, n_turns: int = 80):
    logs = []
    total_correction_turns = 0
    total_dim_mismatch = 0
    for turn in range(n_turns):
        task = rng.choice(TASKS)
        pack = memory.get_pack()
        ans = make_answer(task, pack, rng)
        corr = correction_event(ans, user.habit_dict())
        if corr:
            total_correction_turns += 1
            total_dim_mismatch += len(corr)
        memory.observe(corr, user.habit_dict(), turn, rng)
        logs.append(TurnLog(turn, task, ans, corr, user.habit_dict(), pack))
    return {
        "correction_rate": total_correction_turns / n_turns,
        "dim_mismatch_rate": total_dim_mismatch / (n_turns * 3),
        "logs": logs,
        "final_pack": memory.get_pack(),
        "stats": memory.stats() if hasattr(memory, "stats") else {},
        "history_len": len(getattr(memory, "history", [])),
    }

def run_condition(name: str, users, memory_factory, n_turns=80, seed=123):
    rng = random.Random(seed)
    results = []
    for u in users:
        mem = memory_factory()
        res = run_one_user(u, mem, rng, n_turns=n_turns)
        res["uid"] = u.uid
        res["true_habit"] = u.habit_dict()
        results.append(res)

    avg_corr = statistics.mean(r["correction_rate"] for r in results)
    avg_dim = statistics.mean(r["dim_mismatch_rate"] for r in results)

    # Long-term recall: final pack vs true habit
    recall_hits = 0
    recall_total = 0
    for r in results:
        pack = r["final_pack"] or {}
        for dim, true_v in r["true_habit"].items():
            recall_total += 1
            if pack.get(dim) == true_v:
                recall_hits += 1
    recall = recall_hits / max(1, recall_total)

    # Preference match estimate = 1 - dim mismatch
    pref_match = 1 - avg_dim

    stats_agg = {}
    if results and results[0].get("stats"):
        stats_agg = {
            "avg_slots": statistics.mean(r["stats"]["slots"] for r in results),
            "avg_growth": statistics.mean(r["stats"]["growth"] for r in results),
            "avg_prune": statistics.mean(r["stats"]["prune"] for r in results),
        }

    print(f"\n=== {name} ===")
    print(f"Users: {len(users)} | Turns/user: {n_turns} | Total turns: {len(users)*n_turns}")
    print(f"Correction rate (turn needs correction): {avg_corr:.3f}  (lower better)")
    print(f"Dim mismatch rate (per habit dim): {avg_dim:.3f}  (lower better)")
    print(f"Preference match (1 - mismatch): {pref_match:.3f}  (higher better)")
    print(f"Long-term recall (final pack == true habit): {recall:.3f}")
    if stats_agg:
        print(f"Slots avg: {stats_agg['avg_slots']:.1f} | Growth avg: {stats_agg['avg_growth']:.1f} | Prune avg: {stats_agg['avg_prune']:.1f}")
    # Show a few examples
    for r in results[:3]:
        print(f"  {r['uid']} true={r['true_habit']} final_pack={r['final_pack']} corr_rate={r['correction_rate']:.2f}")

    return {"name": name, "avg_corr": avg_corr, "avg_dim": avg_dim, "pref_match": pref_match, "recall": recall, "stats": stats_agg, "results": results}

def main():
    users = generate_users(n=30, seed=42)
    print(f"Generated {len(users)} users")
    print("Sample habits:")
    for u in users[:5]:
        print(" ", asdict(u))

    rng_factory = lambda: random.Random(999)

    results = []
    results.append(run_condition("A - No Memory", users,
                                 lambda: NoneMemory(), n_turns=80, seed=1))
    results.append(run_condition("B - Static RAG", users,
                                 lambda: StaticRAG(), n_turns=80, seed=1))
    results.append(run_condition("C - Dynamic Cortex (grow/degrade)", users,
                                 lambda: DynamicCortex(seed=0), n_turns=80, seed=1))

    # Summary table
    print("\n\n=== SUMMARY (lower corr/mismatch better, higher match/recall better) ===")
    print(f"{'Condition':<38} {'CorrRate':>8} {'Mismatch':>8} {'PrefMatch':>10} {'Recall':>8}")
    for r in results:
        print(f"{r['name']:<38} {r['avg_corr']:8.3f} {r['avg_dim']:8.3f} {r['pref_match']:10.3f} {r['recall']:8.3f}")

    # Verdict
    a, b, c = results
    print("\n--- Verdict Criteria ---")
    print("Success if C corr < B corr * 0.8 and C recall > 0.8 and C pref_match > B pref_match")
    success = (c["avg_corr"] < b["avg_corr"] * 0.8) and (c["recall"] > 0.8) and (c["pref_match"] > b["pref_match"])
    print(f"Dynamic Cortex wins: {success}")
    if c["avg_corr"] < b["avg_corr"]:
        print(f"Correction reduction vs Static: {(b['avg_corr']-c['avg_corr'])/b['avg_corr']*100:.1f}%")
    if c["recall"] > b["recall"]:
        print(f"Recall gain vs Static: {(c['recall']-b['recall'])*100:.1f}pts")

    # Save summary
    summary = {r["name"]: {k: v for k, v in r.items() if k != "results"} for r in results}
    with open("results_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("\nSaved results_summary.json")

if __name__ == "__main__":
    main()
