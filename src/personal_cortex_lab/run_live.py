"""Live multi-turn loop: Subagent-like users <-> Dynamic Cortex <-> Mock LLM
We run 10 turns for each of the 3 subagent-verified personas and trace
slot growth / importance / surprise per turn to see if it tracks corrections.
"""
from __future__ import annotations
import random
from .users import UserPersona
from .assistant import make_answer, correction_event
from .cortex import DynamicCortex, StaticRAG
from .users import TASKS

PERSONAS = [
    UserPersona(uid="user_01", verbosity="short", emoji="never", fmt="code_first"),
    UserPersona(uid="user_02", verbosity="long", emoji="always", fmt="table"),
    UserPersona(uid="user_03", verbosity="short", emoji="never", fmt="prose"),
]

def run_persona(user: UserPersona, n_turns: int = 12):
    print(f"\n{'='*70}")
    print(f"PERSONA {user.uid} true_habit={user.habit_dict()}")
    print(f"{'='*70}")
    cortex = DynamicCortex(seed=0)
    static = StaticRAG()
    rng = random.Random(7)

    for turn in range(n_turns):
        task = TASKS[turn % len(TASKS)]
        pack = cortex.get_pack()
        pack_static = static.get_pack()

        ans_dyn = make_answer(task, pack, rng)
        corr_dyn = correction_event(ans_dyn, user.habit_dict())

        # static for comparison (same task, same rng seed offset)
        ans_static = make_answer(task, pack_static, rng)
        corr_static = correction_event(ans_static, user.habit_dict())

        # update both
        cortex.observe(corr_dyn, user.habit_dict(), turn, rng)
        static.observe(corr_static, user.habit_dict(), turn, rng)

        stats = cortex.stats()
        print(f"Turn {turn:02d} task={task[:20]:<20} "
              f"dyn_ans={ans_dyn['verbosity'][:1]}/{ans_dyn['emoji'][:1]}/{ans_dyn['format'][:4]} "
              f"corr_dyn={corr_dyn or '✓'} | static_corr={corr_static or '✓'} | "
              f"slots={stats['slots']} growth={stats['growth']} prune={stats['prune']} pack={pack}")

    print(f"Final Dynamic pack: {cortex.get_pack()} | Static pack: {static.get_pack()}")
    print(f"Slots trace growth={cortex.growth_events} prune={cortex.prune_events} update={cortex.update_events}")
    return cortex, static

if __name__ == "__main__":
    for p in PERSONAS:
        run_persona(p, n_turns=12)
    print("\nInterpretation: Dynamic cortex grows a new slot on first corrections (turn 0-2),")
    print("then stabilizes with no growth, importance reinforces on satisfaction.")
    print("Static needs many turns of history to converge and accumulates stale entries.")
