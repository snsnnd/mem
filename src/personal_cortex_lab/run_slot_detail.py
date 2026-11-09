"""Detailed slot partitioning by task with decay & pruning.

4 tasks: code / summary / explain / create
Each task has its own habit pattern (format expected):
  code -> code_first, summary -> table, explain -> prose, create -> prose + long

Experiment phases:
  0-30 turns:  code + summary only
  30-60:        explain + create only (code/summary idle -> should decay/prune)
  60-100:       code + summary return (test prune-regrow vs no-prune)

We compare:
  B Single memory (one global Hebbian)
  H Slot-routed by task WITH decay/pruning
  H Slot-routed by task WITHOUT pruning (no decay, max slots never prune)

Metrics logged per turn: slot count, per-task error, prune/growth events.
"""
from __future__ import annotations
import random, statistics, json
import numpy as np
from .users import generate_users
from .assistant import make_answer

TASK_TO_FORMAT = {"code":"code_first","summary":"table","explain":"prose","create":"prose"}
TASKS_4 = ["写代码并说明一下","总结今天内容","解释一下这段循环","帮我写个创作并展开"]

def task_type(task: str) -> str:
    if "代码" in task: return "code"
    if "总结" in task: return "summary"
    if "解释" in task: return "explain"
    if "创作" in task: return "create"
    return "explain"

# --- slot by task, Hebbian per slot ---
class TaskPartitionedMemory:
    """Each task = one slot region, each slot is Hebbian W (explicit fast value + Hebbian slow weight).
    For this detail test we use explicit value per slot per dim (fast jump, like Rule) to isolate routing effect,
    PLUS Hebbian decay per slot for pruning.
    If use_pruning=False, slots never decay (share B's no-forget flaw).
    """
    name = "partitioned"
    def __init__(self, use_pruning=True, max_slots=4, gamma=0.96, prune_thresh=0.2, seed=0):
        self.use_pruning=use_pruning
        self.gamma=gamma; self.thresh=prune_thresh
        self.slots: dict[str, dict] = {}  # task -> {value: dict, importance: float, last_used}
        self.turn=0; self.growth=0; self.prune=0
        self.rng=np.random.default_rng(seed)

    def _tt(self, task): return task_type(task) if task else "explain"

    def get_pack(self, task=None):
        tt=self._tt(task)
        if tt in self.slots:
            return dict(self.slots[tt]["value"])
        return None  # no slot for this task yet -> blind guess

    def observe(self, corrections, true_ctx, turn, rng, task=None):
        self.turn=turn
        tt=self._tt(task)
        # decay all slots per turn if pruning enabled
        if self.use_pruning:
            for k in list(self.slots.keys()):
                self.slots[k]["importance"] *= self.gamma
        # growth: first correction for unseen task creates slot
        if tt not in self.slots and corrections:
            self.slots[tt]={"value":dict(true_ctx), "importance":1.0, "last_used":turn, "count":len(corrections)}
            self.growth+=1
        elif tt in self.slots:
            # explicit fast update
            for dim in corrections:
                self.slots[tt]["value"][dim]=true_ctx[dim]
            self.slots[tt]["importance"]=min(2.0, self.slots[tt]["importance"]+0.3)
            self.slots[tt]["last_used"]=turn
        # positive reinforcement
        if not corrections and tt in self.slots:
            self.slots[tt]["importance"]=min(2.0, self.slots[tt]["importance"]+0.02)
        # pruning: idle + low importance
        if self.use_pruning:
            for k in list(self.slots.keys()):
                if k!=tt and self.slots[k]["importance"]<self.thresh and (turn-self.slots[k]["last_used"])>15:
                    del self.slots[k]; self.prune+=1

    def stats(self):
        return {"slots":len(self.slots),"growth":self.growth,"prune":self.prune,
                "importances":{k:round(v["importance"],2) for k,v in self.slots.items()}}

# Single global memory (for comparison) - reuse DynamicCortex
from .cortex import DynamicCortex
from .hebbian import HebbianMemory

def run_detailed(name, factory, use_task=True):
    users=generate_users(n=30, seed=42)
    rng=random.Random(123)
    # per-task error tracked
    per_task_err={"code":0,"summary":0,"explain":0,"create":0}
    per_task_n={"code":0,"summary":0,"explain":0,"create":0}
    slot_counts_over_time=[]
    total_growth=0; total_prune=0
    # Track a single exemplar user for trace
    exemplar_mem=factory()
    exemplar_trace=[]
    for uid_idx, u in enumerate(users):
        mem=factory() if uid_idx!=1 else exemplar_mem
        base=u.habit_dict()
        for turn in range(100):
            if 0 <= turn < 30: pool=["code","summary"]
            elif 30 <= turn < 60: pool=["explain","create"]
            else: pool=["code","summary"]
            tt=rng.choice(pool)
            task={"code":"写代码并说明一下","summary":"总结今天内容","explain":"解释一下这段循环","create":"帮我写个创作并展开"}[tt]
            # pack depends on use_task or not; Rule/partitioned support task, Hebbian single doesn't use task for pack selection except routed
            try:
                pack=mem.get_pack(task if use_task else None)
            except TypeError:
                pack=mem.get_pack()
            ans=make_answer(task, pack, rng)
            # expected format for this task
            expected={**base, "format": TASK_TO_FORMAT[tt]}
            # verbosity for create expects long
            if tt=="create": expected["verbosity"]="long"
            else: expected["verbosity"]=base["verbosity"]
            corr=[]
            if ans["verbosity"]!=expected["verbosity"]: corr.append("verbosity")
            if ans["emoji"]!=expected["emoji"]: corr.append("emoji")
            if ans["format"]!=expected["format"]: corr.append("format")
            per_task_n[tt]+=1
            if corr: per_task_err[tt]+=1
            # observe with true ctx
            try:
                mem.observe(corr, expected, turn, rng, task)
            except TypeError:
                mem.observe(corr, expected, turn, rng)
            if uid_idx==1:
                st=mem.stats() if hasattr(mem,"stats") else {}
                exemplar_trace.append((turn,tt,corr,st))
                slot_counts_over_time.append(st.get("slots",0) if isinstance(st,dict) else 0)
        if hasattr(mem,"growth"): total_growth+=getattr(mem,"growth",0)
        if hasattr(mem,"growth"): total_growth=getattr(mem,"growth",0)  # approximate per last user; keep simple
    print(f"\n{name}: per-task error = "+ ", ".join(f"{k}:{per_task_err[k]/max(1,per_task_n[k]):.2f}" for k in per_task_err))
    print(f"  exemplar slot count over time (turn: slots):")
    # sample 10 points
    for i in range(0, len(exemplar_trace), 10):
        t,tt,corr,st=exemplar_trace[i]
        print(f"    turn {t:02d} task={tt:8s} slots={st.get('slots','?')} imp={st.get('importances',{})} growth={st.get('growth')} prune={st.get('prune')}")
    st=exemplar_mem.stats() if hasattr(exemplar_mem,"stats") else {}
    print(f"  final stats exemplar: {st}")
    # summary
    overall=sum(per_task_err.values())/sum(per_task_n.values())
    print(f"  overall corr {overall:.3f}")
    return overall

if __name__=="__main__":
    print("=== DETAILED: 4 task partition, temporal shift ===")
    print("Phase code/summary -> explain/create -> code/summary return")
    run_detailed("B Rule single-value (no partition)", lambda: DynamicCortex(seed=0), use_task=False)
    run_detailed("H Task-partitioned WITH prune", lambda: TaskPartitionedMemory(use_pruning=True), use_task=True)
    run_detailed("H Task-partitioned NO prune", lambda: TaskPartitionedMemory(use_pruning=False), use_task=True)
