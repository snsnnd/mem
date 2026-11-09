# Personal Cortex Lab

让大模型本地维护一个**可动态生长/退化的小网络**，记录并学习使用者的习惯与长期记忆，每次调用前把小网络的输出附加到提示词（或权重）里，让大模型拥有属于个人的长期记忆。

> 一句话：**大模型是大脑通用智能，小网络是你的贴身管家。**

---

## 为什么要做这个

现在大家的记忆系统（RAG、MemGPT/Letta、Hindsight、PlugMem、LongMem）基本都在“存文本、检索文本”，或者“LLM 自己总结自己”。但个人习惯不是一条文本，它是：

* 反复出现、需要预测的规律（你喜欢先给代码、讨厌 emoji、喜欢表格）
* 会漂移的（今年喜欢表格，明年喜欢散文）
* 需要本地私密、低成本的

静态 RAG 实验里就已经看到了：

| 条件 | 纠正率 | 长期召回 |
| :--- | :--- | :--- |
| A-无记忆 | 0.92 | 0% |
| B-静态 RAG | 0.40 | 100% |
| C-动态皮层 | 0.40 | 100% |

但漂移测试一来，动态皮层把适应速度从 0.49 拉到 0.42（14% 提升）。也就是，这个问题需要**预测 + 生长 + 退化**，而不是检索。

---

## 核心设计

```
用户输入
   ↓
[感知层] 编码当前对话（hash向量 / BGE 等，MVP 用轻量hash代替重模型）
   ↓
[个人皮层 Personal Cortex] 动态生长/退化的记忆核心（<1M 参数，本地运行）
   ├─→ 输出 A: 显性记忆 (习惯向量 -> 自然语言包，拼到 Prompt)
   ├─→ 输出 B: 隐性习惯向量 / LoRA 增量 ΔW (表达你的风格)
   └─→ 输出 C: 惊讶度/新颖度 (surprise，触发生长)
   ↓
[大模型] 接收 Prompt + 记忆包（或 LoRA 合并）
   ↓
生成回答 → 收集 Implicit Feedback → 反哺小网络
```

### 双系统

* **快系统（海马体）**：显式槽位，惊讶大时跳变 `slot.value = 真值`，一次就学会。快，但乱。
* **慢系统（皮层）**：每个槽位内的小 MLP 或赫布区，学重要性、路由、置信度，慢慢把学习固化。慢，但稳。

`显式值负责快速遗忘跳变，MLP 只管慢管理`，这是我们在混合版验证过的最高效分工。

### 生长 / 退化规则

```
surprise = | 用户实际反馈 - 小网络预测习惯 |

if surprise > 阈值:
    开新槽位 / 更新已有值 (neurogenesis)
else:
    importance += 0.05

importance(t+1) = γ * importance(t) + utiilty(t)
if importance < prune_thresh and idle > 20:
    剪枝 (degradation)
```

方向和人脑皮层-海马体系统一致：有新体验就长新神经元，没用就弱化和剪枝。

### 为什么不是梯度？

我们试了三条：

1. **全局 MLP（梯度）**：静态能追到规则版（0.407 vs 0.402），漂移时 0.48-0.56，输给规则。结论是梯度跳变太慢，稳定性-可塑性冲突。
2. **槽位 MLP（分区梯度）**：静态 0.406，漂移 0.52 更输。分区隔离了干扰，但跳变还是慢。
3. **赫布学习 / 三因子规则**：`ΔW = η * surprise * x ⊗ error`，纯局部规则，不碰反向传播。静态 0.403，漂移快版 0.407，慢版 0.432——局部规则完全能学，漂移下和规则版打平。
4. **混合版（显式值 + MLP 管理）**：静态 0.402，漂移 0.417，等于规则版，还多了个可学习的剪枝管理层，综合最优。

结论：小网络值得用局部规则 + 显式跳变，不必走重梯度。梯度留给最后一步的 Hypernetwork 生成权重就行。

---

## 实验目录

```
src/personal_cortex_lab/
  users.py              # 假用户习惯向量生成
  assistant.py          # Mock LLM：按 pack 定风格
  cortex.py             # A 无记忆 / B 静态RAG / C 动态皮层（槽位+衰减）
  run_experiment.py     # 静态习惯 30x80
  run_drift.py          # 40 轮习惯翻转
  run_live.py           # 12 轮真多轮追踪
  mlp_predictor.py      # 全局 MLP
  run_mlp.py            # MLP 对照
  slot_mlp.py           # 槽位分区 MLP
  run_slot_mlp.py       # 槽位MLP对照
  hebbian.py            # 赫布局部规则
  run_hebbian.py        # 赫布对照
  hebbian_slot.py       # 赫布+槽位路由
  run_hebbian_slot.py   # 路由对照（含无任务键 vs 有任务键）
  hybrid.py             # 显式值+MLP管理混合版
  run_hybrid.py         # 混合版对照
  text_encoder.py       # 轻量hash编码器（冻结翻译官代理）
  run_text_encoder.py   # 真人话子代理 utterances -> 槽位
  run_multiturn.py      # 3 真人设 10 轮真多轮对话
  hypernetwork.py       # Hyper LoRA 生成器
  run_hypernetwork.py   # Prompt vs ΔW
  subagent_users.py     # 子代理当用户的模板

scripts/
  run_qwen_hypernet.py   # 真 Qwen2.5-0.5B FP16 + Hypernet LoRA (4060 MPS)
  train_hypernet.py      # 真 Qwen 增量训练：单独 LoRA 蒸馏 -> Hypernet 学映射
  test_zero_shot.py      # 零样本组合习惯测试

docs/
  ARCHITECTURE.md       # 详细架构和背景调研
  EXPERIMENTS.md        # 每一步的原理、思考和具体数字
results/
  summary/*.json        # 每次跑的汇总

results_summary.json
```

---

## 已经跑完的关键实验

### 1. 静态习惯（30人x80轮 = 2400轮）

```
A-无记忆     corr 0.920  recall 0%
B-静态RAG    corr 0.402  recall 100%
C-动态皮层   corr 0.402  recall 100%
D-全局MLP    corr 0.407  recall 100%
E-槽位MLP    corr 0.405  recall 100%
F-Hybrid     corr 0.402  recall 100%
G-Hebbian    corr 0.403  recall 100%
```

结论：静态场景全同，差距在漂移。

### 2. 习惯漂移（40轮翻转）

```
B 静态        drift_err 0.490
C 规则        drift_err 0.417
D 全局MLP     drift_err 0.476
E 槽位MLP     drift_err 0.526
F Hybrid      drift_err 0.417
G Hebbian(快) drift_err 0.407
```

结论：梯度MLP最差，赫布快版能追到规则版。跳变快 > 梯度慢。

### 3. 按任务分片（4任务时间轮换）

4 任务：`代码/总结/解释/创作`，60-100 回到 code/summary。

```
B 单值槽      overall corr 0.636
H 分片带剪枝  overall corr 0.406
```

分片直接降纠正率 22 个点，闲置任务重要性从 2.0 衰到 0.5，空闲足够就会剪并在任务回归时再生。

### 4. 真人话（子代理）

3 个子代理人格，编入哈希向量后依然 2-3 轮学到，槽位涨 2 个就稳了：

* user_01（代码优先/短/无emoji）-> 包 100% 对
* user_02（表格/长/有emoji）-> 包 100% 对
* user_03（散文/短/无emoji）-> 包 100% 对

多轮 10 轮追踪也稳在最后包正确。

### 5. Prompt vs ΔW（Hypernet）

```
Prompt pack  corr 0.404 token ~+30
Hypernet ΔW  corr 0.182 token 0 (merged)
```

**Hypernet 不仅更快，而且从 `habit_vec -> LoRA ΔW` 直接跳到权重空间**，完成了“习惯向量，下一跳直接是权重”这层验证。

### 6. 真 Qwen2.5-0.5B（4060测）

* FP16，1GB 模型，RTX 4060 8GB 显存够余 7G。
* 零 LoRA Base vs 三种注入对比。
* 三习惯 LoRA 蒸馏：`code_first: L2 2.40 / table: 2.38 / prose: 2.38`
* Hypernet 训练：`MSE 0.0085 -> 0.0000`
* 零样本组合 `long+emoji+code_first` 生成了代码，但没 emoji——因为训练数据里没见过 emoji，这是预料内的小数据问题。

---

## 如何运行

### Mock 批量（无 GPU）

```bash
uv run python -m personal_cortex_lab.run_experiment   # 静态
uv run python -m personal_cortex_lab.run_drift        # 漂移
uv run python -m personal_cortex_lab.run_hebbian      # 赫布
uv run python -m personal_cortex_lab.run_slot_mlp     # 槽位MLP
uv run python -m personal_cortex_lab.run_hybrid       # 混合
uv run python -m personal_cortex_lab.run_hebbian_slot # 路由
uv run python -m personal_cortex_lab.run_slot_detail  # 4任务分片
uv run python -m personal_cortex_lab.run_text_encoder # 真人话子代理
uv run python -m personal_cortex_lab.run_multiturn    # 10轮真多轮
```

### 真模型（需 RTX 4060）

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers peft accelerate
python scripts/run_qwen_hypernet.py   # 对比三种注入
python scripts/train_hypernet.py      # 训 LoRA + Hypernet
python scripts/test_zero_shot.py      # 零样本组合测试
```

---

## 设计哲学推导

网络本质是预测器（Predictive Coding / Friston）。我们这个小网络就是一个以预测“你下一次想让我怎么答”为目标的预测器，惊讶度 = 预测误差。

* 快系统负责惊讶大时立即跳变（类似海马体一次成型）
* 慢系统负责慢学置信度和重要性（类似皮层固化）
* 分片用感受器把冲突文脉隔离
* 最后 Hypernetwork 把习惯向量直接编译成权重，避免提示词无限变长

这就是 Hypernetwork 个人化（Profile-to-PEFT，2025.10；SHINE，2026）已经证明的方向，和我们的生长/退化槽位在机制上互相补齐。

---

## TODO

* [x] 槽位+衰减+剪枝
* [x] MLP / 赫布 / 路由三套对照
* [x] 真人话子代理多轮
* [x] Prompt vs ΔW
* [ ] 真 Qwen2.5-0.5B 完整训练（EMOJI/long 的 LoRA 数据已蒸馏出来了，但零样本组合的训练任务里没有 emoji，需要补一轮 emoji 习惯数据）
* [ ] BGE / Qwen3-Emb-4B 替换 hash 代理编码器
* [ ] Hypernet 升到多层生成（现在只最后层 22528 维）
* [ ] App 化：本地 FastAPI + SQLite vec

---

## 版权 / 协议

教育实验代码，Apache 2.0。模型权重来自 Qwen2.5-0.5B-Instruct（Apache 2.0）。
