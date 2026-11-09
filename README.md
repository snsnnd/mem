# Personal Cortex Lab

> **研究状态（2026-11-09）**：下文保留早期探索记录，其中“优于梯度”“Hypernet 更优”等表述尚未经过公平、充分验证。当前已确认的价值主要是上下文隔离与证据累积，**尚未证明可训练神经记忆超越强基线**。新的[文献调研、代码审计与 B1.3 协议](docs/RESEARCH_REVIEW_B1_3.md)将共同轨迹回放和学习式控制器验证分开推进；[10 种子 pilot 结果](docs/B1_3_PILOT_RESULTS.md)已记录，但不等于已完成神经控制器实验。

### 可训练控制器实验入口

真正可训练的共享 MLP/GRU 写入与遗忘门控已加入，采用独立的训练、验证与测试阶段。设计、容量差异和监督边界见 [B1.3 控制器协议](docs/B1_3_CONTROLLER_PROTOCOL.md)。训练成功不等于优于基线。

**首轮留出测试已完成**：[完整结果](docs/B1_3_CONTROLLER_RESULTS.md)。ID 宏平均状态准确率：调优 HazardFilter **94.79%**、MLP **94.66%**、GRU **94.57%**。主指标未胜出；GRU 在部分变化场景的局部优势伴随噪声鲁棒性的代价。

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m personal_cortex_lab.train_memory_controller fit --output results/b1_3_controller_fit
.venv/bin/python -m personal_cortex_lab.train_memory_controller test --fit results/b1_3_controller_fit --output results/b1_3_controller_test
```

输出目录必须不存在；大体积轨迹、预测和权重本地保留，摘要与实验选择记录可纳入版本管理。

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
  run_fair_benchmark.py # 公平对照：去 oracle、去污染、加 Latest-Value
  run_b1.py             # B1 Causal Online Preference Learning
  run_longrun_noise.py  # 1000 轮长跑 + 噪音
  run_vague_predictor.py# Transformer 预测纠正类型
  run_pred_as_signal.py # 预测器作为反馈信号
  run_value_parser.py   # 维度+值解析器
  run_llm_encoder_parser.py # Qwen隐状态编码器 vs 哈希 (in scripts/)
  subagent_users.py     # 子代理当用户的模板

scripts/
  run_qwen_hypernet.py   # 真 Qwen2.5-0.5B FP16 + Hypernet LoRA (4060 MPS)
  train_hypernet.py      # 真 Qwen 增量训练：单独 LoRA 蒸馏 -> Hypernet 学映射
  test_zero_shot.py      # 零样本组合习惯测试
  run_llm_encoder_parser.py # LLM-as-encoder (Qwen hidden 896维) vs 哈希

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

### 6. B1 公平基准（去 oracle / 去污染 / 加强基线）

```
dense explicit, no drift:
  NoMemory 0.917 | Latest 0.748 | Rule 0.748 | Bayesian 0.734 | ContextKV 0.397

sparse 30%, noise10%, abrupt drift:
  Latest 0.790 recovery 51.0 | Rule 0.790 | Bayesian 0.761 recov 86.0 | ContextKV 0.441 recov 33.9

sparse noise10% gradual:
  Latest 0.778 | ContextKV 0.449 | Bayesian 0.750

sparse noise10% aba:
  Latest 0.784 | ContextKV 0.450 | Bayesian 0.751

vague, noise10%, abrupt:
  All 0.911 (全部挂) -> 最新问题
```

结论：

* 明确反馈静态时 Latest-Value 已经够，复杂网络没独立优势
* 漂移下 ContextKV 分片赢 0.35，是目前真正的杠杆
* Bayesian 比 Latest 好 0.03 err 但恢复慢 35 轮：稳定性-可塑性权衡真实

### 7. 真 Qwen2.5-0.5B（4060测）

* FP16，1GB 模型，RTX 4060 8GB 显存够余 7G。
* 零 LoRA Base vs 三种注入对比。
* 三习惯 LoRA 蒸馏：`code_first: L2 2.40 / table: 2.38 / prose: 2.38`
* Hypernet 训练：`MSE 0.0085 -> 0.0000`
* 零样本组合 `long+emoji+code_first` 生成了代码，但没 emoji——因为训练数据里没见过 emoji，这是预料内的小数据问题。

### 8. 反馈编码器：Transformer 预测纠正类型

```
训练预测器（任务+答案风格 -> 纠正维）: train acc 0.718 (chance 0.25)
vague 无预测: 0.911
vague 只猜维: 0.886 (delta -0.025)
vague 猜维+值 (dim+value parser): 0.762 (-0.149 vs vague)
```

结论：预测器方向对了，但当前只猜维度增益小（0.913→0.886）。瓶颈是“猜对维以后怎么拿对值”，值错写会把增益抵掉。猜维+值分布把值也一起给，错误回到 0.762，涨得更明显，但仍虚高（句式背诵过拟合）。

### 9. LLM 作为编码器（Qwen2.5-0.5B 隐状态 896维）

```
训练短语: 1.000
未见同义: 0.857 (哈希背诵在未见词上会直接跌)
```

大模型自身的翻译能力会把“删掉表情符号”/“别用 emoji”拉得很近，远比哈希词袋抗换说法。验证了“直接用大模型预测器权重”这条路有泛化增益，且共享表示不需第二套字典。

### 10. 长跑 1000 轮 + 噪音鲁棒

```
1000 轮静态: Static/Rule/Hybrid 全部 0.386，槽位稳在 3 个，没爆
任务轮换 4 相位: Rule 0.70/0.53/0.68/0.54 vs TaskPart 0.41/0.39/0.39/0.38
噪音 10% 假纠正: Static/Rule/Hybrid 全部 0.465，MLP trust 没过滤到噪音
```

生长/退化在“隔离任务冲突”上已经兑现了，分片把跨任务互相覆盖都防住了；噪音鲁棒还没做出来，下一轮先加确认门槛。

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
* [x] B1 公平基准（去 oracle / 去污染）+ Latest-Value
* [x] 长跑、任务轮换、噪音鲁棒
* [x] Transformer 预测纠正类型 + LLM-encoder 未见同义泛化
* [ ] 把 Value Parser 换成 BGE/Qwen3-Emb 真编码器版（现在是冻结的 Qwen hidden）
* [ ] 模糊反馈：从“只猜维”升级到“维+值分布+置信度”并测 false update 率
* [ ] 任务键把 recovery 33.9 优势撑大：临时习惯 vs 长期习惯冲突
* [ ] Hypernet 升到多层生成（现在只最后层 22528 维），补 emoji 数据（训练基没见过 emoji）
* [ ] App 化：本地 FastAPI + SQLite vec

---

## 版权 / 协议

教育实验代码，Apache 2.0。模型权重来自 Qwen2.5-0.5B-Instruct（Apache 2.0）。
