# ARCHITECTURE

## 为什么做 Personal Cortex

现有记忆系统（RAG、MemGPT/Letta、Hindsight、PlugMem、LongMem）大多在做“存文本、检索文本”或“LLM 自我总结”。

但个人习惯不是一条文本，它是：

* 低维、反复出现的偏好（verbosity/emoji/format）
* 会漂移的（时间线反转）
* 需要本地、私密、在线和低 token 成本的

### 背景调研（2025-2026 前沿）

* **MemGPT / Letta**：OS 式三层内存（Core / Recall / Archival），LLM 自决存/取。强点是有 ADE 可视化；弱点是每次决策都花推理 token，且质量取决于模型判断。
* **Hindsight / Evermind**：被动提取 + 多路检索（语义+关键词+图+时间），LongMemEval 94.6%。适合事实回忆，不擅习惯。
* **LongMem / PlugMem**：插件式梯度记忆，把长历史压缩成可微 KV。PlugMem 2026 已经做成任务无关插件。
* **Profile-to-PEFT (P2P, 2025.10)**：Hypernetwork 把用户画像直接生成 LoRA，0 训练秒级上线，比 OPPU 快 33 倍，OOD 更稳。证明了“习惯向量 -> ΔW” 这条路可行。
* **SHINE (2026), DnD**：同样把上下文压缩成 LoRA，一次前向。

我们的贡献是把这三条线拼起来：

* RAG 不做检索
* Hypernetwork 只做最后一步的生成
* 中间层做 **槽位 + 衰减 + Hebbian 预测式学习**，补上生长/退化这一环

## 结构

```
人话 -> [Encoder] -> 数字坐标
           ↓
        [Slot Router] 输出重要性权重加权融合
           ↓
        [Slot Banks] 显式值 + Hebbian 区
           ↓
       习惯向量 (habit embed)
           ↓
      [HyperNet] -> ΔW = BA (LoRA)
           ↓
      大模型 Base (Qwen2.5-0.5B FP16)
```

关键细节：

### 快系统 = 海马体
* 每个维度一个槽位，显式存值
* 惊讶 = 预测习惯 - 真实纠正，只对纠正的维度写
* `slot.value = 真值` 立即跳变，没有梯度等待

### 慢系统 = 皮层
* 协调重要性的 MLP 或赫布权重，学“这个槽值不值得留”
* 重要性衰减 `importance *= γ`，闲置时降到阈值下剪枝
* 槽位原型 `0.9*old + 0.1*x` 做慢固化

### 分片与路由
* 按任务类型（code/summary/explain/create）分片
* 路由用 `softmax(-dist/T)` 给原型距离加权，最大 2 个活跃槽
* 没有任务键时，漂移错 0.65，有任务键 0.43——说明分离必须带上下文输入

### 惊讶计算
1. 用当前包预测习惯
2. 大模型按预测出答案
3. 用户纠正 -> 预测错了，那个维度惊讶=1
4. 没有纠正 -> 惊讶=0，只强化重要性

监督信号全是 implicit feedback，不需要用户点选。

## 隐私与成本

* 权重小到 0.5k 参数左右，完全可在端侧运行
* 习惯向量 5 维，LoRA rank 8 左右，合并后推理多 0 token
* 原始对话不出本地，只有习惯向量在本地生成
