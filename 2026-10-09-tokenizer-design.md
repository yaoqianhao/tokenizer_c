---
AIGC:
    Label: "1"
    ContentProducer: 001191440300708461136T1XGW3
    ProduceID: 04526fdd866a8c5c9aceee6b83ed2c05_20251457c3cf11f1a08152540064ee0f
    ReservedCode1: L5jZhyY8VRkypTHJEvX5jM1BPiFf8qQ7Ycn9fWsPnl72sJ91GBcwLgSCi/a3cUTvA6Mm416/3UsUzgriwcbClTP5B6+j3gPdG6O67OyBuM+fLpfVAJxHwIitNCUtPhvTGQdvSsXqUDIrHoCEmBGxmtg9kSxcVICtJ90KuvXFd1Rl7HxmQjLSXZ6h1LU=
    ContentPropagator: 001191440300708461136T1XGW3
    PropagateID: 04526fdd866a8c5c9aceee6b83ed2c05_20251457c3cf11f1a08152540064ee0f
    ReservedCode2: L5jZhyY8VRkypTHJEvX5jM1BPiFf8qQ7Ycn9fWsPnl72sJ91GBcwLgSCi/a3cUTvA6Mm416/3UsUzgriwcbClTP5B6+j3gPdG6O67OyBuM+fLpfVAJxHwIitNCUtPhvTGQdvSsXqUDIrHoCEmBGxmtg9kSxcVICtJ90KuvXFd1Rl7HxmQjLSXZ6h1LU=
---

# 工业级 Tokenizer 设计方案（Google Gemini Unigram 算法框架）

- 日期：2026-10-09
- 状态：已确认，待实现
- 目标：设计并研发一个工业级 tokenizer，采用 Google Gemini 的算法框架（SentencePiece Unigram）

---

## 1. 背景与目标

### 1.1 背景事实（对标基准）
- Gemini / Gemma 3 官方 tokenizer 使用 **SentencePiece + Unigram 算法**。
- 词表规模 **262,144（256K）**，为当前主流 LLM 中最大词表之一。
- **无预分词（pre-tokenization）**：不在空格/标点处预先切分，直接在原始文本上操作。
- 空白作为普通字符处理，编码时以 `▁`（U+2581）表示空格，解码可逆。

### 1.2 目标
- 训练一个 256K 词表的 Unigram 模型，具备多语言、多领域覆盖能力。
- 提供高性能编码/解码引擎，满足训练管线与在线推理双场景。
- 模型格式可导出为 SentencePiece proto 与 HuggingFace `tokenizer.json`，可被主流框架直接加载。

### 1.3 非目标（YAGNI）
- 不做端到端 LLM 训练，只交付 tokenizer 本身。
- 不做 BPE / WordPiece 等其它算法的训练器（保留扩展接口）。
- 不做分布式集群调度系统，训练并行以单机多核 + 数据分片为主。

---

## 2. 算法选型与依据

### 2.1 为什么是 Unigram
| 对比项 | Unigram（Gemini/T5） | BPE（LLaMA/GPT） | WordPiece（BERT） |
|---|---|---|---|
| 多语言均衡性 | 强（概率模型全局最优） | 中（贪心合并偏向高频语言） | 中 |
| 未知字符处理 | 天然支持字节回退 | 需显式设计 | 较弱 |
| 切分可解释性 | 有概率分数 | 无 | 无 |
| 编码最优性 | Viterbi 全局最优 | 贪心 | 贪心 |
| 词表增长策略 | 概率剪枝（可精确控大小） | 合并步数控制 | 合并步数控制 |

结论：Gemini 选用 Unigram 的核心原因即多语言均衡与字节回退兜底，本项目沿用同一框架。

### 2.2 Unigram 模型原理
- 将文本切分视为序列概率模型：`P(x) = Π P(t_i)`，其中 `t_i` 为词表 piece。
- 训练 = 在语料上最大化似然，同时通过剪枝控制词表规模。
- 编码 = 在 DAG 上求最大概率路径（等价于加权 Viterbi / DP）。

---

## 3. 核心设计决策

| 维度 | 决策 | 说明 |
|---|---|---|
| 归一化 | NFKC + 可选大小写折叠 | 兼容 Gemini 行为，保留开关供领域定制 |
| 未知字符 | UTF-8 字节回退（byte fallback） | 每个未知字节映射到 `<0xXX>` 类 piece，OOV 率恒为 0 |
| 特殊 token | `<unk>/<bos>/<eos>/<pad>` + 自定义段 | 固定 id 区间，训练期即预留 |
| 空格处理 | `▁` 前缀符号 | SentencePiece 标准，保证无损往返 |
| 词表规模 | 256K（可配置 32K~512K） | 默认对齐 Gemini；大词表需配套 embedding 显存评估 |
| 序列化 | SentencePiece proto + JSON | 双格式导出，HF 生态可直接加载 |

---

## 4. 系统分层架构

```
┌─────────────────────────────────────────────┐
│ 数据层   语料采集/清洗/去重/语言识别/配比采样   │
├─────────────────────────────────────────────┤
│ 训练层   Unigram EM 迭代（Viterbi→重估→剪枝） │
├─────────────────────────────────────────────┤
│ 推理层   编码(Viterbi DP)/解码，mmap、批量、缓存│
├─────────────────────────────────────────────┤
│ 接口层   Python API / C API / 双格式导出      │
├─────────────────────────────────────────────┤
│ 评估层   压缩率/OOV/吞吐/与官方模型对齐测试     │
└─────────────────────────────────────────────┘
```

### 4.1 模块职责与接口
- **data_pipeline**：输入原始语料目录，输出按语言/领域配比采样的训练分片（JSONL）。
- **trainer**：输入分片 + 目标词表大小，输出训练中模型（断点可续）。
- **tokenizer_core**：只读加载模型，提供 `encode(text) -> ids`、`decode(ids) -> text`。
- **bindings**：Python 包与 C API，均转发到 tokenizer_core。
- **eval_harness**：对任意模型文件跑压缩率/OOV/速度基准。

各模块通过模型文件与数据格式解耦，可独立测试与替换。

---

## 5. 训练管线详细设计

### 5.1 语料预处理
1. 原始文本清洗（去控制字符、去重、格式规整）。
2. 归一化（NFKC、可选 lower）。
3. 语言识别（fasttext / CLD3 类工具），统计语言分布。
4. 按目标配比采样（默认中文、英文、代码、其它各占合理比例），控制各语言序列数。
5. 输出训练分片，记录元数据（来源、语言、行数）。

### 5.2 种子词表构建
- 从采样语料提取字符 n-gram（n=1..N，如 N=16）作为候选 piece。
- 设候选上限（如 10M）与内存水位，超限按频率截断。
- 合并单字节 UTF-8 序列为字节回退候选。

### 5.3 EM 迭代训练
- **E 步**：对每个训练序列，用当前词表做 Viterbi 最优切分（对数概率 DP）。
- **M 步**：统计各 piece 的出现次数，重估 `P(t) = count(t) / Σcount`。
- **剪枝**：按新概率排序，保留 top-K（K 逐步逼近目标词表），删除低频 piece。
- 收敛条件：目标词表大小达成且连续 2 轮似然提升 < 阈值。

### 5.4 并行与资源控制
- **数据并行**：语料分片到多进程，各自做 E 步计数，主进程合并计数后 M 步。
- **断点续训**：每轮落盘计数与模型快照，中断后可恢复。
- **内存控制**：piece 计数用紧凑 hash map；超长序列截断到最大长度（如 4K chars）参与训练。

### 5.5 后处理与导出
1. 合并字节回退 piece、固定特殊 token id。
2. 按 id 重排序，写入 SentencePiece proto（`.model`）。
3. 生成 HF `tokenizer.json`（含 normalization / pre_tokenizer=none / unigram 配置）。
4. 输出词表统计报告（覆盖度、最长 piece、语言分布）。

---

## 6. 推理引擎设计

### 6.1 编码（Encode）
- 输入归一化后的文本，按字符序列构建 DAG，节点为位置，边为词表 piece。
- 用对数概率做 **Viterbi / DP** 求全局最优路径。
- 优化：
  - **查找加速**：双数组 Trie / 前缀 hash 表，将 piece 匹配从 `O(V)` 降到 `O(len)`。
  - **批量**：多文本批量编码，SIMD 友好的紧凑内存布局。
  - **缓存**：高频子串/整句结果 LRU 缓存。
- **流式/增量编码**：支持 token 级增量（前缀共享复用 DP 前缀结果）。

### 6.2 解码（Decode）
- `ids -> pieces -> 拼接 -> ▁ 还原为空格 -> 字节回退还原 -> 归一化逆操作（如需）`。
- 解码为纯查表 + 拼接，无分支热点，可向量化。

### 6.3 并发与加载
- 模型以 **mmap 只读映射**加载，多线程共享零拷贝。
- 推理核心无全局可变状态，线程安全。
- 超长文本分块处理，防止栈溢出与内存峰值。

---

## 7. 工业级强化点清单

| 类别 | 措施 |
|---|---|
| 训练可靠性 | 断点续训、校验和、日志可审计、失败自动重试 |
| 推理性能 | mmap、Trie 加速、批量编码、LRU 缓存、流式增量 |
| 鲁棒性 | 字节回退兜底、非法输入不崩溃、超长分块 |
| 兼容性 | proto 与 HF tokenizer.json 双导出、与官方模型对齐测试 |
| 可观测 | 编码耗时 p50/p99、吞吐、内存峰值指标上报 |
| 安全 | 模型文件完整性校验、禁止路径穿越读取训练语料 |

---

## 8. 接口与序列化格式

### 8.1 Python API（示意）
```python
from tokenizer import Tokenizer

tok = Tokenizer.load("gemini_like.model")           # 或 tokenizer.json
ids = tok.encode("Hello 世界")                       # -> [..]
text = tok.decode(ids)                               # 无损还原
tok.encode_batch(["...", "..."], num_threads=8)      # 批量
tok.encode_stream("长文本")                          # 增量迭代器
```

### 8.2 C API（示意）
```c
tokenizer_t *tok = tokenizer_load("gemini_like.model");
int *ids; size_t n;
tokenizer_encode(tok, "Hello 世界", &ids, &n);
char *text = tokenizer_decode(tok, ids, n);
tokenizer_free(tok);
```

### 8.3 模型格式
- 主格式：SentencePiece proto（`<unk>`,`<s>`,`</s>`,`<pad>` + `▁` 前缀 piece + 字节回退 piece）。
- 兼容格式：HF `tokenizer.json`（`type: "Unigram"`, `pre_tokenizer: null`, `normalizer: NFKC`）。

---

## 9. 评估与验收标准

| 指标 | 目标 |
|---|---|
| OOV 率 | 恒为 0（字节回退兜底） |
| 压缩率 | 多语言混合基准 ≥ 同等词表规模 SentencePiece 基线 |
| 编码吞吐 | ≥ 基线引擎 1x（纯 Python 目标 ≥ 50k tokens/s 单线程量级） |
| 对齐测试 | 与公开 Gemini tokenizer 在样例集上 token 级一致率 ≥ 95% |
| 训练收敛 | 256K 词表在 10GB 级语料上稳定收敛，可断点续训 |
| 往返一致性 | 100% 样例 decode(encode(x)) == 归一化(x) |

---

## 10. 实现路径（语言待定，方案已解耦）

### 方案 A：纯 Python
- 阶段 1：numpy 向量化 EM 训练器 + 纯 Python 推理（正确性优先）。
- 阶段 2：多进程并行训练、JIT（numba）热点优化、LRU 缓存。
- 优点：开发快、生态接入直接；缺点：极致吞吐受限。

### 方案 B：纯 C
- 阶段 1：C 实现核心训练与推理，暴露 C API。
- 阶段 2：提供 Python ctypes/cffi 绑定与 CLI。
- 优点：性能极致、可嵌入任意语言；缺点：开发量大、需自行管理内存。

### 建议
两个方案共用同一模型格式与测试套件，后续可按性能诉求平移。若选 C，训练器可先以 Python 正确实现（训练非在线热点），仅推理引擎用 C 保证吞吐。

---

## 11. 里程碑建议

1. M1：语料预处理 + 种子词表（1 周）
2. M2：Unigram 训练器收敛（2~3 周）
3. M3：推理引擎 + 双格式导出（2 周）
4. M4：评估对齐 + 工业强化（缓存/并发/断点）（1~2 周）
5. M5：打包发布 + 文档（1 周）

总工期估算：6~9 周（单工程师，语言按所选方案浮动）。

---

## 12. 风险与对策

| 风险 | 对策 |
|---|---|
| 256K 词表训练内存峰值高 | 候选截断、计数分片合并、紧凑 hash |
| 多语言语料配比失衡 | 语言识别 + 配比采样 + 覆盖度报告 |
| 与官方 tokenizer 行为差异 | 建立对齐测试集，逐条 diff 定位 |
| 纯 Python 吞吐不达标 | 热点 JIT / 降级为 C 推理引擎 |
| 字节回退与特殊 token 冲突 | 保留独立 id 区间，训练期校验不重叠 |
*（内容由AI生成，仅供参考）*
