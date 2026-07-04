# 生成加性注意力 Checkpoints

目标：按阶段实现并验证 Bahdanau 风格的加性注意力机制。重点不是一次性写完代码，而是先确认张量形状、接口契约、mask 逻辑和输出语义，再把它接入 seq2seq 解码器。

使用方式：按顺序完成 checkpoint。每个 checkpoint 只要求完成当前阶段，不要求一次性做完整实现。

---

## Checkpoint 0：确认当前基线

### 目标
确认你已经理解当前项目里已有的 attention / seq2seq 代码位置，以及使用的是 PyTorch 张量约定。

### 你要完成
- 找到当前 notebook 或代码中已有的注意力相关实现。
- 确认数据流里的三个核心张量：
  - `queries`
  - `keys`
  - `values`
- 写下注释说明它们在 seq2seq 中分别对应什么。

### 通过标准
- 你能说清楚：
  - decoder hidden state 通常作为 `query`
  - encoder outputs 通常作为 `keys` 和 `values`
  - attention weights 的形状应和 source sequence length 对齐
- 当前 notebook 能从头运行到已有相关单元，不因前置变量缺失报错。

### 常见问题
- 把 `query` 和 `key/value` 的时间维搞反。
- 只关注公式，不先确认 batch 维和 sequence 维顺序。

---

## Checkpoint 1：定义 AdditiveAttention 接口

### 目标
先固定类的调用接口，避免后面实现时形状混乱。

### `AdditiveAttention` 要实现的接口

#### 1. `__init__(self, key_size, query_size, num_hiddens, dropout)`

```python
def __init__(
    self,
    key_size: int,
    query_size: int,
    num_hiddens: int,
    dropout: float,
) -> None:
    """
    key_size: keys 最后一维的特征数
    query_size: queries 最后一维的特征数
    num_hiddens: 注意力打分空间的隐藏维度
    dropout: attention weights 上使用的 dropout 概率
    返回: 无
    """
```

构造阶段应该创建：
- `W_k`: 把 keys 投影到 `num_hiddens`
- `W_q`: 把 queries 投影到 `num_hiddens`
- `w_v`: 把 tanh 后的联合特征投影成标量 score
- `dropout`

构造阶段不应该传入具体 batch 数据。

#### 2. `forward(self, queries, keys, values, valid_lens)`

```python
def forward(
    self,
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    valid_lens: torch.Tensor | None,
) -> torch.Tensor:
    """
    queries: shape (batch_size, num_queries, query_size)
    keys: shape (batch_size, num_kv_pairs, key_size)
    values: shape (batch_size, num_kv_pairs, value_size)
    valid_lens: shape (batch_size,) 或 (batch_size, num_queries)，表示每个样本有效 key/value 长度
    返回: shape (batch_size, num_queries, value_size)
    """
```

运行阶段应该完成：
- 线性投影
- 广播相加
- `tanh`
- 标量打分
- masked softmax
- dropout
- 加权求和

| 方法 | 输入 | 输出 |
| --- | --- | --- |
| `__init__` | 维度配置和 dropout | 无 |
| `forward` | `queries`, `keys`, `values`, `valid_lens` | context vectors |

### 通过标准
- 类可以被实例化。
- `forward` 的输入输出形状在注释里写清楚。
- 你能解释 `__init__` 负责参数层，`forward` 负责具体样本计算。

### 常见问题
- 在 `__init__` 里传入具体序列数据。
- 忘记 `queries` 可能有多个 query，不一定只有一个时间步。

---

## Checkpoint 2：实现未加 mask 的打分逻辑

### 目标
先让加性注意力的 score 计算跑通，不急着处理 padding。

### 你要完成
- 对 `queries` 做线性变换：得到 `(batch_size, num_queries, num_hiddens)`。
- 对 `keys` 做线性变换：得到 `(batch_size, num_kv_pairs, num_hiddens)`。
- 通过扩维广播得到联合特征：
  - queries 扩成 `(batch_size, num_queries, 1, num_hiddens)`
  - keys 扩成 `(batch_size, 1, num_kv_pairs, num_hiddens)`
- 相加后过 `tanh`。
- 用 `w_v` 得到 scores：
  - shape 应为 `(batch_size, num_queries, num_kv_pairs)`。

### 通过标准
- 使用一个小样例能打印出正确 scores 形状。
- 不出现 broadcasting error。
- scores 的最后一维长度等于 `keys` / `values` 的时间长度。

### 常见问题
- 忘记 `squeeze(-1)`，导致 scores 多出一个尾维。
- 在错误维度上扩展，导致 query length 和 key length 混在一起。

---

## Checkpoint 3：接入 masked softmax

### 目标
让 padding 位置不会参与注意力分布。

### 你要完成
- 使用已有的 `masked_softmax`，如果项目中还没有，则先实现或复用对应工具函数。
- 输入 scores：
  - shape `(batch_size, num_queries, num_kv_pairs)`
- 输入 `valid_lens`：
  - shape `(batch_size,)` 或 `(batch_size, num_queries)`
- 得到 `attention_weights`：
  - shape `(batch_size, num_queries, num_kv_pairs)`

### 通过标准
- 对每个 query，非 padding 位置权重和接近 1。
- padding 位置权重接近 0。
- 当 `valid_lens = [2, 3]` 时，第一个样本从第 3 个 key 开始应被 mask。

### 常见问题
- mask 后 softmax 维度选错，应该沿最后一维 `num_kv_pairs` 做 softmax。
- `valid_lens` 没有按 `num_queries` 正确重复。
- 用 0 直接替换 padding scores，而不是用很小的负数，导致 padding 仍可能拿到权重。

---

## Checkpoint 4：计算 context 输出

### 目标
用 attention weights 对 values 加权求和，得到上下文向量。

### 你要完成
- 对 `attention_weights` 使用 dropout。
- 执行 batch matrix multiplication：
  - `attention_weights`: `(batch_size, num_queries, num_kv_pairs)`
  - `values`: `(batch_size, num_kv_pairs, value_size)`
  - 输出: `(batch_size, num_queries, value_size)`
- 将 `attention_weights` 保存到实例属性，方便可视化或调试。

### 通过标准
- `forward` 返回 shape 为 `(batch_size, num_queries, value_size)`。
- `self.attention_weights` 可被外部访问。
- 一个随机输入样例可以完整前向运行。

### 常见问题
- 把 `keys` 当作加权对象，而不是 `values`。
- 使用普通矩阵乘法导致 batch 维不匹配。
- 忘记保存 attention weights，后续无法可视化。

---

## Checkpoint 5：接入 seq2seq decoder

### 目标
把加性注意力用于解码器，让每个解码时间步根据 encoder outputs 动态生成 context。

### 你要完成
- 在 decoder 初始化中创建 `AdditiveAttention`。
- 在每个 decoding step：
  - 使用当前 decoder hidden state 作为 query。
  - 使用 encoder outputs 作为 keys 和 values。
  - 使用 source valid lengths 做 mask。
  - 将 context 和当前输入 token embedding 拼接后送入 RNN / GRU / LSTM。
- 确认输出维度能接到后续预测层。

### Decoder 中推荐保持的接口

```python
def init_state(
    self,
    enc_outputs: torch.Tensor,
    enc_valid_lens: torch.Tensor,
    *args,
) -> tuple:
    """
    enc_outputs: encoder 输出，通常包含所有时间步输出和最终 hidden state
    enc_valid_lens: shape (batch_size,)
    返回: decoder 初始状态，需包含 encoder outputs、hidden state、valid_lens
    """
```

```python
def forward(
    self,
    X: torch.Tensor,
    state: tuple,
) -> tuple[torch.Tensor, tuple]:
    """
    X: shape (batch_size, num_steps)，目标序列输入 token ids
    state: init_state 返回的状态
    返回:
      outputs: shape (batch_size, num_steps, vocab_size)
      state: 更新后的 decoder state
    """
```

### 通过标准
- decoder 前向传播不报 shape 错误。
- 输出 logits 的时间步数等于目标输入序列长度。
- attention weights 的 query 维度能对应 decoder 的每个时间步。

### 常见问题
- encoder outputs 的维度顺序是 `(num_steps, batch_size, hidden_size)` 还是 `(batch_size, num_steps, hidden_size)` 没确认就直接使用。
- query 少了时间维，应该保持类似 `(batch_size, 1, hidden_size)`。
- 拼接 context 和 embedding 时维度不一致。

---

## Checkpoint 6：训练与可视化验证

### 目标
确认加性注意力不仅能运行，而且能在任务中学习到合理对齐。

### 你要完成
- 用小规模训练先跑通完整流程。
- 记录 loss 是否下降。
- 对一个样例预测结果可视化 attention weights。
- 检查 attention heatmap 的横轴是否对应输入序列，纵轴是否对应输出序列。

### 通过标准
- 训练 loss 有下降趋势。
- 推理阶段能生成非空输出。
- attention heatmap 形状合理：
  - 行数约等于输出 token 数
  - 列数约等于输入 token 数
- padding 位置不应出现明显高权重。

### 常见问题
- 可视化时把 batch 维、query 维、key 维顺序搞反。
- 训练失败时只调学习率，不先检查 mask 和 shape。
- 忽略 `<bos>`, `<eos>`, `<pad>` 对 attention 可视化的影响。

---

# 推荐执行顺序

```text
Checkpoint 0 -> Checkpoint 1 -> Checkpoint 2 -> Checkpoint 3 -> Checkpoint 4 -> Checkpoint 5 -> Checkpoint 6
```

# 当前阶段的最终目标

你完成后应拥有一个可复用的 `AdditiveAttention` 模块，并能把它接入 seq2seq decoder。它应该满足：

- 支持 batch 输入。
- 支持多个 query。
- 支持 padding mask。
- 返回 context vectors。
- 保存 attention weights 供可视化。
- 能在 seq2seq 训练和推理流程中正常运行。
