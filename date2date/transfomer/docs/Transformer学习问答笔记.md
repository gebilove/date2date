# Transformer 学习问答笔记

> 围绕原始 Transformer 的分词、Embedding、位置编码、LayerNorm、编码器—解码器连接、并行训练与自回归的整理。

## 0. 先建立完整的数据流

以机器翻译为例：

```text
源文本
  → Tokenizer 切成 token
  → token ID
  → Embedding + 位置编码
  → Encoder
  → 编码器表示（memory）
                         ↘
目标文本右移 → Embedding + 位置编码 → Decoder
                         → Masked Self-Attention
                         → Cross-Attention（读取 memory）
                         → FFN
                         → 词表 logits
                         → Softmax / 解码策略
                         → 下一个 token
```

训练时目标句子已知，可以一次输入右移后的完整目标序列；推理时目标句子未知，需要逐 token 生成。

---

## 1. Transformer 的并行训练体现在哪里？

### RNN 为什么难以沿时间维度并行？

RNN 在时间步 \(t\) 的隐藏状态依赖前一步：

\[
h_t=f(h_{t-1},y_{t-1})
\]

即使训练时使用 teacher forcing，已经知道每一步的正确目标 token，仍然只能提供 \(y_{t-1}\)，不能提前得到 \(h_{t-1}\)。因此：

```text
h1 → h2 → h3 → h4
```

必须顺序计算。

### Transformer 为什么可以并行训练？

Transformer 没有跨时间步递归传递的 hidden state。训练解码器时，将目标序列右移后一次输入：

```text
目标输出： 我   喜欢 学习 <EOS>
解码输入：<BOS> 我   喜欢 学习
```

因果掩码保证位置 \(t\) 只能看到 \(y_{<t}\)，于是所有位置的损失可以在一次矩阵计算中得到：

\[
P(y\mid x)=\prod_{t=1}^{T}P(y_t\mid y_{<t},x)
\]

需要准确区分：

- Transformer 仍然有每层、每个位置的 hidden representation。
- 它没有 RNN 那种必须由前一时间步计算后才能得到的 recurrent hidden state。
- 并行化的根本原因是消除了时间维度上的递归依赖。
- 因果掩码负责防止看到未来答案，不是并行化本身的根源。
- 各层之间仍需顺序计算，因此并不是所有计算都并行。

标准自回归 Transformer 在**训练时可以并行处理多个目标位置**，但在**推理时仍需逐 token 生成**。

---

## 2. Seq2Seq 都是自回归的吗？

不是。

- **Seq2Seq** 描述任务形式：输入序列映射到输出序列。
- **自回归**描述输出概率如何分解以及推理时如何生成。

### 自回归 Seq2Seq

\[
P(y\mid x)=\prod_{t=1}^{T}P(y_t\mid y_{<t},x)
\]

预测 \(y_t\) 时依赖此前已经生成的 \(y_{<t}\)。典型例子包括：

- RNN/LSTM Encoder–Decoder；
- 原始 Transformer 翻译模型；
- T5、BART 的生成过程；
- GPT 式从左到右生成。

### 非自回归方法

输出位置不依赖“此前已经生成的输出”，或者通过整体迭代而非从左到右生成，例如：

- NAT 非自回归机器翻译：预测长度后并行生成各位置；
- CTC：各帧并行预测，再删除 blank、合并重复项；
- Mask-Predict：先并行生成，再反复遮盖并修正低置信度 token；
- 词性标注、NER 等序列标注；
- 扩散或迭代去噪式文本生成；
- BERT 的 Masked Language Modeling。

判断标准：

> 推理时，预测 \(y_t\) 是否必须把已经生成的 \(y_{<t}\) 作为条件？

若必须如此，通常就是自回归。注意：**自回归并不等于训练一定串行**。Transformer 的自回归目标可以借助因果掩码并行训练。

---

## 3. 为什么 Transformer 需要位置编码？

输入张量的各行确实有排列顺序，但普通自注意力不会自动把“行号”当作特征：

\[
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V
\]

\(W_Q,W_K,W_V\) 对所有位置共享。若没有任何位置信号，交换输入 token 只会相应地交换输出；模型无法仅凭注意力理解谁在前、谁在后。

加入位置编码后：

\[
X_i'=E(token_i)+PE(i)
\]

再计算：

\[
Q=X'W_Q,\qquad K=X'W_K,\qquad V=X'W_V
\]

这样 Q、K、V 中便包含位置信息，模型可以学习相邻、前后和距离等关系，进而区分：

```text
我 打 你
你 打 我
```

更准确的表述是：

> 位置编码为 token 表示注入绝对或相对位置信息，使注意力机制能够建模顺序和距离。输入张量虽然有排列次序，但共享的 QKV 投影不会自动理解行号的语义。

### 位置编码与因果掩码的区别

- **因果掩码**控制“能看见谁”：当前位置不能看到未来位置。
- **位置编码**表达“各自在何处、相隔多远”。

二者作用不同，不能互相简单替代。

---

## 4. 除了正弦/余弦，还有哪些位置方法？

### 4.1 固定绝对位置编码

原始 Transformer 使用正弦和余弦函数生成固定位置向量，再加到 token embedding 上。它没有需要训练的位置参数。

### 4.2 可学习绝对位置嵌入

\[
X_i=E(token_i)+P_i
\]

\(P_i\) 是可训练参数。方法直接，但训练长度之外的位置可能没有得到充分训练，因此长度外推通常受限。

### 4.3 相对位置编码与相对位置偏置

直接根据两个 token 的距离 \(i-j\) 修改注意力：

\[
score(i,j)=q_i^\top k_j+r_{i-j}
\]

T5 使用相对位置分桶，将距离映射为可学习的 attention bias。语言中“相隔多远”往往比“绝对位于第几个位置”更有意义。

### 4.4 RoPE

RoPE 根据位置旋转 Q、K：

\[
q_i'=R_iq_i,\qquad k_j'=R_jk_j
\]

于是：

\[
(q_i')^\top k_j'=q_i^\top R_{j-i}k_j
\]

点积自然包含相对位置 \(j-i\)。RoPE 通常作用于 **Q、K，而不是 V**。

### 4.5 ALiBi

ALiBi 不创建位置向量，而是在 attention logits 中加入与距离相关的线性惩罚：

\[
score(i,j)=\frac{q_i^\top k_j}{\sqrt d}-m_h|i-j|
\]

不同 attention head 可使用不同斜率。

### 4.6 NoPE

NoPE 不显式加入位置编码。因果模型可能借助“每个位置能看到的前缀长度不同”隐式学习部分位置信息，但这不意味着一般情况下都能安全删除位置机制。

延伸阅读：

- [Self-Attention with Relative Position Representations](https://arxiv.org/abs/1803.02155)
- [RoFormer / RoPE](https://arxiv.org/abs/2104.09864)
- [ALiBi](https://openreview.net/pdf?id=R8sQPpGCv0)
- [Transformer Language Models without Positional Encodings](https://arxiv.org/abs/2203.16634)

---

## 5. LayerNorm 是以 token 为单位归一化吗？

在标准 Transformer 中，可以这样理解。

设隐藏状态形状为：

\[
X\in\mathbb{R}^{B\times T\times D}
\]

其中 \(B\) 是 batch size，\(T\) 是序列长度，\(D=d_{\text{model}}\)。`LayerNorm(D)` 对每个 `(样本, token)` 的 \(D\) 个特征分别计算：

\[
\mu_{b,t}=\frac1D\sum_{i=1}^{D}x_{b,t,i}
\]

\[
\sigma^2_{b,t}=\frac1D\sum_{i=1}^{D}(x_{b,t,i}-\mu_{b,t})^2
\]

\[
\hat{x}_{b,t,i}
=\gamma_i\frac{x_{b,t,i}-\mu_{b,t}}
{\sqrt{\sigma^2_{b,t}+\epsilon}}+\beta_i
\]

即：

```text
固定 b、t，沿 D 归一化
```

不同 token 不会一起统计均值和方差；不同 batch 样本也不会一起统计。

### 为什么叫 LayerNorm，而不叫 TokenNorm？

LayerNorm 的名称早于 Transformer。它的一般含义是：

> 对单个样本在某一层产生的一组神经元激活进行归一化。

在 Transformer 中，一个 token 的 \(D\) 维向量恰好对应某层在该位置的一组激活，所以表现为逐 token 归一化。“逐 token”是 `LayerNorm(d_model)` 在 Transformer 中的具体结果，不是 LayerNorm 在所有模型中的一般定义。

对比：

- BatchNorm：通常固定特征，跨 batch（有时也跨空间/时间轴）统计。
- LayerNorm：固定样本，沿指定的特征轴统计。

原始 Transformer 使用 **Post-Norm**：

\[
\operatorname{LayerNorm}(x+\operatorname{Sublayer}(x))
\]

后来的很多实现改用 Pre-Norm：

\[
x+\operatorname{Sublayer}(\operatorname{LayerNorm}(x))
\]

---

## 6. 原始 Transformer 的 Embedding 是现成的还是一起训练的？

原始 Transformer 没有直接使用现成的 Word2Vec 或 GloVe。token embedding 是模型参数，通常随机初始化，并与整个 Transformer 一起端到端训练。

设词表大小为 \(V\)：

\[
E\in\mathbb{R}^{V\times d_{\text{model}}}
\]

token ID 为 \(i\) 时，通过查表获得：

\[
x_i=E[i]
\]

反向传播会更新 \(E\)，因此 embedding 的语义结构也是在翻译任务中学习出来的。

原始 Transformer 还有两个关键设计：

1. 编码器输入 embedding、解码器输入 embedding，以及输出 softmax 前的线性变换共享权重，即 weight tying。
2. embedding 输出乘以 \(\sqrt{d_{\text{model}}}\)：

\[
X=\sqrt{d_{\text{model}}}\,E[token]+PE
\]

注意：

- token embedding：可训练；
- 原论文使用的正弦/余弦位置编码：固定，不训练；
- tokenizer 的词表和切分规则：一般在 Transformer 训练前确定，训练模型时保持固定。

---

## 7. 解码器如何使用编码器输出？

通过 **Cross-Attention（交叉注意力）**。

原始 Transformer 的解码器层依次包含：

```text
Masked Self-Attention
  → Add & Norm
  → Cross-Attention
  → Add & Norm
  → FFN
  → Add & Norm
```

交叉注意力中：

\[
Q=X_{\text{decoder}}W_Q
\]

\[
K=H_{\text{encoder}}W_K,\qquad
V=H_{\text{encoder}}W_V
\]

- Q 来自解码器：当前目标位置“想查什么”；
- K 来自编码器：每个源位置可以用什么特征匹配；
- V 来自编码器：匹配后实际取回的信息。

计算：

\[
A=\operatorname{softmax}
\left(\frac{QK^\top}{\sqrt{d_k}}\right)
\]

\[
C=AV
\]

若：

\[
H_{\text{encoder}}\in\mathbb{R}^{B\times S\times D}
\]

\[
X_{\text{decoder}}\in\mathbb{R}^{B\times T\times D}
\]

则每个 head 的注意力权重核心形状为：

\[
A\in\mathbb{R}^{B\times T\times S}
\]

含义是：每个目标位置分别关注源序列的哪些位置。

例如把 `I love cats` 翻译成“我喜欢猫”，生成“喜欢”时，query 可能对 `love` 对应的 key 给出较高权重，再从其 value 取回主要信息。

需要注意：

- Masked Self-Attention 只能看到此前的目标 token。
- Cross-Attention 可以看到完整源序列，一般只屏蔽源序列的 padding。
- 编码器最终输出作为 memory 提供给每一个解码器层。
- 每个解码器层、每个 head 都有自己的投影参数，可用不同方式读取同一份 memory。
- 推理时编码器一般只运行一次，编码器侧 K、V 可以缓存。

---

## 8. Transformer 如何切分文本？

切分发生在 Transformer 之前，由 tokenizer 完成：

```text
文本 → token → token ID → Embedding → Transformer
```

原始论文的英德任务使用 BPE 子词切分，约 37,000 个 token，并共享英德词表；英法任务使用约 32,000 个 word-piece 词表。

### BPE 的基本过程

1. 先将训练语料中的词拆成较小单位，例如字符。
2. 统计相邻符号对的出现频率。
3. 合并最高频的符号对。
4. 重复合并，直到达到指定词表大小。
5. 保存词表与合并规则，用相同规则处理训练、验证和推理文本。

例如，一个低频词可能被切为：

```text
unbelievable → un + believe + able
```

高频词则可能保持完整。

### 为什么使用子词？

- 按完整单词：词表巨大，而且容易出现未登录词。
- 按字符：词表小，但序列过长，单个 token 语义较弱。
- 按子词：在词表大小、序列长度与开放词汇能力之间折中。

还需要加入特殊 token，例如：

```text
<BOS>、<EOS>、<PAD>
```

训练解码器时通常在数据预处理阶段构造：

```text
完整目标：<BOS> 我 喜欢 NLP <EOS>
解码输入：<BOS> 我 喜欢 NLP
监督目标：我 喜欢 NLP <EOS>
```

Tokenizer 与 Embedding 不同：

- Tokenizer 决定怎样切分以及每个 token 对应哪个 ID；
- Embedding 根据 ID 查找向量，并在模型训练中更新；
- Transformer 接收的是向量，不直接处理字符串。

---

## 9. 一页式结论

1. RNN 即使使用 teacher forcing，隐藏状态仍有时间递归依赖；Transformer 去掉这种依赖，因此训练时可以沿序列位置并行。
2. Seq2Seq 不一定自回归；自回归描述输出是否按 \(P(y_t\mid y_{<t},x)\) 逐步生成。
3. Transformer 输入张量有排列，但自注意力不会自动理解行号；需要显式或隐式的位置机制。
4. 位置方法不只有正弦编码，还包括可学习绝对位置、相对位置偏置、RoPE、ALiBi 和 NoPE 等。
5. `LayerNorm(d_model)` 在 Transformer 中是逐 token、沿 hidden dimension 归一化。
6. 原始 Transformer 的 token embedding 与模型一起训练，不是直接使用现成静态词向量。
7. 解码器通过交叉注意力读取编码器输出：Q 来自解码器，K、V 来自编码器。
8. 文本先由 tokenizer 切成子词并映射成 ID，再进入 Embedding；原始 Transformer 使用 BPE/word-piece 类方法。

## 参考

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- 当前学习页面：[Transformer 论文精读](https://github.com/Hoper-J/AI-Guide-and-Demos-zh_CN/blob/master/PaperNotes/Transformer%20%E8%AE%BA%E6%96%87%E7%B2%BE%E8%AF%BB.md)


## 11.比较不同结构的“信息传递路径长度”：
架构	层内能否并行	两个远距离词元建立联系
CNN	可以	需要经过多层卷积
RNN	不可以	需要依次经过多个时间步
自注意力	可以	一层即可直接联系


CNN 和 RNN 的问题不应简单称为“数据丢失”。更准确地说：
CNN 层数不足时，远处的信息根本不在当前位置的感受野中。
即使层数足够，远距离信息也要经过多个中间层，学习起来更困难，可能被削弱或混合。
RNN 的远距离信息必须反复经过隐状态，容易出现遗忘以及梯度消失等问题。
总而言之，卷积神经网络和自注意力都拥有并行计算的优势， 而且自注意力的最大路径长度最短。 但是因为其计算复杂度是关于序列长度的二次方，所以在很长的序列中计算会非常慢。
NN 的计算复杂度关于序列长度 \(n\) 呈线性增长。
Transformer 的优势不是“计算量更少”，而是信息传递路径短、并行度高，因此更容易学习长距离依赖，并能充分利用 GPU；代价是标准自注意力对序列长度具有二次计算和内存复杂度，处理超长序列时成本很高。