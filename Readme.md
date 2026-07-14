# date2date

本项目以“日期格式转换”为一个小型、可控、可自动生成数据的序列转换任务：

```text
YYYY-MM-DD -> DD/MM/YYYY
```

例如：

```text
2026-06-28 -> 28/06/2026
```

项目目标不是用神经网络替代规则字符串处理，而是通过同一个任务，从零实现并比较不同序列建模结构。

## 环境安装

当前核心环境使用 Python 3.13 和 PyTorch 2.8.x：

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

核心训练、实验 CLI 和测试只依赖 PyTorch；测试使用标准库 `unittest`，不需要安装 pytest。需要运行受支持的 notebook 时，额外安装：

```bash
.venv/bin/python -m pip install -r requirements-notebooks.txt
```

`attention/attention-Nadaraya-Watson.ipynb` 和 `attention/attention_train_test.ipynb` 仍依赖旧版 `d2l`，属于 legacy 教学材料，未纳入当前 Python 3.13 依赖环境。后续应将其中的 `d2l` API 替换为原生 PyTorch/Matplotlib 实现。

## 实验结果

<!-- EXPERIMENT_RESULTS_START -->

| 模型 | Batch | 测试集 | Exact Match | 字符准确率 | 参数量 | 训练耗时 |
|---|---:|---:|---:|---:|---:|---:|
| bahdanau | 1 | 100 | 100.00% | 100.00% | 1,128,719 | 202.3s |
| bahdanau | 32 | 100 | 100.00% | 100.00% | 1,128,719 | 6.4s |
| dot_product_attention | 32 | 100 | 100.00% | 100.00% | 996,878 | 4.2s |
| multi_head_attention | 32 | 100 | 100.00% | 100.00% | 2,048,782 | 14.4s |
| vanilla_gru | 1 | 100 | 76.00% | 96.10% | 800,270 | 118.9s |
| vanilla_gru | 32 | 100 | 82.00% | 96.70% | 800,270 | 2.8s |
| vanilla_rnn | 32 | 100 | 0.00% | 69.90% | 273,934 | 1.9s |

> 结果由 `experiments/experiments.csv` 自动生成；每种模型和 Batch 配置展示测试 Exact Match 最优的一次运行。

<!-- EXPERIMENT_RESULTS_END -->

## 支持的模型

`--model` 支持以下完整的序列转换模型：

- `vanilla_gru`：使用 `torch.nn.GRU` 的无注意力 Encoder-Decoder
- `vanilla_rnn`：使用 `torch.nn.RNN`（tanh）的无注意力 Encoder-Decoder
- `bahdanau`：使用加性注意力的 GRU Decoder
- `dot_product_attention`：使用缩放点积注意力的 GRU Decoder
- `multi_head_attention`：使用多头缩放点积注意力的 GRU Decoder

`--attention-heads` 控制多头注意力的 head 数，默认值为 4；其他模型会记录该配置但不使用它。

## 自动运行实验

在项目根目录运行：

```bash
.venv/bin/python -m date2date.experiment \
  --model multi_head_attention \
  --hidden-size 256 \
  --attention-heads 4 \
  --lr 0.001 \
  --train-size 1000 \
  --test-size 100 \
  --epochs 10 \
  --seed 42 \
  --notes "Multi-Head Attention 基准实验"
```

命令会自动复用固定测试集，并保存配置、指标、固定样例预测、checkpoint 和 `experiments/experiments.csv`。训练成功后，README 中的实验结果表也会同步更新。

已有实验记录时，也可以只刷新 README：

```bash
.venv/bin/python -m date2date.report
```

本项目从 **Encoder-Decoder RNN** 开始，逐步演进到：

1. Encoder-Decoder RNN
2. Encoder-Decoder RNN + Attention
3. Transformer Encoder-Decoder
4. Decoder-only GPT 风格模型

项目约束：

- 只使用 PyTorch 实现模型、训练循环和评估流程。
- 自己构造训练数据。
- 自己完成训练。
- 自己设计评估指标并分析结果。
- 不使用 Hugging Face Transformers、PyTorch Lightning 或任何预训练模型。

希望通过这个项目观察不同模型在字符级序列转换、位置重排、条件生成和泛化能力上的表现差异。