"""字符表、张量转换工具、数据生成。"""

import random

import torch

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 本实验室字符级建模，词表中每个字符都是一个token。
chars = "0123456789-/"
# - `<SOS>`：decoder 生成时的起始 token；
# - `<EOS>`：target 序列结束标记。
special_tokens = ["<SOS>", "<EOS>"]

itos = special_tokens + list(chars)
stoi = {ch: i for i, ch in enumerate(itos)}

SOS_token = stoi["<SOS>"]
EOS_token = stoi["<EOS>"]

vocab_size = len(itos)


def tensor_from_string(s, add_eos=False):
    ids = [stoi[ch] for ch in s]
    if add_eos:
        ids.append(EOS_token)

    return torch.tensor(ids, dtype=torch.long, device=DEVICE).unsqueeze(0)


def string_from_tensor(tensor):
    chars = []
    tensor = tensor.squeeze(0)
    for token in tensor:
        idx = token.item()
        if idx == EOS_token:
            break
        if idx == SOS_token:
            continue
        chars.append(itos[idx])
    return "".join(chars)


# 生成的样本符合
# 输入长度固定为 10：YYYY-MM-DD。日和月都是2位，且日以28天计数
# 输出长度固定为 10：DD/MM/YYYY
# 再加 EOS 后 target 长度为 11。
def make_sample(TRAIN_SIZE):
    ep = TRAIN_SIZE
    inputs = []
    targets = []
    for i in range(ep):
        year = random.randint(1950, 2050)
        day = random.randint(1, 28)
        month = random.randint(1, 12)
        input = f"{year}-{month:02d}-{day:02d}"
        inputs.append(input)
        target = f"{day:02d}/{month:02d}/{year}"
        targets.append(target)
    return inputs, targets


fixed_samples = [
    ("2002-01-23", "23/01/2002"),
    ("1999-12-08", "08/12/1999"),
    ("2020-03-07", "07/03/2020"),
    ("1950-11-28", "28/11/1950"),
    ("2048-06-15", "15/06/2048"),
]
