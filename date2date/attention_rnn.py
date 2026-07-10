"""Bahdanau Attention（加性注意力）及其 Decoder。"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

try:
    from .data import SOS_token
except ImportError:
    from data import SOS_token


class BahdanauAttention(nn.Module):
    # key_size,
    def __init__(self, query_size, key_size, hidden_size) -> None:
        super().__init__()
        self.wq = nn.Linear(query_size, hidden_size)
        self.wk = nn.Linear(key_size, hidden_size)
        self.wv = nn.Linear(hidden_size, 1)

    # quries.shape(batch_size, len, query_size)
    # keys.shape(batch_size, len, key_size)
    # values.shape(batch_size, len,  value_size)
    def forward(self, queries, keys, values):
        q = self.wq(queries)
        k = self.wk(keys)
        t = torch.tanh(q + k)
        v = self.wv(t)
        self.attention_weights = nn.functional.softmax(v, dim=1)
        preds = (self.attention_weights * values).sum(dim=1)
        return preds


class DecoderRNN_WithAttention(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention = BahdanauAttention(self.hidden_size, self.hidden_size, self.hidden_size)
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.GRU(hidden_size*2, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_output: Tensor, target_tensor: Optional[Tensor] = None, max_len: int = 20):
        # encoder_output 是encoder所有时间步最后一层的输出的 hiddeng向量
        # encoder_output.shape (batch_size, len, dierction*hidden)
        # 传入 attention 的keys和values 都是 encoder_output
        # 传入 attention 的query是 分别是 encoder_hidden（只在第一次用） 和 decode的 t-1步hidden，
        # target_tensor.shape is (batch_size, seq_len)
        # foward.shape is (batch_size, seq_len,vocab_size)
        batch_size = encoder_output.shape[0]
        hidden = encoder_output[:,-1:].permute(1, 0, 2)
        outputs = []
        # input_token的格式是 (batch_size,seq_len)
        input_token = torch.full(
            (batch_size, 1), SOS_token, dtype=torch.long, device=encoder_output.device
        )  # 设置起始词元
        loop_len = max_len
        if target_tensor is not None:
            loop_len = target_tensor.shape[1]

        for i in range(loop_len):
            logits, hidden, pred_token = self.forward_step(hidden, input_token, encoder_output)
            outputs.append(logits)

            if target_tensor is not None:
                input_token = target_tensor[:,i].unsqueeze(1)
            else :
                input_token = pred_token.detach()
        decoder_outputs = torch.stack(tensors=outputs, dim=1)
        return decoder_outputs, hidden

    def forward_step(self, hidden, input_token, encoder_output):
        query = hidden[-1].unsqueeze(1)
        embedded = self.embedding(input_token)
        context = self.attention(query, encoder_output, encoder_output)
        # rnn的hidden输入是(batch_size,seq_len,hidden)
        context = context.unsqueeze(1)
        rnn_input = torch.cat([embedded, context], dim=-1)
        # rnn中输入的hidden不受 batch_first=True的控制，其shape仍然为(direction*num,batch_size,hidden_size)
        output, hidden = self.rnn(rnn_input, hidden)
        # output.shape 是 (batch_size,seq_len,direction*hidden_size)
        logits = self.out(hidden[-1]) # hidden[-1]表示最后一个num
        pred_token = logits.argmax(dim = -1,keepdim= True)
        return logits, hidden, pred_token
