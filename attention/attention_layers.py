"""Reusable attention layers and helpers for the attention notebooks."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


class NWKernelRegression(nn.Module):
    """Parametric Nadaraya-Watson kernel regression.

    Expected shapes:
        queries: (batch,)
        keys: (batch, num_kv)
        values: (batch, num_kv)
    """

    def __init__(self, init_weight: float | None = None) -> None:
        super().__init__()
        if init_weight is None:
            weight = torch.rand(1)
        else:
            weight = torch.tensor([init_weight], dtype=torch.float32)
        self.w = nn.Parameter(weight)
        self.attention_weights: torch.Tensor | None = None

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        queries = queries.reshape(-1, 1)
        scores = -((queries - keys) * self.w) ** 2 / 2
        self.attention_weights = F.softmax(scores, dim=1)
        return (self.attention_weights * values).sum(dim=1)


class AdditiveAttention(nn.Module):
    # key_size, 
    def __init__(self, query_size, key_size, hidden_size) -> None:
        super().__init__()
        self.wq = nn.Linear(query_size, hidden_size)
        self.wk = nn.Linear(key_size, hidden_size)
        self.wv = nn.Linear(hidden_size, 1)

    # quries.shape(len, query_size)
    # keys.shape(len, len - 1, key_size)
    # values.shape(len, len - 1, value_size)
    def forward(self, queries, keys, values):
        q = self.wq(queries)
        k = self.wk(keys)
        t = torch.tanh(q.unsqueeze(1) + k)
        v = self.wv(t)
        self.attention_weights = nn.functional.softmax(v, dim=1)
        preds = (self.attention_weights * values).sum(dim=1)

        return preds

class ScalarAdditiveAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = AdditiveAttention(
            query_size=1,
            key_size=1,
            hidden_size=hidden_size,
        )

    def forward(self, queries, keys, values):
        queries = queries.reshape(-1, 1)
        keys = keys.unsqueeze(-1)
        values = values.unsqueeze(-1)

        output = self.attention(queries, keys, values)
        self.attention_weights = self.attention.attention_weights.squeeze(-1)
        return output.reshape(-1)

class DotProductAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    # queries.shape(batch, len_q, h)
    # keys.shape(batch, len_k, h)
    # values.shape(batch, len_k, value_size)
    def forward(self, queries, keys, values):
        d = queries.shape[-1]

        # scores.shape (batch, len_q, len_k)
        scores = torch.bmm( queries, keys.transpose(1,2)) /  (d**0.5)

        # attention_weights.shape (batch, len_q, len_k)
        self.attention_weights = nn.functional.softmax(scores,dim = -1)
        preds = torch.bmm(self.attention_weights, values)
        return preds


@dataclass
class KernelRegressionData:
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_test: torch.Tensor
    y_truth: torch.Tensor

    @property
    def n_train(self) -> int:
        return len(self.x_train)

    @property
    def n_test(self) -> int:
        return len(self.x_test)


def target_function(x: torch.Tensor) -> torch.Tensor:
    return 2 * torch.sin(x) + x**0.8


def make_kernel_regression_data(
    n_train: int = 50,
    x_max: float = 5.0,
    test_step: float = 0.1,
    noise_std: float = 0.5,
    seed: int | None = 1,
) -> KernelRegressionData:
    if seed is not None:
        torch.manual_seed(seed)

    x_train, _ = torch.sort(torch.rand(n_train) * x_max)
    y_train = target_function(x_train) + torch.normal(0.0, noise_std, (n_train,))
    x_test = torch.arange(0, x_max, test_step)
    y_truth = target_function(x_test)
    return KernelRegressionData(x_train=x_train, y_train=y_train, x_test=x_test, y_truth=y_truth)


def make_leave_one_out_pairs(x_train: torch.Tensor, y_train: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n_train = len(x_train)
    mask = (1 - torch.eye(n_train, device=x_train.device)).bool()
    keys = x_train.repeat(n_train, 1)[mask].reshape(n_train, -1)
    values = y_train.repeat(n_train, 1)[mask].reshape(n_train, -1)
    return keys, values


def make_test_pairs(x_train: torch.Tensor, y_train: torch.Tensor, n_test: int) -> tuple[torch.Tensor, torch.Tensor]:
    keys = x_train.repeat(n_test, 1)
    values = y_train.repeat(n_test, 1)
    return keys, values
